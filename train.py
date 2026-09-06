"""Train CMS-Mamba using the paper's validation-only selection protocol."""

import argparse
import os

import numpy as np
import torch
import yaml

from core.dataset import MMDataLoader
from core.losses import MultimodalLoss
from core.metric import MetricsTop
from core.scheduler import get_scheduler
from core.utils import save_model, setup_seed
from core.validation import (
    ValidationCheckpointSelector,
    regression_mae,
    validation_grid,
)
from models.TFMamba import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/train_sims.yaml")
    parser.add_argument("--seed", type=int, default=-1)
    return parser.parse_args()


def batch_to_model_inputs(data, device, indicator_source="automatic"):
    incomplete_input = (
        data["vision_m"].to(device),
        data["audio_m"].to(device),
        data["text_m"].to(device),
    )
    automatic_masks = (
        data["text_auto_missing_mask"].to(device),
        data["audio_auto_missing_mask"].to(device),
        data["vision_auto_missing_mask"].to(device),
    )
    reference_masks = (
        data["text_missing_mask"].to(device),
        data["audio_missing_mask"].to(device),
        data["vision_missing_mask"].to(device),
    )
    valid_masks = (
        data["text_valid_mask"].to(device),
        data["audio_valid_mask"].to(device),
        data["vision_valid_mask"].to(device),
    )
    normalized_source = str(indicator_source).lower()
    if normalized_source == "automatic":
        conditioning_masks = automatic_masks
    elif normalized_source == "reference":
        conditioning_masks = reference_masks
    else:
        raise ValueError(
            "mcssm_indicator_source must be 'automatic' or 'reference'"
        )
    labels = {
        "sentiment_labels": data["labels"]["M"].to(device),
        "text_reconstruction_mask": reference_masks[0],
    }
    return (
        incomplete_input,
        automatic_masks,
        conditioning_masks,
        valid_masks,
        data["text"].to(device),
        labels,
    )


def train_epoch(model, train_loader, optimizer, loss_fn, device, indicator_source):
    model.train()
    losses = []
    predictions, targets = [], []

    for data in train_loader:
        (
            incomplete_input,
            automatic_masks,
            conditioning_masks,
            valid_masks,
            complete_text,
            labels,
        ) = batch_to_model_inputs(data, device, indicator_source)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            incomplete_input,
            automatic_masks,
            valid_masks,
            conditioning_masks=conditioning_masks,
            complete_text=complete_text,
        )
        loss = loss_fn(output, labels)["loss"]
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach()))
        predictions.append(output["sentiment_preds"].detach().cpu())
        targets.append(labels["sentiment_labels"].detach().cpu())

    if not predictions:
        raise RuntimeError("training loader produced no batches")
    return float(np.mean(losses)), torch.cat(predictions), torch.cat(targets)


@torch.no_grad()
def evaluate(model, eval_loader, device, metrics, indicator_source):
    model.eval()
    predictions, targets = [], []

    for data in eval_loader:
        (
            incomplete_input,
            automatic_masks,
            conditioning_masks,
            valid_masks,
            _,
            labels,
        ) = batch_to_model_inputs(data, device, indicator_source)
        output = model(
            incomplete_input,
            automatic_masks,
            valid_masks,
            conditioning_masks=conditioning_masks,
        )
        predictions.append(output["sentiment_preds"].cpu())
        targets.append(labels["sentiment_labels"].cpu())

    if not predictions:
        raise RuntimeError("evaluation loader produced no batches")
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    return metrics(predictions, targets), regression_mae(predictions, targets)


def evaluate_validation_grid(model, valid_loader, device, metrics, base_args):
    rates = base_args.get("validation_missing_rates", [0.0, 0.1, 0.3, 0.5, 0.7])
    seeds = base_args.get("validation_mask_seeds", [1111, 2222, 3333])
    conditions = validation_grid(rates, seeds)
    condition_results = []

    for missing_rate, mask_seed in conditions:
        valid_loader.dataset.set_evaluation_corruption(
            pattern="independent",
            rates=(missing_rate, missing_rate, missing_rate),
            seed=mask_seed,
        )
        result, selection_mae = evaluate(
            model,
            valid_loader,
            device,
            metrics,
            base_args.get("mcssm_indicator_source", "automatic"),
        )
        condition_results.append(
            {
                "missing_rate": missing_rate,
                "mask_seed": mask_seed,
                "MAE": selection_mae,
            }
        )

    mean_mae = float(np.mean([result["MAE"] for result in condition_results]))
    return mean_mae, condition_results


def main():
    options = parse_args()
    with open(options.config_file, encoding="utf-8") as handle:
        args = yaml.safe_load(handle)

    seed = int(args["base"]["seed"] if options.seed == -1 else options.seed)
    setup_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_root = os.path.join("ckpt", args["dataset"]["datasetName"])
    os.makedirs(checkpoint_root, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_root, f"best_validation_MAE_{seed}.pth"
    )

    model = build_model(args).to(device)
    data_loaders = MMDataLoader(args)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args["base"]["lr"]),
        weight_decay=float(args["base"]["weight_decay"]),
    )
    scheduler = get_scheduler(optimizer, args)
    loss_fn = MultimodalLoss(args)
    metrics = MetricsTop(args["base"]["train_mode"]).getMetics(
        args["dataset"]["datasetName"]
    )
    selector = ValidationCheckpointSelector()
    indicator_source = args["base"].get("mcssm_indicator_source", "automatic")

    for epoch in range(1, int(args["base"]["n_epochs"]) + 1):
        data_loaders["train"].dataset.set_epoch(epoch)
        loss, predictions, targets = train_epoch(
            model,
            data_loaders["train"],
            optimizer,
            loss_fn,
            device,
            indicator_source,
        )
        train_metrics = metrics(predictions, targets)
        print(f"Epoch {epoch} train MSE: {loss:.6f}; metrics: {train_metrics}")

        if bool(args["base"].get("do_validation", True)):
            mean_mae, validation_results = evaluate_validation_grid(
                model,
                data_loaders["valid"],
                device,
                metrics,
                args["base"],
            )
            print(
                f"Epoch {epoch} perturbed-validation mean MAE: {mean_mae:.6f} "
                f"over {len(validation_results)} conditions"
            )
            if selector.update(mean_mae, epoch):
                save_model(checkpoint_path, epoch - 1, model, optimizer)
                print(f"Saved validation-selected checkpoint: {checkpoint_path}")

        scheduler.step()

    if selector.best_epoch < 0:
        raise RuntimeError("no checkpoint selected: validation must be enabled")
    print(
        f"Best epoch: {selector.best_epoch}; "
        f"perturbed-validation mean MAE: {selector.best_mae:.6f}"
    )


if __name__ == "__main__":
    main()
