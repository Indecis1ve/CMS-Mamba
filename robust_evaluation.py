"""Robustness evaluation for paper-aligned CMS-Mamba checkpoints.

This entry point only evaluates an existing, architecture-compatible checkpoint.
It never trains a model or silently accepts parameters from an older architecture.
"""

import argparse
from dataclasses import dataclass, field
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from core.dataset import MMDataEvaluationLoader


try:
    from jtop import jtop  # type: ignore

    JTOP_AVAILABLE = True
except Exception:
    jtop = None
    JTOP_AVAILABLE = False


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
PATTERNS = (
    "independent",
    "continuous",
    "block",
    "mixed_burst",
    "text_missing",
    "av_missing",
    "text_heavy",
    "av_heavy",
)
NAMED_PATTERN_RATES = {
    "text_missing": (1.0, 0.0, 0.0),
    "av_missing": (0.0, 1.0, 1.0),
    "text_heavy": (0.7, 0.1, 0.1),
    "av_heavy": (0.1, 0.7, 0.7),
}


def _validate_rate(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")
    return value


def resolve_pattern_rates(
    pattern: str,
    missing_rate: float,
    text_rate: Optional[float] = None,
    audio_rate: Optional[float] = None,
    vision_rate: Optional[float] = None,
) -> tuple[float, float, float]:
    """Resolve a robustness condition to text/audio/vision missing rates."""

    normalized = str(pattern).lower().replace("-", "_")
    if normalized not in PATTERNS:
        raise ValueError(f"unsupported missingness pattern: {pattern}")
    base_rates = NAMED_PATTERN_RATES.get(
        normalized,
        (_validate_rate("missing_rate", missing_rate),) * 3,
    )
    overrides = (text_rate, audio_rate, vision_rate)
    names = ("text_rate", "audio_rate", "vision_rate")
    return tuple(
        base if override is None else _validate_rate(name, override)
        for base, override, name in zip(base_rates, overrides, names)
    )


def dataset_pattern(pattern: str) -> str:
    normalized = str(pattern).lower().replace("-", "_")
    return "independent" if normalized in NAMED_PATTERN_RATES else normalized


@dataclass
class MissingRateTracker:
    missing: Dict[str, float] = field(
        default_factory=lambda: {"text": 0.0, "audio": 0.0, "vision": 0.0}
    )
    eligible: Dict[str, float] = field(
        default_factory=lambda: {"text": 0.0, "audio": 0.0, "vision": 0.0}
    )

    def update(self, batch_data: Dict[str, Any]):
        for modality in self.missing:
            missing = batch_data[f"{modality}_missing_mask"].to(dtype=torch.bool)
            valid = batch_data[f"{modality}_valid_mask"].to(dtype=torch.bool)
            if torch.any(missing & ~valid):
                raise RuntimeError(f"{modality} missing mask includes padding")
            self.missing[modality] += float((missing & valid).sum())
            self.eligible[modality] += float(valid.sum())

    def rates(self) -> Dict[str, float]:
        return {
            modality: (
                self.missing[modality] / self.eligible[modality]
                if self.eligible[modality]
                else 0.0
            )
            for modality in self.missing
        }


class PerformanceTracker:
    def __init__(self):
        self.latency_list: List[float] = []
        self.gpu_pwr_list: List[float] = []
        self.temp_list: List[float] = []
        self.mem_allocated_max = 0

    def update_memory(self):
        if USE_CUDA:
            self.mem_allocated_max = max(
                self.mem_allocated_max, torch.cuda.max_memory_allocated()
            )

    def update_jetson(self, jetson_obj: Any):
        if jetson_obj is None:
            return
        try:
            if not jetson_obj.ok():
                return
            power = jetson_obj.power
            if isinstance(power, dict) and "tot" in power and "avg" in power["tot"]:
                self.gpu_pwr_list.append(float(power["tot"]["avg"]) / 1000.0)
            temperature = jetson_obj.temperature
            if isinstance(temperature, dict):
                for key in ("tj", "cpu"):
                    if key in temperature and isinstance(temperature[key], dict):
                        self.temp_list.append(
                            float(temperature[key].get("temp", 0.0))
                        )
                        break
        except Exception:
            pass

    def report(
        self,
        batch_size: int,
        metrics: Dict[str, float],
        pattern: str,
        requested_rates: tuple[float, float, float],
        realized_rates: Dict[str, float],
        checkpoint_path: str,
    ):
        latency_ms = (
            float(np.mean(self.latency_list) * 1000.0)
            if self.latency_list
            else 0.0
        )
        throughput = batch_size / (latency_ms / 1000.0) if latency_ms else 0.0
        average_power = (
            float(np.mean(self.gpu_pwr_list)) if self.gpu_pwr_list else None
        )
        average_temperature = (
            float(np.mean(self.temp_list)) if self.temp_list else None
        )

        print("\n" + "=" * 72)
        print("CMS-Mamba 鲁棒性评估报告")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Pattern: {pattern}")
        print(
            "Requested missing rates (T/A/V): "
            + "/".join(f"{rate:.3f}" for rate in requested_rates)
        )
        print(
            "Realized missing rates (T/A/V): "
            + "/".join(
                f"{realized_rates[name]:.3f}"
                for name in ("text", "audio", "vision")
            )
        )
        print(f"Average batch latency: {latency_ms:.2f} ms")
        print(f"Estimated throughput: {throughput:.2f} samples/s")
        print(f"Peak CUDA memory: {self.mem_allocated_max / 1024 ** 2:.2f} MB")
        print(
            f"Average power: {average_power:.2f} W"
            if average_power is not None
            else "Average power: N/A"
        )
        print(
            f"Average temperature: {average_temperature:.2f} C"
            if average_temperature is not None
            else "Average temperature: N/A"
        )
        for name, value in metrics.items():
            print(f"{name}: {float(value):.4f}")
        print("=" * 72 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one paper-defined CMS-Mamba missingness condition."
    )
    parser.add_argument("--config_file", default="configs/eval_mosei.yaml")
    parser.add_argument(
        "--ckpt_path", default="ckpt/mosei/best_validation_MAE_1111.pth"
    )
    parser.add_argument("--pattern", choices=PATTERNS, default="continuous")
    parser.add_argument("--missing_rate", type=float, default=0.5)
    parser.add_argument("--text_rate", type=float)
    parser.add_argument("--audio_rate", type=float)
    parser.add_argument("--vision_rate", type=float)
    parser.add_argument("--block_rate", type=float, default=0.0)
    parser.add_argument("--mask_seed", type=int, default=1111)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--disable_jtop", action="store_true")
    parser.add_argument("--disable_amp", action="store_true")
    return parser.parse_args()


def load_config(
    config_file: str,
    pattern: str,
    rates: tuple[float, float, float],
    block_rate: float,
    mask_seed: int,
    num_workers: Optional[int],
):
    with open(config_file, encoding="utf-8") as handle:
        args = yaml.safe_load(handle)
    args.setdefault("base", {})
    args["base"]["missing_pattern"] = dataset_pattern(pattern)
    args["base"]["missing_rate_eval_test"] = list(rates)
    args["base"]["block_rate"] = _validate_rate("block_rate", block_rate)
    args["base"]["seed"] = int(mask_seed)
    if num_workers is not None:
        args["base"]["num_workers"] = int(num_workers)
    return args


def normalize_state_dict(state_dict: Dict[str, torch.Tensor]):
    normalized = {}
    for name, value in state_dict.items():
        if name.startswith("module."):
            name = name[7:]
        elif name.startswith("model."):
            name = name[6:]
        normalized[name] = value
    return normalized


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device=DEVICE):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    try:
        model.load_state_dict(normalize_state_dict(state_dict), strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "checkpoint is incompatible with the paper-aligned architecture; "
            "retraining is required before evaluation"
        ) from error


def model_forward(model, batch_data, disable_amp: bool):
    incomplete_input = (
        batch_data["vision_m"].to(DEVICE),
        batch_data["audio_m"].to(DEVICE),
        batch_data["text_m"].to(DEVICE),
    )
    missing_masks = (
        batch_data["text_missing_mask"].to(DEVICE),
        batch_data["audio_missing_mask"].to(DEVICE),
        batch_data["vision_missing_mask"].to(DEVICE),
    )
    valid_masks = (
        batch_data["text_valid_mask"].to(DEVICE),
        batch_data["audio_valid_mask"].to(DEVICE),
        batch_data["vision_valid_mask"].to(DEVICE),
    )
    if USE_CUDA and not disable_amp:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return model(incomplete_input, missing_masks, valid_masks)
    return model(incomplete_input, missing_masks, valid_masks)


def evaluate_one_condition():
    options = parse_args()
    rates = resolve_pattern_rates(
        options.pattern,
        options.missing_rate,
        options.text_rate,
        options.audio_rate,
        options.vision_rate,
    )
    args = load_config(
        options.config_file,
        options.pattern,
        rates,
        options.block_rate,
        options.mask_seed,
        options.num_workers,
    )

    # Lazy import keeps protocol/config checks usable without model dependencies.
    from core.metric import MetricsTop
    from models.TFMamba import build_model

    model = build_model(args).to(DEVICE)
    load_checkpoint(model, options.ckpt_path)
    model.eval()

    dataset_name = args["dataset"]["datasetName"]
    metrics = MetricsTop(args["base"]["train_mode"]).getMetics(dataset_name)
    data_loader = MMDataEvaluationLoader(args, mode="test")
    performance = PerformanceTracker()
    missingness = MissingRateTracker()
    predictions, labels = [], []

    use_jtop = JTOP_AVAILABLE and not options.disable_jtop
    jetson_context = jtop() if use_jtop else None

    def run_loop(jetson_obj=None):
        with torch.no_grad():
            for batch_index, batch_data in enumerate(data_loader):
                missingness.update(batch_data)
                target = batch_data["labels"]["M"].to(DEVICE)
                if USE_CUDA:
                    torch.cuda.reset_peak_memory_stats()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    output = model_forward(model, batch_data, options.disable_amp)
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed = start_event.elapsed_time(end_event) / 1000.0
                else:
                    start = time.perf_counter()
                    output = model_forward(model, batch_data, options.disable_amp)
                    elapsed = time.perf_counter() - start
                performance.latency_list.append(elapsed)
                performance.update_memory()
                performance.update_jetson(jetson_obj)
                predictions.append(output["sentiment_preds"].float().cpu())
                labels.append(target.float().cpu())
                if (batch_index + 1) % 10 == 0:
                    print(f"Processed {batch_index + 1} batches")

    if jetson_context is None:
        run_loop()
    else:
        with jetson_context as jetson:
            run_loop(jetson)

    if not predictions:
        raise RuntimeError("test loader produced no batches")
    results = metrics(torch.cat(predictions), torch.cat(labels))
    performance.report(
        int(args["base"]["batch_size"]),
        results,
        options.pattern,
        rates,
        missingness.rates(),
        options.ckpt_path,
    )


if __name__ == "__main__":
    evaluate_one_condition()
