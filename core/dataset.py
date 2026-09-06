"""Dataset loading with explicit, reproducible multimodal missingness."""

import pickle
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.missingness import (
    automatic_continuous_missingness,
    automatic_text_missingness,
    corrupt_continuous,
    corrupt_text,
    evaluation_missingness,
    fit_continuous_feature_statistics,
    standardize_received_continuous,
    training_missingness,
)


__all__ = ["MMDataset", "MMDataLoader", "MMDataEvaluationLoader"]


class MMDataset(Dataset):
    """CMU-MOSI/MOSEI/CH-SIMS features with paper-aligned corruption.

    Reference masks record synthetic corruption, while automatic masks are
    derived from the received corrupted inputs.  Validity masks are returned
    separately so padding never becomes a missing observation.
    """

    def __init__(self, args, mode="train"):
        self.mode = str(mode)
        self.train_mode = args["base"]["train_mode"]
        self.dataset_name = args["dataset"]["datasetName"]
        self.data_path = args["dataset"]["dataPath"]
        self.missing_seed = int(args["base"].get("seed", 1111))
        self.epoch = 0
        estimator = args["base"].get("automatic_missingness", {})
        self.audio_threshold = float(estimator.get("audio_threshold", 0.18))
        self.vision_threshold = float(estimator.get("vision_threshold", 0.21))
        if self.audio_threshold < 0.0 or self.vision_threshold < 0.0:
            raise ValueError("automatic missingness thresholds must be non-negative")

        self.eval_pattern = str(
            args["base"].get("missing_pattern", "independent")
        )
        self.eval_rates = self._resolve_eval_missing_rates(args["base"])
        self.eval_block_rate = float(args["base"].get("block_rate", 0.0))

        with open(self.data_path, "rb") as handle:
            payload = pickle.load(handle)
        if self.mode not in payload:
            raise KeyError(f"dataset split {self.mode!r} is not present in {self.data_path}")
        data = payload[self.mode]

        self.text = data["text_bert"].astype(np.float32)
        self.vision = data["vision"].astype(np.float32)
        self.audio = data["audio"].astype(np.float32)
        self.audio_lengths = np.asarray(data["audio_lengths"], dtype=np.int64)
        self.vision_lengths = np.asarray(data["vision_lengths"], dtype=np.int64)
        self.raw_text = data.get("raw_text", np.array([""] * len(self.text)))
        self.ids = data.get("id", np.arange(len(self.text)))

        label_key = f"{self.train_mode}_labels"
        self.labels = {"M": data[label_key].astype(np.float32)}
        if self.dataset_name == "sims":
            for modality in "TAV":
                modality_key = f"{label_key}_{modality}"
                if modality_key in data:
                    self.labels[modality] = data[modality_key].astype(np.float32)

        self._validate_lengths()
        self._validate_feature_dimensions(args)
        self.audio_statistics, self.vision_statistics = self._fit_training_statistics(
            payload
        )

    def _fit_training_statistics(self, payload):
        if "train" not in payload:
            raise KeyError("automatic missingness requires the training split")
        train_data = payload["train"]
        for key in ("audio", "vision", "audio_lengths", "vision_lengths"):
            if key not in train_data:
                raise KeyError(
                    f"automatic missingness requires train split field {key!r}"
                )
        return (
            fit_continuous_feature_statistics(
                train_data["audio"],
                train_data["audio_lengths"],
            ),
            fit_continuous_feature_statistics(
                train_data["vision"],
                train_data["vision_lengths"],
            ),
        )

    @staticmethod
    def _normalize_modalities(modalities) -> set[str]:
        if modalities is None:
            return {"A", "V"}
        if isinstance(modalities, str):
            value = modalities.replace(",", "").replace(" ", "").upper()
            return set(value.replace("L", "T"))
        normalized = set()
        for modality in modalities:
            value = str(modality).strip().upper()
            aliases = {
                "TEXT": "T",
                "LANGUAGE": "T",
                "L": "T",
                "AUDIO": "A",
                "VISION": "V",
                "VIDEO": "V",
            }
            normalized.add(aliases.get(value, value[:1]))
        return normalized

    @classmethod
    def _resolve_eval_missing_rates(cls, base_args) -> tuple[float, float, float]:
        configured = base_args.get(
            "missing_rate_eval_test",
            base_args.get("missing_rate", 0.0),
        )
        if isinstance(configured, dict):
            rates = (
                configured.get("text", configured.get("language", 0.0)),
                configured.get("audio", 0.0),
                configured.get("vision", configured.get("video", 0.0)),
            )
        elif isinstance(configured, (list, tuple, np.ndarray)):
            if len(configured) != 3:
                raise ValueError(
                    "missing_rate_eval_test must contain text, audio, and vision rates"
                )
            rates = tuple(configured)
        else:
            rate = float(configured)
            modalities = cls._normalize_modalities(
                base_args.get("eval_missing_modalities", "AV")
            )
            rates = (
                rate if "T" in modalities else 0.0,
                rate if "A" in modalities else 0.0,
                rate if "V" in modalities else 0.0,
            )
        result = tuple(float(value) for value in rates)
        if any(value < 0.0 or value > 1.0 for value in result):
            raise ValueError(f"missing rates must be in [0, 1], got {result}")
        return result

    def _validate_lengths(self):
        if len(self.text) != len(self.audio) or len(self.text) != len(self.vision):
            raise ValueError("text, audio, and vision splits have different sample counts")
        if np.any(self.audio_lengths < 0) or np.any(
            self.audio_lengths > self.audio.shape[1]
        ):
            raise ValueError("audio lengths exceed the configured audio sequence")
        if np.any(self.vision_lengths < 0) or np.any(
            self.vision_lengths > self.vision.shape[1]
        ):
            raise ValueError("vision lengths exceed the configured vision sequence")

    def _validate_feature_dimensions(self, args):
        configured = args.get("model", {}).get("tmm", {}).get("input_dim")
        if configured is None:
            return
        expected_vision = int(configured[1])
        expected_audio = int(configured[2])
        actual_vision = int(self.vision.shape[-1])
        actual_audio = int(self.audio.shape[-1])
        if actual_vision != expected_vision:
            raise ValueError(
                f"vision feature dimension mismatch: expected {expected_vision}, "
                f"got {actual_vision}"
            )
        if actual_audio != expected_audio:
            raise ValueError(
                f"audio feature dimension mismatch: expected {expected_audio}, "
                f"got {actual_audio}"
            )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def set_evaluation_corruption(
        self,
        pattern,
        rates: Sequence[float],
        seed,
        block_rate=0.0,
    ):
        if self.mode == "train":
            raise RuntimeError("evaluation corruption cannot be assigned to training data")
        if len(rates) != 3:
            raise ValueError("rates must contain text, audio, and vision values")
        self.eval_pattern = str(pattern)
        self.eval_rates = tuple(float(value) for value in rates)
        self.missing_seed = int(seed)
        self.eval_block_rate = float(block_rate)

    def _valid_masks(self, index):
        text_valid = self.text[index, 1, :].astype(bool)
        audio_valid = np.arange(self.audio.shape[1]) < int(self.audio_lengths[index])
        vision_valid = np.arange(self.vision.shape[1]) < int(
            self.vision_lengths[index]
        )
        return {
            "text": text_valid,
            "audio": audio_valid,
            "vision": vision_valid,
        }

    @staticmethod
    def _text_eligible(text_valid):
        eligible = np.array(text_valid, copy=True, dtype=bool)
        valid_positions = np.flatnonzero(text_valid)
        if len(valid_positions):
            eligible[valid_positions[0]] = False
            eligible[valid_positions[-1]] = False
        return eligible

    def __getitem__(self, index):
        valid = self._valid_masks(index)
        text_eligible = self._text_eligible(valid["text"])
        if self.mode == "train":
            missing = training_missingness(
                valid,
                text_eligible,
                self.missing_seed,
                self.epoch,
                index,
            )
        else:
            missing = evaluation_missingness(
                valid,
                text_eligible,
                self.eval_pattern,
                self.eval_rates,
                self.missing_seed,
                index,
                self.eval_block_rate,
            )

        input_ids = corrupt_text(self.text[index, 0, :], missing.text)
        text_auto_missing = automatic_text_missingness(input_ids, valid["text"])
        text_m = np.stack(
            (input_ids, self.text[index, 1, :], self.text[index, 2, :])
        ).astype(np.float32)

        audio_received = corrupt_continuous(self.audio[index], missing.audio)
        audio_m, audio_direct_missing = standardize_received_continuous(
            audio_received,
            valid["audio"],
            self.audio_statistics,
        )
        audio_auto_missing = automatic_continuous_missingness(
            audio_m,
            valid["audio"],
            audio_direct_missing,
            self.audio_threshold,
        )

        vision_received = corrupt_continuous(self.vision[index], missing.vision)
        vision_m, vision_direct_missing = standardize_received_continuous(
            vision_received,
            valid["vision"],
            self.vision_statistics,
        )
        vision_auto_missing = automatic_continuous_missingness(
            vision_m,
            valid["vision"],
            vision_direct_missing,
            self.vision_threshold,
        )

        labels = {
            name: torch.from_numpy(np.asarray(values[index]).reshape(-1)).float()
            for name, values in self.labels.items()
        }
        return {
            "text": torch.from_numpy(self.text[index]).float(),
            "text_m": torch.from_numpy(text_m).float(),
            "text_valid_mask": torch.from_numpy(valid["text"].astype(np.float32)),
            "text_missing_mask": torch.from_numpy(missing.text.astype(np.float32)),
            "text_auto_missing_mask": torch.from_numpy(
                text_auto_missing.astype(np.float32)
            ),
            "audio": torch.from_numpy(self.audio[index]).float(),
            "audio_m": torch.from_numpy(audio_m).float(),
            "audio_valid_mask": torch.from_numpy(valid["audio"].astype(np.float32)),
            "audio_missing_mask": torch.from_numpy(missing.audio.astype(np.float32)),
            "audio_auto_missing_mask": torch.from_numpy(
                audio_auto_missing.astype(np.float32)
            ),
            "vision": torch.from_numpy(self.vision[index]).float(),
            "vision_m": torch.from_numpy(vision_m).float(),
            "vision_valid_mask": torch.from_numpy(
                valid["vision"].astype(np.float32)
            ),
            "vision_missing_mask": torch.from_numpy(
                missing.vision.astype(np.float32)
            ),
            "vision_auto_missing_mask": torch.from_numpy(
                vision_auto_missing.astype(np.float32)
            ),
            "requested_missing_rate": torch.tensor(
                missing.eta, dtype=torch.float32
            ),
            "index": index,
            "id": self.ids[index],
            "labels": labels,
        }

    def __len__(self):
        return len(self.labels["M"])


def _loader(dataset, args, shuffle):
    return DataLoader(
        dataset,
        batch_size=int(args["base"]["batch_size"]),
        num_workers=int(args["base"].get("num_workers", 0)),
        shuffle=bool(shuffle),
    )


def MMDataLoader(args):
    datasets = {
        split: MMDataset(args, mode=split) for split in ("train", "valid", "test")
    }
    return {
        split: _loader(dataset, args, split == "train")
        for split, dataset in datasets.items()
    }


def MMDataEvaluationLoader(args, mode="test"):
    return _loader(MMDataset(args, mode=mode), args, False)
