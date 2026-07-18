"""Text-aware modality mixing with shared feature/mask alignment."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


__all__ = ["CTCModule", "EnhanceSubNet", "TMMOutput"]


@dataclass
class TMMOutput:
    text: torch.Tensor
    vision: torch.Tensor
    audio: torch.Tensor
    text_missing: torch.Tensor
    vision_missing: torch.Tensor
    audio_missing: torch.Tensor
    vision_alignment: torch.Tensor
    audio_alignment: torch.Tensor


def _masked_alignment(logits, valid_mask):
    """Normalize text-to-source alignment rows over valid source frames."""

    if logits.ndim != 3:
        raise ValueError(f"alignment logits must have shape [B, T, S], got {logits.shape}")
    if valid_mask.shape != (logits.shape[0], logits.shape[2]):
        raise ValueError(
            "alignment validity mask must match the batch and source dimensions"
        )
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    if torch.any(valid.sum(dim=-1) == 0):
        raise ValueError("alignment source sequence has no valid positions")
    masked = logits.masked_fill(
        ~valid[:, None, :],
        torch.finfo(logits.dtype).min,
    )
    alignment = torch.softmax(masked, dim=-1)
    if not torch.isfinite(alignment).all():
        raise ValueError("alignment matrix contains non-finite values")
    return alignment


class CTCModule(nn.Module):
    def __init__(self, in_dim, out_seq_len):
        super().__init__()
        self.pred_output_position_inclu_blank = nn.LSTM(
            in_dim,
            out_seq_len + 1,
            num_layers=2,
            batch_first=True,
        )
        self.out_seq_len = int(out_seq_len)

    def forward(self, x, valid_mask):
        if x.ndim != 3:
            raise ValueError(f"CTC input must have shape [B, S, D], got {x.shape}")
        position_logits, _ = self.pred_output_position_inclu_blank(x)
        # Index zero is the CTC-inspired blank channel. The remaining logits
        # describe each text target position and are normalized over source time.
        target_logits = position_logits[:, :, 1:].transpose(1, 2)
        alignment = _masked_alignment(target_logits, valid_mask)
        return torch.bmm(alignment, x), alignment


class EnhanceSubNet(nn.Module):
    """Align audio/vision and their masks to the text timeline."""

    def __init__(self, input_length, input_dim, hidden_dim):
        super().__init__()
        seq_len_t, _, _ = input_length
        in_dim_t, in_dim_v, in_dim_a = input_dim
        self.dst_len = int(seq_len_t)
        self.dst_dim = int(hidden_dim)
        self.eps = 1e-9

        self.ctc_vt = CTCModule(in_dim_v, self.dst_len)
        self.logit_scale_vt = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.ctc_at = CTCModule(in_dim_a, self.dst_len)
        self.logit_scale_at = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.proj_a = nn.Sequential(
            nn.LayerNorm(in_dim_a, eps=1e-6),
            nn.Linear(in_dim_a, self.dst_dim),
            nn.LayerNorm(self.dst_dim, eps=1e-6),
        )
        self.proj_v = nn.Sequential(
            nn.LayerNorm(in_dim_v, eps=1e-6),
            nn.Linear(in_dim_v, self.dst_dim),
            nn.LayerNorm(self.dst_dim, eps=1e-6),
        )
        self.proj_t = nn.Sequential(
            nn.LayerNorm(in_dim_t, eps=1e-6),
            nn.Linear(in_dim_t, self.dst_dim),
        )

    def get_seq_len(self):
        return self.dst_len

    def _normalize(self, features):
        norm = features.norm(dim=-1, keepdim=True)
        return features / torch.clamp_min(norm, self.eps)

    def forward(
        self,
        text_x,
        video_x,
        audio_x,
        text_missing,
        video_missing,
        audio_missing,
        video_valid,
        audio_valid,
    ):
        if text_x.shape[1] != self.dst_len:
            raise ValueError(
                f"text sequence length must be {self.dst_len}, got {text_x.shape[1]}"
            )
        if text_missing.shape != text_x.shape[:2]:
            raise ValueError("text missing mask must match the text sequence")
        if video_missing.shape != video_x.shape[:2]:
            raise ValueError("vision missing mask must match the vision sequence")
        if audio_missing.shape != audio_x.shape[:2]:
            raise ValueError("audio missing mask must match the audio sequence")

        pseudo_video, vision_alignment = self.ctc_vt(video_x, video_valid)
        pseudo_audio, audio_alignment = self.ctc_at(audio_x, audio_valid)

        vision_missing_aligned = torch.bmm(
            vision_alignment,
            video_missing.to(vision_alignment.dtype).unsqueeze(-1),
        ).squeeze(-1)
        audio_missing_aligned = torch.bmm(
            audio_alignment,
            audio_missing.to(audio_alignment.dtype).unsqueeze(-1),
        ).squeeze(-1)
        vision_missing_aligned = vision_missing_aligned.clamp(0.0, 1.0)
        audio_missing_aligned = audio_missing_aligned.clamp(0.0, 1.0)

        vision_common = self.proj_v(pseudo_video)
        audio_common = self.proj_a(pseudo_audio)
        text_common = self.proj_t(text_x)

        vision_logits = self.logit_scale_vt.exp() * torch.bmm(
            self._normalize(vision_common),
            self._normalize(text_common).transpose(1, 2),
        )
        vision_weights = vision_logits.softmax(dim=-1)
        vision_gate = (vision_weights > (1 / self.dst_len)).to(vision_weights.dtype)

        audio_logits = self.logit_scale_at.exp() * torch.bmm(
            self._normalize(audio_common),
            self._normalize(text_common).transpose(1, 2),
        )
        audio_weights = audio_logits.softmax(dim=-1)
        audio_gate = (audio_weights > (1 / self.dst_len)).to(audio_weights.dtype)

        vision_out = vision_common + torch.bmm(
            vision_gate * vision_weights,
            text_common,
        )
        audio_out = audio_common + torch.bmm(
            audio_gate * audio_weights,
            text_common,
        )
        return TMMOutput(
            text=text_common,
            vision=vision_out,
            audio=audio_out,
            text_missing=text_missing.to(text_common.dtype),
            vision_missing=vision_missing_aligned,
            audio_missing=audio_missing_aligned,
            vision_alignment=vision_alignment,
            audio_alignment=audio_alignment,
        )
