"""Attention layers used by CMS-Mamba."""

import torch
from torch import nn


class Attention(nn.Module):
    """Conventional multi-head attention retained for internal compatibility."""

    def __init__(self, dim, heads=8, dim_head=None, dropout=0.0):
        super().__init__()
        if dim_head is None:
            if dim % heads:
                raise ValueError("attention dimension must be divisible by heads")
            dim_head = dim // heads
        self.heads = int(heads)
        self.head_dim = int(dim_head)
        self.scale = self.head_dim**-0.5
        inner_dim = self.heads * self.head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def _split(self, tensor):
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value):
        q = self._split(self.to_q(query))
        k = self._split(self.to_k(key))
        v = self._split(self.to_v(value))
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attended = torch.matmul(torch.softmax(scores, dim=-1), v)
        attended = attended.transpose(1, 2).contiguous().view(
            query.shape[0], query.shape[1], -1
        )
        return self.to_out(attended)


class RotaryKeyCrossAttention(nn.Module):
    """Text-query cross-attention with RoPE applied to modal keys only."""

    def __init__(
        self,
        query_dim,
        modal_dim,
        heads,
        head_dim=None,
        dropout=0.0,
        rope_base=10000.0,
    ):
        super().__init__()
        self.heads = int(heads)
        if head_dim is None:
            if int(query_dim) % self.heads:
                raise ValueError("query dimension must be divisible by attention heads")
            head_dim = int(query_dim) // self.heads
        self.head_dim = int(head_dim)
        if self.head_dim % 2:
            raise ValueError(
                f"RoPE requires an even head dimension, got {self.head_dim}"
            )
        if float(rope_base) <= 0:
            raise ValueError(f"RoPE base must be positive, got {rope_base}")

        inner_dim = self.heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(modal_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(modal_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )
        inverse_frequency = 1.0 / (
            float(rope_base)
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("rope_inverse_frequency", inverse_frequency, persistent=False)

    def _split_heads(self, tensor):
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def _apply_rope(self, keys):
        positions = torch.arange(
            keys.shape[-2],
            device=keys.device,
            dtype=self.rope_inverse_frequency.dtype,
        )
        angles = torch.outer(positions, self.rope_inverse_frequency)
        cosine = torch.repeat_interleave(angles.cos(), 2, dim=-1)[None, None]
        sine = torch.repeat_interleave(angles.sin(), 2, dim=-1)[None, None]
        cosine = cosine.to(dtype=keys.dtype)
        sine = sine.to(dtype=keys.dtype)
        first = keys[..., 0::2]
        second = keys[..., 1::2]
        rotated = torch.stack((-second, first), dim=-1).flatten(-2)
        return keys * cosine + rotated * sine

    def project_qkv(self, query, modal, apply_rope=True):
        if query.ndim != 3 or modal.ndim != 3:
            raise ValueError("cross-attention inputs must have shape [B, L, D]")
        if query.shape[:2] != modal.shape[:2]:
            raise ValueError("text query and modal sequence must share batch and length")
        query_projection = self._split_heads(self.to_q(query))
        key_projection = self._split_heads(self.to_k(modal))
        value_projection = self._split_heads(self.to_v(modal))
        if apply_rope:
            key_projection = self._apply_rope(key_projection)
        return query_projection, key_projection, value_projection

    def forward(self, query, modal):
        q, k, v = self.project_qkv(query, modal, apply_rope=True)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attended = torch.matmul(torch.softmax(scores, dim=-1), v)
        attended = attended.transpose(1, 2).contiguous().view(
            query.shape[0], query.shape[1], -1
        )
        return query + self.to_out(attended)
