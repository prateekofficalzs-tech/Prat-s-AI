"""Token embeddings and Rotary Positional Embeddings (RoPE)."""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple


class TokenEmbedding(nn.Module):
    """Token embedding lookup with dropout."""

    def __init__(self, vocab_size: int, embed_dim: int, dropout: float = 0.0,
                 pad_token_id: int = 0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(token_ids))


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).

    Precomputes cos/sin tables and applies rotation to Q and K.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = seq_len

    def forward(self, seq_len: int, device: torch.device,
                offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len + offset > self.max_seq_len:
            self._build_cache(seq_len + offset)
        cos = self.cos_cached[offset:offset + seq_len].to(device)
        sin = self.sin_cached[offset:offset + seq_len].to(device)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    q, k: (B, H, T, D)
    cos, sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class SinusoidalEmbedding(nn.Module):
    """Classic sinusoidal positional embedding (alternative to RoPE)."""

    def __init__(self, embed_dim: int, max_seq_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                             -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)].unsqueeze(0)
