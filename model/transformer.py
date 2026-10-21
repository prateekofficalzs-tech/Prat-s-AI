"""Pre-LayerNorm Transformer decoder block."""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .ffn import FeedForward


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: LN -> Attn -> Residual -> LN -> FFN -> Residual."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.0, max_seq_len: int = 4096,
                 use_rope: bool = True) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
            max_seq_len=max_seq_len, use_rope=use_rope,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout=dropout)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                attn_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_cache
