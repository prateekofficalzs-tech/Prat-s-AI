"""Multi-Head Causal Self-Attention with optional RoPE and KV cache."""
from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import RotaryEmbedding, apply_rope


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention.

    Uses torch.nn.functional.scaled_dot_product_attention when available
    for efficient flash-attention-style execution.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 max_seq_len: int = 4096, use_rope: bool = True) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout
        self.use_rope = use_rope

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        else:
            self.rope = None

        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                attn_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        # (B, H, T, D)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        offset = 0
        if kv_cache is not None:
            past_k, past_v = kv_cache
            offset = past_k.size(2)

        if self.use_rope and self.rope is not None:
            cos, sin = self.rope(T, x.device, offset=offset)
            q, k = apply_rope(q, k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        is_causal = kv_cache is None  # only causal-mask during training/prefill
        # When using kv cache for single-token generation, T==1 and no mask needed.
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal and attn_mask is None,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if is_causal and attn_mask is None:
                T_q, T_k = scores.size(-2), scores.size(-1)
                mask = torch.ones(T_q, T_k, device=x.device, dtype=torch.bool).tril()
                scores = scores.masked_fill(~mask, float("-inf"))
            elif attn_mask is not None:
                scores = scores + attn_mask
            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout_p, training=self.training)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out, new_cache
