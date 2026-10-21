"""Full decoder-only GPT-style language model."""
from __future__ import annotations
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .embeddings import TokenEmbedding, SinusoidalEmbedding
from .transformer import TransformerBlock


class MiniGPT(nn.Module):
    """A configurable decoder-only transformer language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = TokenEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            dropout=config.dropout,
            pad_token_id=config.pad_token_id,
        )
        self.pos_emb = None
        if not config.use_rope:
            self.pos_emb = SinusoidalEmbedding(config.embed_dim, config.max_seq_len)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
                use_rope=config.use_rope,
            ) for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_emb.embedding.weight

        # Gradient checkpointing flag
        self.grad_checkpointing = False

    # -------- utilities --------
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    def parameter_summary(self) -> str:
        total = self.num_parameters(trainable_only=False)
        trainable = self.num_parameters(trainable_only=True)
        return (f"MiniGPT: layers={self.config.num_layers}, "
                f"embed_dim={self.config.embed_dim}, heads={self.config.num_heads}, "
                f"vocab={self.config.vocab_size}, max_seq={self.config.max_seq_len}\n"
                f"Total params: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M")

    def enable_gradient_checkpointing(self) -> None:
        self.grad_checkpointing = True

    # -------- forward --------
    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                attention_mask: Optional[torch.Tensor] = None,
                label_smoothing: float = 0.0
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List]:
        """Forward pass.

        Args:
            input_ids: (B, T)
            targets: (B, T) optional — if provided, returns loss
            kv_caches: optional list of (k, v) per layer for fast generation
            attention_mask: optional additive mask (B, 1, T_q, T_k) in log space
            label_smoothing: cross-entropy label smoothing

        Returns:
            logits (B, T, V), optional loss, new kv_caches list
        """
        x = self.token_emb(input_ids)
        if self.pos_emb is not None:
            x = self.pos_emb(x)

        new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            if self.grad_checkpointing and self.training and cache_i is None:
                x, new_cache = torch.utils.checkpoint.checkpoint(
                    block, x, None, attention_mask, use_reentrant=False,
                )
            else:
                x, new_cache = block(x, kv_cache=cache_i, attn_mask=attention_mask)
            new_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.pad_token_id,
                label_smoothing=label_smoothing,
            )
        return logits, loss, new_caches
