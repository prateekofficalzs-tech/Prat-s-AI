"""Position-wise Feed-Forward Network."""
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Two-layer MLP: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))
