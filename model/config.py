"""Model configuration using a dataclass."""
from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class ModelConfig:
    """Configuration for the GPT-style decoder-only transformer."""

    vocab_size: int = 16000
    embed_dim: int = 384
    num_heads: int = 6
    num_layers: int = 6
    ffn_dim: Optional[int] = None  # auto = 4 * embed_dim
    max_seq_len: int = 512
    dropout: float = 0.1
    use_rope: bool = True
    tie_weights: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    def __post_init__(self) -> None:
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.embed_dim
        assert self.embed_dim % self.num_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        )

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


@dataclass
class TrainConfig:
    """Configuration for training."""

    batch_size: int = 16
    epochs: int = 3
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    label_smoothing: float = 0.0
    log_interval: int = 10
    save_interval: int = 500
    use_amp: bool = True
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
