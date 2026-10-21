"""Cosine learning-rate schedule with linear warmup."""
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer


def cosine_warmup_scheduler(optimizer: Optimizer, warmup_steps: int,
                            total_steps: int, min_lr_ratio: float = 0.1) -> LambdaLR:
    """Returns a LambdaLR with linear warmup then cosine decay to min_lr_ratio*lr."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
