"""Misc helper utilities."""
from __future__ import annotations
import os
import random
import torch
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_from_checkpoint(ckpt_path: str, model_cls, model_config_cls,
                               device: str = "cpu"):
    """Restore a model + its config from a checkpoint produced by Trainer."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = model_config_cls(**ckpt["model_config"])
    model = model_cls(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)
