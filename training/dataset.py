"""Dataset for causal LM training: sliding-window chunks over tokenized text."""
from __future__ import annotations
from typing import List, Optional, Tuple
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizer import BPETokenizer


class TextDataset(Dataset):
    """Reads raw text files, tokenizes, builds sliding-window chunks.

    Each sample yields (input_ids, target_ids) where target = input shifted by 1.
    """

    def __init__(self, file_paths: List[str], tokenizer: BPETokenizer,
                 seq_len: int = 256, stride: Optional[int] = None,
                 add_bos: bool = True, add_eos: bool = True) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

        all_ids: List[int] = []
        for p in file_paths:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
            all_ids.extend(ids)
        if len(all_ids) < seq_len + 1:
            # Pad to at least one sample
            all_ids = all_ids + [tokenizer.pad_id] * (seq_len + 1 - len(all_ids))
        self.tokens = np.asarray(all_ids, dtype=np.int64)

        # Pre-compute chunk start positions
        last_start = len(self.tokens) - seq_len - 1
        self.starts = list(range(0, max(1, last_start + 1), self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.starts[idx]
        chunk = self.tokens[s:s + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


class InstructionDataset(Dataset):
    """Instruction-tuning dataset of {"prompt": ..., "response": ...} pairs.

    Builds: <BOS> prompt <EOS> response <EOS>
    Loss is computed on the response tokens only (prompt tokens get pad target).
    """

    def __init__(self, pairs: List[dict], tokenizer: BPETokenizer,
                 seq_len: int = 256) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        for pair in pairs:
            prompt = pair["prompt"]
            response = pair["response"]
            p_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False) + [tokenizer.eos_id]
            r_ids = tokenizer.encode(response, add_bos=False, add_eos=True)
            ids = (p_ids + r_ids)[:seq_len + 1]
            if len(ids) < 2:
                continue
            # Pad
            pad = tokenizer.pad_id
            ids = ids + [pad] * (seq_len + 1 - len(ids))
            x = np.asarray(ids[:-1], dtype=np.int64)
            y = np.asarray(ids[1:], dtype=np.int64)
            # Mask prompt tokens in targets
            mask_until = max(0, len(p_ids) - 1)
            y[:mask_until] = pad
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
