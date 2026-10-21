"""HuggingFace tokenizer wrapper exposing the same surface as BPETokenizer.

Lets the same TextDataset / InstructionDataset / Generator / ChatEngine work
on top of any HF AutoTokenizer.
"""
from __future__ import annotations
from typing import List
import os
import numpy as np


class HFTokenizerWrapper:
    """Wraps `AutoTokenizer` with a BPETokenizer-compatible API."""

    def __init__(self, model_name_or_path: str) -> None:
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name_or_path)
        self._normalize_specials()

    def _normalize_specials(self) -> None:
        # Many causal LMs (e.g. gpt2) lack pad — reuse eos.
        if self.tok.pad_token_id is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token
        self._pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self._eos = self.tok.eos_token_id if self.tok.eos_token_id is not None else self._pad
        self._bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self._eos
        self._unk = self.tok.unk_token_id if self.tok.unk_token_id is not None else self._eos

    # ---- properties matching BPETokenizer ----
    @property
    def vocab_size(self) -> int: return len(self.tok)
    @property
    def pad_id(self) -> int: return int(self._pad)
    @property
    def bos_id(self) -> int: return int(self._bos)
    @property
    def eos_id(self) -> int: return int(self._eos)
    @property
    def unk_id(self) -> int: return int(self._unk)

    # ---- encode / decode ----
    def encode(self, text: str, add_bos: bool = False,
               add_eos: bool = False) -> List[int]:
        ids = self.tok.encode(text, add_special_tokens=False)
        if add_bos and self._bos is not None:
            ids = [int(self._bos)] + ids
        if add_eos and self._eos is not None:
            ids = ids + [int(self._eos)]
        return ids

    def encode_np(self, text: str, **kwargs) -> np.ndarray:
        return np.asarray(self.encode(text, **kwargs), dtype=np.int64)

    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        return self.tok.decode(list(tokens), skip_special_tokens=skip_special,
                               clean_up_tokenization_spaces=False)

    # ---- persistence (saved as a directory, like HF expects) ----
    def save(self, path: str) -> None:
        # If user passed a .json path, drop the extension and use the dir.
        if path.endswith(".json"):
            path = os.path.splitext(path)[0]
        os.makedirs(path, exist_ok=True)
        self.tok.save_pretrained(path)

    @classmethod
    def load(cls, path: str) -> "HFTokenizerWrapper":
        if path.endswith(".json") and not os.path.isdir(path):
            # Try the matching directory
            alt = os.path.splitext(path)[0]
            if os.path.isdir(alt):
                path = alt
        return cls(path)
