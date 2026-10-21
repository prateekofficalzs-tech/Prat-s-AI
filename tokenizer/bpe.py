"""Byte-Pair Encoding (BPE) tokenizer implemented with NumPy / pure Python.

Operates over UTF-8 bytes so it can encode arbitrary text.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from typing import Dict, List, Tuple, Iterable
import numpy as np

from .special_tokens import SPECIAL_TOKENS, PAD, BOS, EOS, UNK


# Pre-tokenization regex (similar in spirit to GPT-2's): splits on whitespace,
# punctuation, and contractions to give BPE a sensible word boundary.
_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def _get_pair_counts(splits: List[List[bytes]], weights: List[int]) -> Counter:
    pair_counts: Counter = Counter()
    for tokens, w in zip(splits, weights):
        for a, b in zip(tokens[:-1], tokens[1:]):
            pair_counts[(a, b)] += w
    return pair_counts


def _merge_pair(tokens: List[bytes], pair: Tuple[bytes, bytes],
                new_token: bytes) -> List[bytes]:
    out: List[bytes] = []
    i = 0
    a, b = pair
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
            out.append(new_token)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


class BPETokenizer:
    """A byte-level BPE tokenizer.

    Special tokens occupy ids 0..len(SPECIAL_TOKENS)-1.
    Then 256 single-byte tokens, then learned merges.
    """

    def __init__(self) -> None:
        self.special_tokens: List[str] = list(SPECIAL_TOKENS)
        self.token_to_id: Dict[bytes, int] = {}
        self.id_to_token: Dict[int, bytes] = {}
        self.merges: List[Tuple[bytes, bytes]] = []
        self.merge_ranks: Dict[Tuple[bytes, bytes], int] = {}
        self._build_base_vocab()

    # ---------- vocab init ----------
    def _build_base_vocab(self) -> None:
        self.token_to_id.clear()
        self.id_to_token.clear()
        # Specials first
        for i, tok in enumerate(self.special_tokens):
            b = tok.encode("utf-8")
            self.token_to_id[b] = i
            self.id_to_token[i] = b
        # 256 bytes next
        offset = len(self.special_tokens)
        for b in range(256):
            tok = bytes([b])
            self.token_to_id[tok] = offset + b
            self.id_to_token[offset + b] = tok

    # ---------- properties ----------
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int: return self.token_to_id[PAD.encode("utf-8")]
    @property
    def bos_id(self) -> int: return self.token_to_id[BOS.encode("utf-8")]
    @property
    def eos_id(self) -> int: return self.token_to_id[EOS.encode("utf-8")]
    @property
    def unk_id(self) -> int: return self.token_to_id[UNK.encode("utf-8")]

    # ---------- training ----------
    def train(self, texts: Iterable[str], vocab_size: int = 8000,
              verbose: bool = True) -> None:
        """Train BPE merges from an iterable of texts."""
        self._build_base_vocab()
        target_merges = max(0, vocab_size - self.vocab_size)

        # Pre-tokenize and count word frequencies
        word_counter: Counter = Counter()
        for text in texts:
            for m in _PAT.findall(text):
                word_counter[m] += 1

        # Each "word" becomes a list of single-byte tokens
        splits: List[List[bytes]] = []
        weights: List[int] = []
        for word, freq in word_counter.items():
            wb = word.encode("utf-8")
            splits.append([bytes([b]) for b in wb])
            weights.append(freq)

        for step in range(target_merges):
            pair_counts = _get_pair_counts(splits, weights)
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break
            new_tok = best_pair[0] + best_pair[1]
            # Add to vocab
            new_id = self.vocab_size
            self.token_to_id[new_tok] = new_id
            self.id_to_token[new_id] = new_tok
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merge_ranks)
            # Apply merge to all splits
            splits = [_merge_pair(s, best_pair, new_tok) for s in splits]
            if verbose and (step + 1) % 200 == 0:
                print(f"  [BPE] step {step+1}/{target_merges} | "
                      f"vocab={self.vocab_size} | best_pair_freq={best_count}")

        if verbose:
            print(f"[BPE] Done. Final vocab size: {self.vocab_size}")

    # ---------- encode / decode ----------
    def _bpe_word(self, word_bytes: bytes) -> List[int]:
        tokens: List[bytes] = [bytes([b]) for b in word_bytes]
        if len(tokens) < 2:
            return [self.token_to_id.get(t, self.unk_id) for t in tokens]
        # Iteratively apply the lowest-rank merge
        while True:
            best_rank = None
            best_idx = -1
            for i in range(len(tokens) - 1):
                rank = self.merge_ranks.get((tokens[i], tokens[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None:
                break
            a, b = tokens[best_idx], tokens[best_idx + 1]
            tokens = tokens[:best_idx] + [a + b] + tokens[best_idx + 2:]
            if len(tokens) == 1:
                break
        return [self.token_to_id.get(t, self.unk_id) for t in tokens]

    def encode(self, text: str, add_bos: bool = False,
               add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for piece in _PAT.findall(text):
            ids.extend(self._bpe_word(piece.encode("utf-8")))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_np(self, text: str, **kwargs) -> np.ndarray:
        return np.asarray(self.encode(text, **kwargs), dtype=np.int64)

    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id} if skip_special else set()
        out = b""
        for t in tokens:
            if t in specials:
                continue
            b = self.id_to_token.get(int(t))
            if b is None:
                continue
            # Strip the special tokens' raw bytes too
            if skip_special and b in (PAD.encode(), BOS.encode(),
                                       EOS.encode(), UNK.encode()):
                continue
            out += b
        return out.decode("utf-8", errors="replace")

    # ---------- persistence ----------
    def save(self, path: str) -> None:
        data = {
            "special_tokens": self.special_tokens,
            "merges": [[a.hex(), b.hex()] for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.special_tokens = data["special_tokens"]
        tok._build_base_vocab()
        for a_hex, b_hex in data["merges"]:
            a = bytes.fromhex(a_hex)
            b = bytes.fromhex(b_hex)
            new = a + b
            tok.merges.append((a, b))
            tok.merge_ranks[(a, b)] = len(tok.merge_ranks)
            new_id = tok.vocab_size
            tok.token_to_id[new] = new_id
            tok.id_to_token[new_id] = new
        return tok
