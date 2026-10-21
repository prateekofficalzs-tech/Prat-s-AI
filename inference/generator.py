"""Text generation utilities: greedy / top-k / top-p / temperature / repetition penalty."""
from __future__ import annotations
from typing import Optional, Iterator, List
import torch
import torch.nn.functional as F

from model import MiniGPT
from tokenizer import BPETokenizer


def _apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor,
                              penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, input_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits = logits.scatter(1, input_ids, score)
    return logits


def _top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0,
                        top_p: float = 1.0) -> torch.Tensor:
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        thresh = torch.topk(logits, k)[0][..., -1, None]
        logits = torch.where(logits < thresh,
                             torch.full_like(logits, float("-inf")), logits)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
    return logits


class Generator:
    """Autoregressive sampler with KV cache."""

    def __init__(self, model: MiniGPT, tokenizer: BPETokenizer,
                 device: Optional[str] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95,
                 repetition_penalty: float = 1.1,
                 greedy: bool = False, stop_on_eos: bool = True,
                 use_kv_cache: bool = True) -> str:
        out_ids = list(self._generate_ids(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            greedy=greedy, stop_on_eos=stop_on_eos, use_kv_cache=use_kv_cache,
        ))
        return self.tokenizer.decode(out_ids)

    @torch.no_grad()
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream decoded text piece-by-piece as tokens are produced."""
        buffer: List[int] = []
        for tok in self._generate_ids(prompt, **kwargs):
            buffer.append(tok)
            text = self.tokenizer.decode(buffer)
            # only yield when we have a clean utf-8 chunk
            if not text.endswith("�"):
                yield text
                buffer = []
        if buffer:
            yield self.tokenizer.decode(buffer)

    @torch.no_grad()
    def _generate_ids(self, prompt: str, max_new_tokens: int = 100,
                      temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95,
                      repetition_penalty: float = 1.1,
                      greedy: bool = False, stop_on_eos: bool = True,
                      use_kv_cache: bool = True) -> Iterator[int]:
        max_seq_len = self.model.config.max_seq_len
        ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        ids = ids[-max_seq_len:]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        generated = input_ids.clone()
        kv_caches = None

        # Prefill
        logits, _, kv_caches = self.model(input_ids, kv_caches=None if use_kv_cache else None)
        if not use_kv_cache:
            kv_caches = None

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]
            next_logits = _apply_repetition_penalty(next_logits, generated,
                                                    repetition_penalty)
            if greedy or temperature <= 0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / max(1e-6, temperature)
                next_logits = _top_k_top_p_filter(next_logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            tok = int(next_id.item())
            yield tok
            if stop_on_eos and tok == self.tokenizer.eos_id:
                break

            generated = torch.cat([generated, next_id], dim=1)
            if generated.size(1) >= max_seq_len:
                # Drop oldest, invalidate cache
                generated = generated[:, -max_seq_len:]
                kv_caches = None
                logits, _, kv_caches = self.model(generated, kv_caches=None)
                continue

            if use_kv_cache and kv_caches is not None:
                logits, _, kv_caches = self.model(next_id, kv_caches=kv_caches)
            else:
                logits, _, _ = self.model(generated)
