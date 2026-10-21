"""HuggingFace causal LM wrapper that conforms to the MiniGPT interface.

This lets the existing Trainer / Generator / ChatEngine work unchanged on top
of any HF AutoModelForCausalLM (gpt2, distilgpt2, Qwen2, TinyLlama, ...).

Supports three training modes:
    1. Full fine-tuning              (default)
    2. Frozen base + trainable head  (freeze_base=True)
    3. LoRA adapters                 (use_lora=True; requires `peft`)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Any
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HFModelConfig:
    """Config for a HF-backed model. Mirrors MiniGPT's ModelConfig surface."""

    model_name: str = "distilgpt2"
    max_seq_len: int = 512
    dropout: float = 0.0  # placeholder for interface parity
    pad_token_id: int = 0
    bos_token_id: int = 0
    eos_token_id: int = 0
    unk_token_id: int = 0
    vocab_size: int = 0  # populated after model load

    # Trainability options
    freeze_base: bool = False
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None  # auto if None
    tie_weights: bool = True  # for parity; HF handles its own tying

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "HFModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


def _auto_lora_targets(hf_model) -> List[str]:
    """Best-effort default target_modules per architecture family."""
    name = hf_model.__class__.__name__.lower()
    if "gpt2" in name or "gptneo" in name:
        return ["c_attn"]
    if "llama" in name or "mistral" in name or "qwen" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if "gptj" in name:
        return ["q_proj", "v_proj"]
    # Fallback: attention/MLP linear-ish names; user can override.
    return ["q_proj", "v_proj", "c_attn"]


class HFCausalLM(nn.Module):
    """Wraps `AutoModelForCausalLM` so it looks like MiniGPT to the trainer/generator.

    Forward returns: (logits, optional_loss, past_key_values)
    """

    def __init__(self, cfg: HFModelConfig) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoConfig

        self.cfg = cfg
        hf_cfg = AutoConfig.from_pretrained(cfg.model_name)
        self.hf_model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

        # Fill back config fields from the real model
        cfg.vocab_size = int(getattr(hf_cfg, "vocab_size", 0)) or self.hf_model.get_input_embeddings().num_embeddings
        cfg.max_seq_len = (
            getattr(hf_cfg, "n_positions", None)
            or getattr(hf_cfg, "max_position_embeddings", None)
            or cfg.max_seq_len
        )

        # Trainability
        if cfg.freeze_base:
            for p in self.hf_model.parameters():
                p.requires_grad = False
            # Keep the LM head + final norm trainable
            head = getattr(self.hf_model, "lm_head", None)
            if head is not None:
                for p in head.parameters():
                    p.requires_grad = True

        if cfg.use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                targets = cfg.lora_target_modules or _auto_lora_targets(self.hf_model)
                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    target_modules=targets,
                    bias="none",
                )
                self.hf_model = get_peft_model(self.hf_model, lora_cfg)
            except ImportError:
                print("[HFCausalLM] `peft` not installed — LoRA disabled. "
                      "Run: pip install peft")
                cfg.use_lora = False

        # Match MiniGPT's surface so Trainer/Generator/ChatEngine work unchanged
        self.config = cfg
        self.grad_checkpointing = False

    # ----- parity helpers -----
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (p.requires_grad or not trainable_only)
        )

    def parameter_summary(self) -> str:
        total = self.num_parameters(trainable_only=False)
        trainable = self.num_parameters(trainable_only=True)
        return (
            f"Prat`sAI model | vocab={self.cfg.vocab_size} "
            f"max_seq={self.cfg.max_seq_len}\n"
            f"  freeze_base={self.cfg.freeze_base} use_lora={self.cfg.use_lora}\n"
            f"  Total params: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M "
            f"({100*trainable/max(1,total):.2f}%)"
        )

    def enable_gradient_checkpointing(self) -> None:
        self.grad_checkpointing = True
        try:
            self.hf_model.gradient_checkpointing_enable()
        except Exception as e:
            print(f"[HFCausalLM] gradient checkpointing not supported: {e}")

    # ----- forward -----
    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                kv_caches: Optional[Any] = None,
                attention_mask: Optional[torch.Tensor] = None,
                label_smoothing: float = 0.0
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Any]:
        use_cache = (targets is None)
        out = self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kv_caches,
            use_cache=use_cache,
            return_dict=True,
        )
        logits = out.logits
        new_caches = getattr(out, "past_key_values", None)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.cfg.pad_token_id,
                label_smoothing=label_smoothing,
            )
        return logits, loss, new_caches

    # ----- LoRA save/merge helpers -----
    def save_lora_adapter(self, path: str) -> None:
        """Save only LoRA adapter weights (small)."""
        if not self.cfg.use_lora:
            raise RuntimeError("Not a LoRA model.")
        self.hf_model.save_pretrained(path)

    def merge_and_unload(self) -> None:
        """Fold LoRA weights into base for vanilla inference."""
        if self.cfg.use_lora and hasattr(self.hf_model, "merge_and_unload"):
            self.hf_model = self.hf_model.merge_and_unload()
            self.cfg.use_lora = False
