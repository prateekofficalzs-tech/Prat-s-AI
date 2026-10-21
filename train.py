"""End-to-end training entrypoint.

Two backends:
  --backend custom   → from-scratch MiniGPT + NumPy BPE  (default)
  --backend hf       → HuggingFace pretrained, still trainable
                       (full FT / frozen base / LoRA)

Examples:
    # train from scratch
    python train.py --data data/sample.txt --epochs 3

    # full fine-tune distilgpt2
    python train.py --backend hf --hf-model-name distilgpt2 \
                    --data data/sample.txt --epochs 1 --batch-size 4 --lr 5e-5

    # LoRA fine-tune (requires `pip install peft`)
    python train.py --backend hf --hf-model-name distilgpt2 --use-lora \
                    --data data/sample.txt --instructions data/instructions.jsonl

    # Freeze base, only train LM head
    python train.py --backend hf --hf-model-name distilgpt2 --freeze-base \
                    --data data/sample.txt
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from typing import List, Tuple, Any

import torch
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ModelConfig, TrainConfig, MiniGPT, HFCausalLM, HFModelConfig
from tokenizer import BPETokenizer, HFTokenizerWrapper
from training import TextDataset, InstructionDataset, Trainer
from utils import set_seed, auto_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["custom", "hf"], default="custom")
    p.add_argument("--hf-model-name", type=str, default="distilgpt2",
                   help="HF model id (gpt2, distilgpt2, Qwen/Qwen2.5-0.5B, ...)")
    p.add_argument("--freeze-base", action="store_true",
                   help="(HF) freeze base, only train LM head")
    p.add_argument("--use-lora", action="store_true",
                   help="(HF) train LoRA adapters; requires `peft`")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)

    p.add_argument("--data", type=str, nargs="+", default=["data/sample.txt"])
    p.add_argument("--instructions", type=str, default=None)
    p.add_argument("--vocab-size", type=int, default=4000)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--tokenizer-path", type=str, default=None,
                   help="Defaults to <ckpt-dir>/tokenizer.json (custom) "
                        "or <ckpt-dir>/hf_tokenizer (hf)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true")
    return p.parse_args()


def expand_files(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pat in patterns:
        matched = glob.glob(pat)
        files.extend(matched if matched else ([pat] if os.path.exists(pat) else []))
    return files


def build_custom_backend(args, files) -> Tuple[Any, Any, Any]:
    """Returns (tokenizer, model, model_config) for the from-scratch path."""
    tok_path = args.tokenizer_path or os.path.join(args.ckpt_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        print(f"[Tokenizer] Loading {tok_path}")
        tokenizer = BPETokenizer.load(tok_path)
    else:
        print(f"[Tokenizer] Training BPE (target vocab={args.vocab_size})")
        texts = []
        for fp in files:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
        tokenizer = BPETokenizer()
        tokenizer.train(texts, vocab_size=args.vocab_size, verbose=True)
        tokenizer.save(tok_path)
        print(f"[Tokenizer] Saved to {tok_path}")

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, max_seq_len=args.seq_len, dropout=args.dropout,
        pad_token_id=tokenizer.pad_id, bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id, unk_token_id=tokenizer.unk_id,
    )
    cfg.save(os.path.join(args.ckpt_dir, "model_config.json"))
    model = MiniGPT(cfg)
    if args.grad_checkpoint:
        model.enable_gradient_checkpointing()
    return tokenizer, model, cfg


def build_hf_backend(args) -> Tuple[Any, Any, Any]:
    """Returns (tokenizer, model, model_config) for the HF path."""
    tok_path = args.tokenizer_path or os.path.join(args.ckpt_dir, "hf_tokenizer")
    if os.path.isdir(tok_path):
        print(f"[Tokenizer] Loading HF tokenizer from {tok_path}")
        tokenizer = HFTokenizerWrapper.load(tok_path)
    else:
        print(f"[Tokenizer] Loading HF tokenizer for {args.hf_model_name}")
        tokenizer = HFTokenizerWrapper(args.hf_model_name)
        tokenizer.save(tok_path)
        print(f"[Tokenizer] Saved to {tok_path}")

    cfg = HFModelConfig(
        model_name=args.hf_model_name,
        max_seq_len=args.seq_len,
        pad_token_id=tokenizer.pad_id, bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id, unk_token_id=tokenizer.unk_id,
        freeze_base=args.freeze_base, use_lora=args.use_lora,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    )
    model = HFCausalLM(cfg)
    # Honor the model's own max_seq_len if shorter than requested
    cfg.max_seq_len = min(args.seq_len, model.config.max_seq_len)
    cfg.save(os.path.join(args.ckpt_dir, "hf_model_config.json"))
    if args.grad_checkpoint:
        model.enable_gradient_checkpointing()
    return tokenizer, model, cfg


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    files = expand_files(args.data)
    if not files:
        raise SystemExit(f"No data files found: {args.data}")
    print(f"[Data] Using files: {files}")

    # ---- backend dispatch ----
    if args.backend == "hf":
        tokenizer, model, model_config = build_hf_backend(args)
    else:
        tokenizer, model, model_config = build_custom_backend(args, files)

    # ---- dataset ----
    seq_len = model_config.max_seq_len
    text_ds = TextDataset(files, tokenizer, seq_len=seq_len)
    datasets = [text_ds]
    if args.instructions and os.path.exists(args.instructions):
        with open(args.instructions, "r", encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f if line.strip()]
        ins_ds = InstructionDataset(pairs, tokenizer, seq_len=seq_len)
        if len(ins_ds) > 0:
            datasets.append(ins_ds)
            print(f"[Data] Added {len(ins_ds)} instruction examples")
    full_ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"[Data] Total training samples: {len(full_ds)}")

    loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=False)

    # ---- train ----
    train_config = TrainConfig(
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        warmup_steps=args.warmup, use_amp=not args.no_amp,
        checkpoint_dir=args.ckpt_dir, seed=args.seed,
    )
    trainer = Trainer(model, loader, model_config, train_config,
                      device=auto_device())
    trainer.train()
    trainer.save_checkpoint("final.pt")
    print("[Done] Training complete.")


if __name__ == "__main__":
    main()
