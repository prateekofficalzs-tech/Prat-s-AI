"""Interactive terminal chat client.

Usage:
    # custom MiniGPT
    python chat.py --ckpt checkpoints/final.pt --tokenizer checkpoints/tokenizer.json

    # HF backend
    python chat.py --backend hf --ckpt checkpoints/final.pt \
                   --tokenizer checkpoints/hf_tokenizer
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ModelConfig, MiniGPT, HFCausalLM, HFModelConfig
from tokenizer import BPETokenizer, HFTokenizerWrapper
from inference import Generator, ChatEngine
from utils import load_model_from_checkpoint, auto_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["custom", "hf"], default="custom")
    p.add_argument("--ckpt", type=str, default="checkpoints/final.pt")
    p.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = auto_device()
    print(f"\n◆ Prat`sAI  ·  device: {device}  ·  backend: {args.backend}")

    if args.backend == "hf":
        tokenizer = HFTokenizerWrapper.load(args.tokenizer)
        model, cfg = load_model_from_checkpoint(
            args.ckpt, HFCausalLM, HFModelConfig, device=device,
        )
    else:
        tokenizer = BPETokenizer.load(args.tokenizer)
        model, cfg = load_model_from_checkpoint(
            args.ckpt, MiniGPT, ModelConfig, device=device,
        )
    print(model.parameter_summary())

    gen = Generator(model, tokenizer, device=device)
    chat = ChatEngine(gen)

    print("\nType 'exit' to quit, 'reset' to clear history.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            break
        if user.lower() == "reset":
            chat.reset()
            print("[history cleared]")
            continue
        response = chat.chat(
            user, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"AI : {response}\n")


if __name__ == "__main__":
    main()
