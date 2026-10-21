"""Prat`sAI — terminal chat client.

Usage:
    python run.py
    python run.py --model-path models/base
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


APP_NAME = "Prat`sAI"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="models/base",
                   help="Folder with the model files")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n◆ {APP_NAME}")
    print(f"Loading model from {args.model_path} on {device}…")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).to(device).eval()
    n = sum(x.numel() for x in mdl.parameters()) / 1e6
    print(f"Ready · {n:.0f}M params · type 'exit' to quit, 'reset' to clear.\n")

    history = []
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not q: continue
        if q.lower() in ("exit", "quit"): break
        if q.lower() == "reset":
            history = []; print("[history cleared]\n"); continue

        history.append({"role": "user", "content": q})
        sys_msg = "You are Prat`sAI, a helpful assistant. Never output <think> or </think> tags."
        chat_msgs = [{"role": "system", "content": sys_msg}] + history
        try:
            text = tok.apply_chat_template(
                chat_msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tok.apply_chat_template(
                chat_msgs, tokenize=False, add_generation_prompt=True,
            )
        ids = tok(text, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(
                **ids, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=args.temperature, top_p=args.top_p,
                pad_token_id=tok.eos_token_id,
            )
        reply = tok.decode(out[0][ids.input_ids.shape[1]:],
                           skip_special_tokens=True).strip()
        # Strip any <think> tags that leaked through
        import re
        reply = re.sub(r"<[/]?think[^>]*>", "", reply, flags=re.IGNORECASE).strip()
        history.append({"role": "assistant", "content": reply})
        print(f"AI : {reply}\n")


if __name__ == "__main__":
    main()
