"""Prat`sAI — Codex-style desktop chat.

FastAPI backend + hand-built HTML/CSS/JS UI (no Streamlit).

Run:
    python ui/app.py
    python ui/app.py --model-path models/base --port 8000

Opens automatically in your default browser.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import re
import threading
import webbrowser
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models" / "base"))
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--no-browser", action="store_true")
args, _ = parser.parse_known_args()


# ============================================================
# Model loading (once at startup)
# ============================================================
print(f"\n◆ Prat`sAI  ·  loading model from {args.model_path}…")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model_path, torch_dtype=dtype
).to(device).eval()
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  · device: {device}  ·  params: {n_params:.0f}M  ·  ready\n")


# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="PratsAI")
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(str(HERE / "static" / "index.html"))


@app.get("/api/info")
async def info():
    return JSONResponse({"params": f"{n_params:.0f}", "device": device})


# ------------------------------------------------------------
#  Server-side thinking-stripper for Qwen3 and similar models.
#  Buffers incoming tokens.  While inside a <think> block
#  (or with an unclosed <think> tag), we send empty heartbeat
#  events so the frontend stays alive.  After </think> we
#  forward the real tokens to the client.
# ------------------------------------------------------------
_THINK_RE = re.compile(r"</?think[^>]*>", re.IGNORECASE)


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    max_new = int(body.get("max_new_tokens", 1024))
    min_new = int(body.get("min_new_tokens", 20))
    temperature = float(body.get("temperature", 0.7))
    top_p = float(body.get("top_p", 0.9))
    rep_pen = float(body.get("repetition_penalty", 1.1))

    # Tell Qwen3 to NOT use thinking mode.
    # The `enable_thinking` flag is passed through chat_template_kwargs.
    system_prompt = (
        "You are Prat`sAI, a helpful, knowledgeable assistant. "
        "Always be thorough and detailed. "
        "Explain reasoning step by step. "
        "For math, show full working. "
        "Never output <think> or </think> tags."
    )
    chat_msgs = [{"role": "system", "content": system_prompt}] + messages

    try:
        text = tokenizer.apply_chat_template(
            chat_msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,  # ← Qwen3-specific
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            chat_msgs, tokenize=False, add_generation_prompt=True,
        )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    eos_id = tokenizer.eos_token_id
    gen_min = min(min_new, max_new // 2)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True,
    )

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new,
        min_new_tokens=gen_min,
        do_sample=temperature > 0.0,
        temperature=max(0.01, temperature),
        top_p=top_p,
        top_k=max(1, int(top_p * 200)),
        repetition_penalty=rep_pen,
        pad_token_id=eos_id,
        eos_token_id=eos_id,
        streamer=streamer,
        no_repeat_ngram_size=4,
    )
    gen_thread = threading.Thread(
        target=model.generate, kwargs=gen_kwargs, daemon=True,
    )
    gen_thread.start()

    async def event_stream():
        loop = asyncio.get_event_loop()
        full = ""

        while True:
            piece = await loop.run_in_executor(
                None, lambda: next(streamer, None)
            )
            if piece is None:
                break
            full += piece

            # While accumulating, strip any <think> blocks that leaked
            # through (Qwen3 sometimes emits them despite enable_thinking=False).
            stripped = re.sub(
                r"<think[\s\S]*?</think>", "", full, flags=re.IGNORECASE
            )
            # Also handle unclosed <think> — remove everything after it
            unclosed = re.search(r"<think[^>]*>", stripped, re.IGNORECASE)
            if unclosed:
                stripped = stripped[:unclosed.start()]

            yield f"data: {json.dumps({'token': stripped})}\n\n"

        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ============================================================
# Entrypoint
# ============================================================
def main() -> None:
    import uvicorn
    url = f"http://{args.host}:{args.port}"
    print(f"  Open: {url}\n")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
