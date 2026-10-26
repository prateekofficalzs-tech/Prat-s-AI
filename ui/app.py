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
import sys
import threading
import time
import webbrowser
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


# In the packaged Windows app, bundled UI files are unpacked into _MEIPASS
# while user models remain beside Prat-sAI.exe in the project folder.
if getattr(sys, "frozen", False):
    HERE = Path(sys._MEIPASS) / "ui"
    PROJECT_ROOT = Path(sys.executable).parent
else:
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
MODELS_DIR = PROJECT_ROOT / "models"
_model_lock = threading.RLock()
_active_generations = 0
_heartbeat_lock = threading.Lock()
_last_browser_heartbeat = None


def load_model(model_path: Path):
    """Load one local Hugging Face model folder for the chat server."""
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype
    ).to(device).eval()
    params = sum(p.numel() for p in mdl.parameters()) / 1e6
    return tok, mdl, params


active_model_path = Path(args.model_path).resolve()
tokenizer, model, n_params = load_model(active_model_path)
active_model_name = active_model_path.name
print(f"  · device: {device}  ·  params: {n_params:.0f}M  ·  ready\n")


def available_models() -> list[str]:
    """Only expose direct folders in ./models; never accept arbitrary paths."""
    if not MODELS_DIR.is_dir():
        return [active_model_name]
    names = sorted(folder.name for folder in MODELS_DIR.iterdir() if folder.is_dir())
    return names if active_model_name in names else [active_model_name, *names]


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
    with _model_lock:
        return JSONResponse({
            "params": f"{n_params:.0f}",
            "device": device,
            "active_model": active_model_name,
            "models": available_models(),
        })


@app.post("/api/heartbeat")
async def browser_heartbeat():
    """Record an open chat tab so the packaged server can stop when it closes."""
    global _last_browser_heartbeat
    with _heartbeat_lock:
        _last_browser_heartbeat = time.monotonic()
    return JSONResponse({"ok": True})


@app.post("/api/model")
async def select_model(req: Request):
    """Switch models on demand, after the current answer has finished."""
    global tokenizer, model, n_params, active_model_path, active_model_name
    body = await req.json()
    requested = body.get("model")
    if not isinstance(requested, str) or requested not in available_models():
        return JSONResponse({"detail": "Unknown model."}, status_code=400)

    with _model_lock:
        if _active_generations:
            return JSONResponse(
                {"detail": "Wait for the current response to finish before switching models."},
                status_code=409,
            )
        if requested == active_model_name:
            return JSONResponse({"model": active_model_name, "params": f"{n_params:.0f}", "device": device})

    model_path = (MODELS_DIR / requested).resolve()
    if model_path.parent != MODELS_DIR.resolve() or not model_path.is_dir():
        return JSONResponse({"detail": "Invalid model folder."}, status_code=400)

    try:
        # Loading is blocking and can take a while, so do it outside the event loop.
        new_tokenizer, new_model, new_params = await asyncio.to_thread(load_model, model_path)
    except Exception as exc:
        return JSONResponse({"detail": f"Could not load {requested}: {exc}"}, status_code=500)

    with _model_lock:
        if _active_generations:
            return JSONResponse(
                {"detail": "A response started while the model was loading; please try again."},
                status_code=409,
            )
        tokenizer, model, n_params = new_tokenizer, new_model, new_params
        active_model_path, active_model_name = model_path, requested
    if device == "cuda":
        torch.cuda.empty_cache()
    return JSONResponse({"model": active_model_name, "params": f"{n_params:.0f}", "device": device})


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
    global _active_generations
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
        "Always respond in clear, fluent, grammatically correct English. "
        "Explain reasoning step by step. "
        "For math problems, show full step-by-step working and format formulas using standard LaTeX ($...$ for inline math, $$...$$ for block math equations). "
        "Never output <think> or </think> tags."
    )
    chat_msgs = [{"role": "system", "content": system_prompt}] + messages

    # Reserve the currently selected model before constructing the prompt so a
    # model switch cannot pair one tokenizer with another model.
    with _model_lock:
        chat_tokenizer, chat_model = tokenizer, model
        _active_generations += 1

    try:
        text = chat_tokenizer.apply_chat_template(
            chat_msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,  # ← Qwen3-specific
        )
    except TypeError:
        text = chat_tokenizer.apply_chat_template(
            chat_msgs, tokenize=False, add_generation_prompt=True,
        )
    inputs = chat_tokenizer(text, return_tensors="pt").to(chat_model.device)

    eos_id = chat_tokenizer.eos_token_id
    gen_min = min(min_new, max_new // 2)

    streamer = TextIteratorStreamer(
        chat_tokenizer, skip_prompt=True, skip_special_tokens=True,
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
    )
    gen_thread = threading.Thread(
        target=chat_model.generate, kwargs=gen_kwargs, daemon=True,
    )
    gen_thread.start()

    async def event_stream():
        loop = asyncio.get_event_loop()
        full = ""
        try:
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
        finally:
            with _model_lock:
                _active_generations -= 1

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
    # A PyInstaller --noconsole app has no stderr; Uvicorn's default logging
    # formatter calls stderr.isatty(), which raises when stderr is None.
    server = uvicorn.Server(uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        log_config=None,
        access_log=False,
    ))

    def stop_when_browser_closes() -> None:
        # Allow refreshes or opening a replacement tab, but stop the local
        # server after all chat tabs have stopped sending heartbeats.
        while not server.should_exit:
            time.sleep(2)
            with _heartbeat_lock:
                last_seen = _last_browser_heartbeat
            if last_seen is not None and time.monotonic() - last_seen > 12:
                server.should_exit = True
                break

    threading.Thread(target=stop_when_browser_closes, daemon=True).start()
    server.run()


if __name__ == "__main__":
    main()
