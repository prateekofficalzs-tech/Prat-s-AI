"""Training loop with AMP, gradient clipping, checkpointing, logging."""
from __future__ import annotations
import math
import os
import time
from typing import Optional
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from model import MiniGPT, ModelConfig, TrainConfig
from .scheduler import cosine_warmup_scheduler


class Trainer:
    """Encapsulates training of a MiniGPT model."""

    def __init__(self, model: MiniGPT, train_loader: DataLoader,
                 model_config: ModelConfig, train_config: TrainConfig,
                 val_loader: Optional[DataLoader] = None,
                 device: Optional[str] = None) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_config = model_config
        self.cfg = train_config

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Param groups: no weight decay for biases & LayerNorms
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or n.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        self.optimizer = AdamW(
            [
                {"params": decay, "weight_decay": self.cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.cfg.lr, betas=self.cfg.betas,
        )

        total_steps = max(1, self.cfg.epochs * len(train_loader))
        min_lr_ratio = self.cfg.min_lr / self.cfg.lr if self.cfg.lr > 0 else 0.0
        self.scheduler = cosine_warmup_scheduler(
            self.optimizer, self.cfg.warmup_steps, total_steps,
            min_lr_ratio=min_lr_ratio,
        )

        self.use_amp = self.cfg.use_amp and self.device.startswith("cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        self.global_step = 0

    # ------- helpers -------
    def _amp_ctx(self):
        if self.use_amp:
            return torch.cuda.amp.autocast(dtype=torch.float16)
        from contextlib import nullcontext
        return nullcontext()

    # ------- main loop -------
    def train(self) -> None:
        print(self.model.parameter_summary())
        print(f"Device: {self.device} | AMP: {self.use_amp}")
        self.model.train()
        for epoch in range(self.cfg.epochs):
            self._run_epoch(epoch)
            self.save_checkpoint(f"epoch_{epoch+1}.pt")
            if self.val_loader is not None:
                vloss = self.evaluate()
                print(f"[Eval] epoch={epoch+1} val_loss={vloss:.4f} "
                      f"val_ppl={math.exp(min(20, vloss)):.2f}")

    def _run_epoch(self, epoch: int) -> None:
        t0 = time.time()
        running_loss = 0.0
        n = 0
        for step, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with self._amp_ctx():
                _, loss, _ = self.model(x, targets=y,
                                         label_smoothing=self.cfg.label_smoothing)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1

            running_loss += loss.item()
            n += 1

            if (step + 1) % self.cfg.log_interval == 0:
                avg = running_loss / n
                lr = self.scheduler.get_last_lr()[0]
                ppl = math.exp(min(20, avg))
                dt = time.time() - t0
                print(f"epoch={epoch+1} step={step+1}/{len(self.train_loader)} "
                      f"loss={avg:.4f} ppl={ppl:.2f} lr={lr:.2e} "
                      f"({dt:.1f}s)")
                running_loss = 0.0
                n = 0
                t0 = time.time()

            if self.global_step % self.cfg.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        total_loss, n = 0.0, 0
        for x, y in self.val_loader:
            x = x.to(self.device); y = y.to(self.device)
            with self._amp_ctx():
                _, loss, _ = self.model(x, targets=y)
            total_loss += loss.item(); n += 1
        self.model.train()
        return total_loss / max(1, n)

    # ------- checkpoints -------
    def save_checkpoint(self, name: str) -> None:
        path = os.path.join(self.cfg.checkpoint_dir, name)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "model_config": self.model_config.__dict__,
        }, path)
        print(f"[Checkpoint] Saved {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt.get("global_step", 0)
        print(f"[Checkpoint] Loaded {path} (step={self.global_step})")
