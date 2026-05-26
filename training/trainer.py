# training/trainer.py
# ─────────────────────────────────────────────────────────────────────────────
# HERMES Trainer
#
# Features:
#   • Mixed-precision (AMP) with GradScaler
#   • Gradient accumulation (configurable ACCUM_STEPS)
#   • Exponential Moving Average (EMA) of weights for inference
#   • OTTA meta-loss: randomly reset SSM state mid-sequence, penalise
#     slow recovery — forces SSM to be a fast in-context adapter
#   • Curriculum: Phase 1 (text), Phase 2 (binary) with separate loaders
#   • Checkpoint save/resume with full optimizer + scheduler state
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import math
import random
import time
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import contextlib

@contextlib.contextmanager
def _autocast(enabled: bool):
    """AMP context manager — uses torch.amp (non-deprecated) API."""
    with torch.amp.autocast('cuda', enabled=enabled):
        yield
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(
    logits:      torch.Tensor,   # [B, T, 256]
    targets:     torch.Tensor,   # [B, T]
    aux_loss:    torch.Tensor,   # scalar from model
    exit_logits: List[torch.Tensor],  # list of [B, S+1, 256] patch-level logits
    patch_size:  int = 4,
    aux_weight:  float = 1.0,
    distill_weight: float = 0.2,
) -> tuple[torch.Tensor, float]:
    """
    Returns (total_loss, bpc).

    Losses:
      1. Main NLL on byte targets
      2. Auxiliary model losses (router load-balance + exit probe NLL)
      3. Exit distillation: early exit logits should match final logits
    """
    B, T, V = logits.shape

    # 1. Main NLL
    nll = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
    bpc = nll.item() / math.log(2)

    total = nll + aux_weight * aux_loss

    # 2. Distillation loss from exit gates → teach early blocks to predict
    if exit_logits and distill_weight > 0:
        with torch.no_grad():
            # Downsample final logits to patch resolution for comparison
            # Take logit at first byte of each patch
            final_patch = logits[:, ::patch_size, :]         # [B, S, V]

        for ex_log in exit_logits:
            # ex_log is [B, S+1, V] (includes format token position 0)
            ex_log_trimmed = ex_log[:, 1:, :]                # [B, S, V]
            # Trim to same size
            min_s = min(ex_log_trimmed.shape[1], final_patch.shape[1])
            kl = F.kl_div(
                F.log_softmax(ex_log_trimmed[:, :min_s, :], dim=-1),
                F.softmax(final_patch[:, :min_s, :].detach(), dim=-1),
                reduction='batchmean',
            )
            total = total + distill_weight * kl

    return total, bpc


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule: warmup + cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def _lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * p))


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class HERMESTrainer:
    """
    Two-phase curriculum trainer for HERMES.

    Usage:
        trainer = HERMESTrainer(model, ckpt_dir, device)
        trainer.run_phase(trn_loader, val_loader, n_epochs=15, phase_name='text')
        trainer.run_phase(trn_loader2, val_loader2, n_epochs=25, lr=5e-5, phase_name='binary')
        trainer.export_ema('hermes_ema.pt')
    """

    def __init__(
        self,
        model:       nn.Module,
        ckpt_dir:    str,
        device:      torch.device,
        lr:          float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int  = 500,
        grad_clip:   float = 1.0,
        accum_steps: int   = 8,
        ema_decay:   float = 0.9995,
        otta_prob:   float = 0.3,    # prob of applying OTTA meta-loss per step
    ):
        self.model      = model
        self.ckpt_dir   = ckpt_dir
        self.device     = device
        self.grad_clip  = grad_clip
        self.accum_steps = accum_steps
        self.otta_prob  = otta_prob
        os.makedirs(ckpt_dir, exist_ok=True)

        # Optimizer (AdamW with decoupled weight decay)
        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr,
            weight_decay=weight_decay, betas=(0.9, 0.95),
            fused=True if device.type == 'cuda' else False,
        )

        # Scaler for AMP
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

        # EMA model (used for inference / export)
        self.ema_model = AveragedModel(
            model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay)
        )

        # Bookkeeping
        self.global_step = 0
        self.best_bpc    = float('inf')
        self.history: dict = {'train_bpc': [], 'val_bpc': []}

        # Scheduler placeholder — set per phase
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        self.warmup_steps = warmup_steps

    # ── phase entry point ─────────────────────────────────────────────────────

    def run_phase(
        self,
        trn_loader:  DataLoader,
        val_loader:  DataLoader,
        n_epochs:    int,
        lr:          Optional[float] = None,
        phase_name:  str = 'phase',
        resume_path: Optional[str]  = None,
    ) -> float:
        """Train for n_epochs and return best validation BPC."""

        if lr is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = lr

        total_steps = n_epochs * len(trn_loader) // self.accum_steps
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda s: _lr_lambda(s, self.warmup_steps, total_steps),
        )

        start_epoch = 0
        if resume_path and os.path.exists(resume_path):
            start_epoch = self._load(resume_path)
            print(f'[{phase_name}] Resumed from epoch {start_epoch}')

        for epoch in range(start_epoch, n_epochs):
            t0 = time.time()
            train_bpc = self._train_epoch(trn_loader, epoch, phase_name)
            val_bpc   = self._val_epoch(val_loader)
            elapsed   = time.time() - t0

            self.history['train_bpc'].append(train_bpc)
            self.history['val_bpc'].append(val_bpc)

            print(f'[{phase_name}] Epoch {epoch+1:3d}/{n_epochs} | '
                  f'Train {train_bpc:.4f} BPC | Val {val_bpc:.4f} BPC | '
                  f'{elapsed:.0f}s')

            self._save(epoch, val_bpc, label=f'{phase_name}_latest')
            if val_bpc < self.best_bpc:
                self.best_bpc = val_bpc
                self._save(epoch, val_bpc, label='best')
                print(f'  ★ New best: {val_bpc:.4f} BPC')

        return self.best_bpc

    # ── training epoch ────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, epoch: int,
                     phase_name: str) -> float:
        self.model.train()
        total_bpc, n_updates = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)

        for micro_step, (bx, by, fmt) in enumerate(loader):
            bx  = bx.to(self.device, non_blocking=True)
            by  = by.to(self.device, non_blocking=True)
            fmt = fmt.to(self.device, non_blocking=True)

            with _autocast(enabled=(self.device.type == 'cuda')):
                logits, h_list, aux_loss, exit_logits = self.model(
                    bx, fmt, h_list=None, targets=by, training=True
                )
                loss, bpc = compute_loss(
                    logits, by, aux_loss, exit_logits,
                    patch_size=self.model.patch_size,
                )

                # ── OTTA meta-loss ─────────────────────────────────────────
                # Simulate mid-file cold-start: reset SSM and re-run the
                # second half of the sequence, penalising recovery failure.
                if random.random() < self.otta_prob:
                    T = bx.shape[1]
                    split = max(32, T // 3)
                    bx2   = bx[:, split:]
                    by2   = by[:, split:]
                    # Reset Mamba state (simulate mid-file start)
                    logits2, _, aux2, _ = self.model(
                        bx2, fmt, h_list=None, targets=by2, training=True
                    )
                    otta_nll = F.cross_entropy(
                        logits2.reshape(-1, 256), by2.reshape(-1)
                    )
                    loss = loss + 0.2 * (otta_nll + aux2)

                loss = loss / self.accum_steps

            self.scaler.scale(loss).backward()

            # Optimizer step every accum_steps micro-batches
            if (micro_step + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler:
                    self.scheduler.step()
                self.ema_model.update_parameters(self.model)
                self.global_step += 1

                total_bpc += bpc
                n_updates  += 1

                if self.global_step % 50 == 0:
                    lr_now = self.optimizer.param_groups[0]['lr']
                    print(f'  Step {self.global_step:6d} | '
                          f'BPC {bpc:.4f} | LR {lr_now:.2e}')

        return total_bpc / max(n_updates, 1)

    # ── validation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for bx, by, fmt in loader:
            bx  = bx.to(self.device, non_blocking=True)
            by  = by.to(self.device, non_blocking=True)
            fmt = fmt.to(self.device, non_blocking=True)
            with _autocast(enabled=(self.device.type == 'cuda')):
                logits, _, aux, exits = self.model(
                    bx, fmt, h_list=None, targets=None, training=False
                )
            nll = F.cross_entropy(logits.reshape(-1, 256), by.reshape(-1))
            total += nll.item() / math.log(2)
            n += 1
        self.model.train()
        return total / max(n, 1)

    # ── checkpoint ───────────────────────────────────────────────────────────

    def _save(self, epoch: int, bpc: float, label: str = 'latest'):
        path = os.path.join(self.ckpt_dir, f'{label}.pt')
        torch.save({
            'epoch':      epoch,
            'global_step': self.global_step,
            'bpc':        bpc,
            'best_bpc':   self.best_bpc,
            'model':      self.model.state_dict(),
            'ema_model':  self.ema_model.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'scaler':     self.scaler.state_dict(),
            'scheduler':  self.scheduler.state_dict() if self.scheduler else None,
            'history':    self.history,
            'model_config': self.model.config(),
        }, path)

    def _load(self, path: str) -> int:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.ema_model.load_state_dict(ckpt['ema_model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scaler.load_state_dict(ckpt['scaler'])
        if ckpt.get('scheduler') and self.scheduler:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        self.best_bpc    = ckpt['best_bpc']
        self.global_step = ckpt['global_step']
        self.history     = ckpt.get('history', self.history)
        return ckpt['epoch'] + 1

    # ── EMA export ───────────────────────────────────────────────────────────

    def export_ema(self, path: str):
        """Copy EMA weights into the base model and save state_dict."""
        # Update base model with EMA weights
        for p_ema, p_base in zip(self.ema_model.module.parameters(),
                                  self.model.parameters()):
            p_base.data.copy_(p_ema.data)
        torch.save({
            'model': self.model.state_dict(),
            'model_config': self.model.config(),
        }, path)
        print(f'EMA weights exported → {path}')
