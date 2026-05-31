# benchmarks/retrain_loop.py
#
# BenchmarkDrivenRetrainer — runs benchmark → fine-tunes on failing categories
# → re-evaluates, indefinitely until passing or max_iterations reached.
#
from __future__ import annotations

import os
import math
from typing import List, Optional

import torch
import torch.nn.functional as F

from .benchmark import BenchmarkSuite, BenchmarkResult
from .thresholds import PASS_THRESHOLD, PERFECT_THRESHOLD, CATEGORY_CRITERIA
from .visualizer import BenchmarkVisualizer


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tune schedule
# ─────────────────────────────────────────────────────────────────────────────

_FINETUNE_CFG = {
    #  Iter 0 → lots of epochs, high LR; decays geometrically
    'base_lr':        2e-5,
    'lr_decay':       0.65,    # multiply LR each failed iteration
    'base_epochs':    8,
    'epoch_decay':    0.80,    # multiply epoch count each iteration
    'min_epochs':     3,
    'min_lr':         1e-7,
    'n_samples_text': 6_000,
    'n_samples_bin':  4_000,
    'batch_size':     4,
    'seq_len_text':   1024,
    'seq_len_bin':    2048,
    'accum_steps':    8,
}


def _failing_categories(result: BenchmarkResult) -> List[str]:
    """Return categories whose score is below 50 %."""
    return [
        cat for cat, s in result.category_scores.items()
        if s < 0.50
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Retrainer
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkDrivenRetrainer:
    """
    Evaluates HERMES repeatedly and fine-tunes it on failing categories until
    the benchmark passes (or max_iterations is reached / score stagnates).

    Usage:
        retrainer = BenchmarkDrivenRetrainer(model, trainer, corpus_builder,
                                             device, out_dir)
        model = retrainer.run(max_iterations=None)   # run until pass/perfect
    """

    def __init__(
        self,
        model,           # HERMES
        trainer,         # HERMESTrainer (already initialised, EMA exported)
        corpus_builder,  # CorpusBuilder
        device:          torch.device,
        out_dir:         str,
        chunk_size:      int = 4096,
        target_score:    float = PERFECT_THRESHOLD,
    ):
        self.model          = model
        self.trainer        = trainer
        self.corpus_builder = corpus_builder
        self.device         = device
        self.out_dir        = out_dir
        self.chunk_size     = chunk_size
        self.target_score   = target_score

        self.suite     = BenchmarkSuite(chunk_size=chunk_size)
        self.viz       = BenchmarkVisualizer()
        self.history:  List[BenchmarkResult] = []
        self._best_score  = 0.0
        self._stagnation  = 0     # consecutive iters without ≥1 point improvement

        os.makedirs(out_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, max_iterations: Optional[int] = None) -> object:
        """
        Main loop: benchmark → (fine-tune if failing) → repeat.
        Stops when: score >= target_score, max_iterations reached, or
        score stagnates for 4 consecutive iterations.
        Returns the model with best weights loaded.
        """
        iteration = 0
        print('\n' + '═' * 60)
        print('  HERMES Benchmark-Driven Retrain Loop')
        print(f'  Target score : {self.target_score:.0f} / 100')
        if max_iterations:
            print(f'  Max iterations: {max_iterations}')
        print('═' * 60)

        while True:
            # ── Benchmark ────────────────────────────────────────────────────
            result = self.suite.run(self.model, self.device, iteration)
            self.history.append(result)
            self._save_plots(result, iteration)

            # ── Check pass condition ──────────────────────────────────────────
            if result.overall_score >= self.target_score:
                print(f'\n★ Target reached: {result.overall_score:.1f} ≥ {self.target_score:.0f}')
                break

            # ── Stagnation guard ─────────────────────────────────────────────
            improvement = result.overall_score - self._best_score
            if improvement >= 1.0:
                self._best_score = result.overall_score
                self._stagnation = 0
                self._save_best_checkpoint(iteration)
            else:
                self._stagnation += 1
                if self._stagnation >= 4:
                    print(f'\n⚠ Score stagnated for 4 iterations at {self._best_score:.1f}. Stopping.')
                    break

            # ── Max iterations ────────────────────────────────────────────────
            if max_iterations is not None and iteration >= max_iterations - 1:
                print(f'\n⚠ Reached max_iterations={max_iterations}. Stopping.')
                break

            # ── Fine-tune on failing categories ───────────────────────────────
            iteration += 1
            self._fine_tune(result, iteration)

        # Load best weights before returning
        best_ckpt = os.path.join(self.out_dir, 'benchmark_best.pt')
        if os.path.exists(best_ckpt):
            ckpt = torch.load(best_ckpt, map_location='cpu', weights_only=False)
            self.model.load_state_dict(ckpt['model'])
            self.model.to(self.device).eval()
            print(f'  Best checkpoint loaded (score {ckpt["score"]:.1f})')

        self._save_final_report()
        return self.model

    # ── Fine-tuning ───────────────────────────────────────────────────────────

    def _fine_tune(self, result: BenchmarkResult, iteration: int) -> None:
        from training.data_pipeline import build_loaders

        failing = _failing_categories(result)
        needs_text   = any(c in ('text', 'structured')     for c in failing) or not failing
        needs_binary = any(c in ('binary', 'incompressible') for c in failing) or not failing

        # Decay schedule
        lr     = max(_FINETUNE_CFG['min_lr'],
                     _FINETUNE_CFG['base_lr'] * (_FINETUNE_CFG['lr_decay'] ** (iteration - 1)))
        epochs = max(_FINETUNE_CFG['min_epochs'],
                     int(_FINETUNE_CFG['base_epochs'] * (_FINETUNE_CFG['epoch_decay'] ** (iteration - 1))))

        print(f'\n{"─"*60}')
        print(f'  Fine-tune  iter={iteration}  lr={lr:.2e}  epochs={epochs}')
        print(f'  Failing categories: {failing if failing else "none (score low overall)"}')
        print(f'{"─"*60}')

        for g in self.trainer.optimizer.param_groups:
            g['lr'] = lr

        if needs_text:
            print('  → Text fine-tune …')
            bufs = self.corpus_builder.build_text_buffers()
            trn, val = build_loaders(
                bufs,
                seq_len=_FINETUNE_CFG['seq_len_text'],
                n_samples=_FINETUNE_CFG['n_samples_text'],
                batch_size=_FINETUNE_CFG['batch_size'],
            )
            self.trainer.run_phase(trn, val, n_epochs=epochs,
                                   lr=lr, phase_name=f'ft{iteration}_text')

        if needs_binary:
            print('  → Binary fine-tune …')
            bufs = self.corpus_builder.build_binary_buffers()
            trn, val = build_loaders(
                bufs,
                seq_len=_FINETUNE_CFG['seq_len_bin'],
                n_samples=_FINETUNE_CFG['n_samples_bin'],
                batch_size=_FINETUNE_CFG['batch_size'],
            )
            self.trainer.run_phase(trn, val, n_epochs=epochs,
                                   lr=lr, phase_name=f'ft{iteration}_binary')

        # Re-export EMA weights so the benchmark tests the fine-tuned model
        ema_path = os.path.join(self.out_dir, 'hermes_ema.pt')
        self.trainer.export_ema(ema_path)
        ckpt = torch.load(ema_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device).eval()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _save_best_checkpoint(self, iteration: int) -> None:
        path = os.path.join(self.out_dir, 'benchmark_best.pt')
        torch.save({
            'model':     self.model.state_dict(),
            'score':     self._best_score,
            'iteration': iteration,
        }, path)
        print(f'  ★ New best checkpoint  score={self._best_score:.1f}  → {path}')

    def _save_plots(self, result: BenchmarkResult, iteration: int) -> None:
        report_path = os.path.join(
            self.out_dir, f'benchmark_report_iter{iteration:03d}.png'
        )
        self.viz.plot_benchmark_report(result, report_path)

        if len(self.history) > 1:
            hist_path = os.path.join(self.out_dir, 'benchmark_iteration_history.png')
            self.viz.plot_iteration_history(self.history, hist_path)

    def _save_final_report(self) -> None:
        if not self.history:
            return
        final = self.history[-1]
        print('\n' + '═' * 60)
        print('  RETRAIN LOOP COMPLETE')
        print(f'  Iterations run : {len(self.history)}')
        print(f'  Best score     : {self._best_score:.1f} / 100')
        print(f'  Final score    : {final.overall_score:.1f} / 100')
        status = ('PERFECT ★' if final.overall_score >= PERFECT_THRESHOLD
                  else 'PASSED ✓' if final.overall_score >= PASS_THRESHOLD
                  else 'FAILED ✗')
        print(f'  Status         : {status}')
        print('═' * 60)

        # Summary plot
        dashboard_path = os.path.join(self.out_dir, 'benchmark_final_dashboard.png')
        self.viz.plot_iteration_history(self.history, dashboard_path)
