# benchmarks/visualizer.py
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from .benchmark import BenchmarkResult, FileResult
from .thresholds import PASS_THRESHOLD, PERFECT_THRESHOLD, CATEGORY_CRITERIA


# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────

C = {
    'hermes':  '#1565C0',   # deep blue
    'gzip':    '#E65100',   # deep orange
    'bz2':     '#6A1B9A',   # deep purple
    'lzma':    '#1B5E20',   # deep green
    'pass':    '#2E7D32',
    'fail':    '#C62828',
    'warn':    '#E65100',
    'bg':      '#F5F7FA',
    'grid':    '#DDE1E7',
    'panel':   '#FFFFFF',
    'text':    '#1A1A2E',
    'muted':   '#5C6370',
    'accent':  '#0D47A1',
}

CAT_COLORS = {
    'text':           '#1565C0',
    'structured':     '#00838F',
    'binary':         '#6A1B9A',
    'incompressible': '#827717',
}


def _apply_style() -> None:
    plt.rcParams.update({
        'font.family':          'DejaVu Sans',
        'font.size':            9,
        'axes.facecolor':       C['bg'],
        'figure.facecolor':     C['panel'],
        'axes.edgecolor':       C['grid'],
        'axes.linewidth':       0.8,
        'axes.grid':            True,
        'grid.color':           C['grid'],
        'grid.linestyle':       '--',
        'grid.alpha':           0.7,
        'axes.spines.top':      False,
        'axes.spines.right':    False,
        'xtick.color':          C['muted'],
        'ytick.color':          C['muted'],
        'text.color':           C['text'],
        'axes.labelcolor':      C['text'],
        'axes.titlesize':       10,
        'axes.titleweight':     'bold',
        'axes.labelsize':       9,
        'legend.fontsize':      8,
        'legend.framealpha':    0.9,
    })


def _score_color(score: float) -> str:
    if score >= PERFECT_THRESHOLD:
        return C['pass']
    if score >= PASS_THRESHOLD:
        return '#558B2F'
    if score >= PASS_THRESHOLD * 0.7:
        return C['warn']
    return C['fail']


# ─────────────────────────────────────────────────────────────────────────────
# Individual panels
# ─────────────────────────────────────────────────────────────────────────────

def _panel_score_gauge(ax: plt.Axes, score: float, iteration: int) -> None:
    """Horizontal score bar with coloured zones."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Overall Score (0 – 100)')
    ax.set_title(f'Iteration {iteration} — Score')

    # Background zones
    ax.barh(0.5, PASS_THRESHOLD,           height=0.55, left=0,
            color='#FFEBEE', zorder=1)
    ax.barh(0.5, PERFECT_THRESHOLD - PASS_THRESHOLD, height=0.55,
            left=PASS_THRESHOLD, color='#E8F5E9', zorder=1)
    ax.barh(0.5, 100 - PERFECT_THRESHOLD, height=0.55,
            left=PERFECT_THRESHOLD, color='#E3F2FD', zorder=1)

    # Score bar
    bar_color = _score_color(score)
    ax.barh(0.5, score, height=0.40, color=bar_color, zorder=2, alpha=0.9)

    # Zone labels
    ax.text(PASS_THRESHOLD / 2, 0.13, 'FAIL', ha='center', va='center',
            fontsize=7, color='#C62828', fontweight='bold')
    ax.text((PASS_THRESHOLD + PERFECT_THRESHOLD) / 2, 0.13, 'PASS',
            ha='center', va='center', fontsize=7, color='#2E7D32', fontweight='bold')
    ax.text((PERFECT_THRESHOLD + 100) / 2, 0.13, 'PERFECT',
            ha='center', va='center', fontsize=7, color='#0D47A1', fontweight='bold')

    # Threshold lines
    for xval in (PASS_THRESHOLD, PERFECT_THRESHOLD):
        ax.axvline(xval, color=C['muted'], lw=1.2, ls=':', zorder=3)

    # Score label
    ax.text(min(score + 2, 97), 0.5, f'{score:.1f}',
            va='center', fontsize=14, fontweight='bold', color=bar_color, zorder=4)

    status = ('PERFECT ★' if score >= PERFECT_THRESHOLD
              else 'PASSED ✓' if score >= PASS_THRESHOLD else 'FAILED ✗')
    ax.text(99, 0.88, status, ha='right', va='top', fontsize=9,
            fontweight='bold', color=bar_color)


def _panel_bpc_comparison(ax: plt.Axes, files: List[FileResult]) -> None:
    """Grouped bar chart: HERMES vs gzip/bz2/lzma per file."""
    valid  = [f for f in files if f.error is None]
    names  = [f.name.replace('_', '\n') for f in valid]
    x      = np.arange(len(valid))
    width  = 0.20
    refs   = ['gzip', 'bz2', 'lzma']
    colors = [C['gzip'], C['bz2'], C['lzma']]
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    # HERMES bars
    hermes_bpc = [f.hermes_bpc for f in valid]
    ax.bar(x + offsets[0], hermes_bpc, width, label='HERMES',
           color=C['hermes'], alpha=0.9, zorder=3)

    for i, (ref, col) in enumerate(zip(refs, colors)):
        ref_vals = [f.reference_bpc.get(ref, 0) for f in valid]
        ax.bar(x + offsets[i + 1], ref_vals, width, label=ref.upper(),
               color=col, alpha=0.75, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel('Bits per Byte (BPC)')
    ax.set_title('BPC Comparison — HERMES vs Reference Compressors')
    ax.legend(loc='upper right', ncol=4)

    # Highlight files where HERMES beats gzip
    for i, f in enumerate(valid):
        if 'gzip' in f.reference_bpc and f.hermes_bpc < f.reference_bpc['gzip']:
            ax.text(i + offsets[0], f.hermes_bpc + 0.05, '★',
                    ha='center', va='bottom', fontsize=8, color='#FFD600')


def _panel_category_scores(ax: plt.Axes, cat_scores: Dict[str, float]) -> None:
    """Horizontal bar chart of per-category scores (0–100)."""
    cats   = list(cat_scores.keys())
    scores = [cat_scores[c] * 100 for c in cats]
    colors = [CAT_COLORS.get(c, C['hermes']) for c in cats]

    y = np.arange(len(cats))
    bars = ax.barh(y, scores, color=colors, alpha=0.85, height=0.55, zorder=3)

    ax.set_xlim(0, 110)
    ax.set_yticks(y)
    ax.set_yticklabels([c.capitalize() for c in cats])
    ax.set_xlabel('Category Score (0–100)')
    ax.set_title('Score by Data Category')
    ax.axvline(PASS_THRESHOLD, color=C['fail'], lw=1.2, ls='--', zorder=4,
               label=f'Pass ({PASS_THRESHOLD:.0f})')
    ax.axvline(PERFECT_THRESHOLD, color=C['pass'], lw=1.2, ls='--', zorder=4,
               label=f'Perfect ({PERFECT_THRESHOLD:.0f})')
    ax.legend(fontsize=7)

    for bar, s in zip(bars, scores):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f'{s:.0f}', va='center', fontsize=8, fontweight='bold')


def _panel_roundtrip(ax: plt.Axes, files: List[FileResult]) -> None:
    """Pass/fail grid for roundtrip correctness."""
    valid = [f for f in files if f.error is None]
    n = len(valid)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Roundtrip Correctness')
    ax.invert_yaxis()

    for idx, f in enumerate(valid):
        row, col = divmod(idx, cols)
        color = C['pass'] if f.roundtrip_ok else C['fail']
        mark  = '✓' if f.roundtrip_ok else '✗'
        ax.add_patch(plt.Rectangle(
            (col - 0.45, row - 0.40), 0.90, 0.80,
            color=color, alpha=0.85, zorder=2,
        ))
        ax.text(col, row - 0.02, mark, ha='center', va='center',
                fontsize=12, color='white', fontweight='bold', zorder=3)
        ax.text(col, row + 0.35, f.name[:10], ha='center', va='center',
                fontsize=5.5, color='white', zorder=3)


def _panel_bpc_scatter(ax: plt.Axes, files: List[FileResult]) -> None:
    """Scatter: HERMES BPC vs best reference BPC, coloured by category."""
    valid = [f for f in files if f.error is None and f.best_reference_bpc < float('inf')]
    if not valid:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    for cat, col in CAT_COLORS.items():
        cat_f = [f for f in valid if f.category == cat]
        if not cat_f:
            continue
        ax.scatter(
            [f.best_reference_bpc for f in cat_f],
            [f.hermes_bpc for f in cat_f],
            color=col, label=cat.capitalize(), s=60, alpha=0.85, zorder=3,
        )

    lo = min(f.best_reference_bpc for f in valid) * 0.85
    hi = max(max(f.best_reference_bpc for f in valid),
             max(f.hermes_bpc for f in valid)) * 1.05
    ax.plot([lo, hi], [lo, hi], '--', color=C['muted'], lw=1.2, label='HERMES = Ref', zorder=2)
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], alpha=0.06, color=C['pass'],
                    label='HERMES beats ref')

    ax.set_xlabel('Best Reference BPC (min of gzip/bz2/lzma)')
    ax.set_ylabel('HERMES BPC')
    ax.set_title('HERMES vs Best Classical Compressor')
    ax.legend(fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Main figures
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkVisualizer:

    def plot_benchmark_report(
        self,
        result:    BenchmarkResult,
        save_path: str,
    ) -> None:
        """5-panel dashboard for one benchmark result."""
        _apply_style()
        fig = plt.figure(figsize=(18, 11), facecolor=C['panel'])
        fig.suptitle(
            f'HERMES Benchmark Report — Iteration {result.iteration} — {result.timestamp}',
            fontsize=13, fontweight='bold', color=C['text'], y=0.98,
        )

        gs = gridspec.GridSpec(
            3, 3,
            figure=fig,
            hspace=0.48, wspace=0.35,
            left=0.06, right=0.97, top=0.93, bottom=0.06,
        )

        # Row 0: score gauge (full width)
        ax_gauge = fig.add_subplot(gs[0, :])
        _panel_score_gauge(ax_gauge, result.overall_score, result.iteration)

        # Row 1: BPC comparison (full width)
        ax_bpc = fig.add_subplot(gs[1, :])
        _panel_bpc_comparison(ax_bpc, result.files)

        # Row 2: three panels
        ax_cat = fig.add_subplot(gs[2, 0])
        _panel_category_scores(ax_cat, result.category_scores)

        ax_rt = fig.add_subplot(gs[2, 1])
        _panel_roundtrip(ax_rt, result.files)

        ax_sc = fig.add_subplot(gs[2, 2])
        _panel_bpc_scatter(ax_sc, result.files)

        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Benchmark report → {save_path}')

    def plot_iteration_history(
        self,
        results:   List[BenchmarkResult],
        save_path: str,
    ) -> None:
        """Score + BPC progression across retrain iterations."""
        if not results:
            return
        _apply_style()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle('HERMES — Retrain Loop Progress', fontsize=12,
                     fontweight='bold', y=1.01)

        iters = [r.iteration for r in results]

        # Panel 0: Overall score
        ax = axes[0]
        scores = [r.overall_score for r in results]
        colors = [_score_color(s) for s in scores]
        ax.plot(iters, scores, 'o-', color=C['hermes'], lw=2, ms=7, zorder=3)
        for x, y, c in zip(iters, scores, colors):
            ax.scatter(x, y, color=c, s=80, zorder=4)
        ax.axhline(PASS_THRESHOLD,    color=C['fail'], ls='--', lw=1.2,
                   label=f'Pass ({PASS_THRESHOLD:.0f})')
        ax.axhline(PERFECT_THRESHOLD, color=C['pass'], ls='--', lw=1.2,
                   label=f'Perfect ({PERFECT_THRESHOLD:.0f})')
        ax.set_xlabel('Iteration'); ax.set_ylabel('Overall Score')
        ax.set_title('Score Progression'); ax.legend(fontsize=7)
        ax.set_ylim(0, 102)

        # Panel 1: Avg BPC over iterations
        ax = axes[1]
        avg_bpc = [r.avg_hermes_bpc for r in results]
        ax.plot(iters, avg_bpc, 'o-', color=C['hermes'], lw=2, ms=7, zorder=3)
        ax.set_xlabel('Iteration'); ax.set_ylabel('Average BPC')
        ax.set_title('Average BPC Trend')

        # Panel 2: Category scores stacked area (last 3 cats)
        ax = axes[2]
        cat_keys = list(CATEGORY_CRITERIA.keys())
        ys = np.array([
            [r.category_scores.get(c, 0.0) * 100 for c in cat_keys]
            for r in results
        ])
        for i, (cat, col) in enumerate(zip(cat_keys, CAT_COLORS.values())):
            ax.plot(iters, ys[:, i], 'o-', label=cat.capitalize(),
                    color=col, lw=1.8, ms=5)
        ax.axhline(PASS_THRESHOLD / 4, color=C['fail'], ls=':', lw=1, alpha=0.6)
        ax.set_xlabel('Iteration'); ax.set_ylabel('Category Score')
        ax.set_title('Category Score Progression')
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(0, 102)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Iteration history → {save_path}')

    def plot_training_dashboard(
        self,
        trainer_history: dict,
        benchmark_results: List[BenchmarkResult],
        save_path: str,
    ) -> None:
        """Combined training curve + benchmark score overlay."""
        _apply_style()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('HERMES Training Dashboard', fontsize=12, fontweight='bold')

        # Left: training BPC curve
        ax = axes[0]
        t_bpc = trainer_history.get('train_bpc', [])
        v_bpc = trainer_history.get('val_bpc',   [])
        if t_bpc:
            ax.plot(t_bpc, label='Train BPC', color=C['hermes'], lw=2)
        if v_bpc:
            ax.plot(v_bpc, label='Val BPC',   color=C['gzip'],   lw=2, ls='--')
        ax.set_xlabel('Epoch'); ax.set_ylabel('BPC')
        ax.set_title('Training / Validation BPC')
        ax.legend()

        # Right: benchmark score bar
        ax = axes[1]
        if benchmark_results:
            iters  = [r.iteration for r in benchmark_results]
            scores = [r.overall_score for r in benchmark_results]
            bar_colors = [_score_color(s) for s in scores]
            ax.bar(iters, scores, color=bar_colors, alpha=0.85, zorder=3)
            ax.axhline(PASS_THRESHOLD,    ls='--', lw=1.2, color=C['fail'],
                       label=f'Pass ({PASS_THRESHOLD:.0f})')
            ax.axhline(PERFECT_THRESHOLD, ls='--', lw=1.2, color=C['pass'],
                       label=f'Perfect ({PERFECT_THRESHOLD:.0f})')
            ax.set_xlabel('Benchmark Iteration'); ax.set_ylabel('Overall Score')
            ax.set_title('Benchmark Score per Iteration')
            ax.set_ylim(0, 105)
            ax.legend(fontsize=7)

            # Pass/fail stamps
            for x, s in zip(iters, scores):
                stamp = '✓' if s >= PASS_THRESHOLD else '✗'
                ax.text(x, s + 1.5, stamp, ha='center', va='bottom',
                        color='white' if s >= PASS_THRESHOLD else C['fail'],
                        fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No benchmark results yet',
                    ha='center', va='center', transform=ax.transAxes)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Training dashboard → {save_path}')
