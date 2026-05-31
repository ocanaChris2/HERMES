# benchmarks/thresholds.py
from __future__ import annotations
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Score thresholds
# ─────────────────────────────────────────────────────────────────────────────

PASS_THRESHOLD    = 62.0   # minimum overall score (0–100) to mark "passed"
PERFECT_THRESHOLD = 88.0   # threshold for "excellent / production-ready"

# Per-category: (excellent_bpc, acceptable_bpc, weight_in_overall_score)
#   score_cat = clamp((acceptable - actual) / (acceptable - excellent), 0, 1)
#   Roundtrip correctness is a hard gate (always 20 % of total score).
CATEGORY_CRITERIA: Dict[str, Tuple[float, float, float]] = {
    #  category          excellent   acceptable   weight
    'text':           (2.0,         4.8,          0.28),
    'structured':     (2.5,         5.5,          0.14),
    'binary':         (3.5,         7.5,          0.24),
    'incompressible': (7.6,         8.6,          0.08),
}

ROUNDTRIP_WEIGHT   = 0.20   # 20 % hard gate — all roundtrips must pass
GZIP_BEAT_BONUS    = 0.06   # up to 6 % bonus for beating gzip on every file


class BenchmarkCriteria:
    """Human-readable description of pass/fail criteria."""

    @staticmethod
    def describe() -> str:
        lines = [
            'Pass criteria (overall ≥ %.0f / 100):' % PASS_THRESHOLD,
            '  • Roundtrip   : 100 %% files decode correctly (hard gate, 20 %% weight)',
        ]
        for cat, (exc, acc, w) in CATEGORY_CRITERIA.items():
            lines.append(
                f'  • {cat:<14}: BPC < {acc:.1f} to score, '
                f'< {exc:.1f} for full marks  ({w*100:.0f} %% weight)'
            )
        lines += [
            f'  • Gzip-beat bonus : up to {GZIP_BEAT_BONUS*100:.0f} %% for beating gzip on all files',
            f'Perfect threshold : {PERFECT_THRESHOLD:.0f} / 100',
        ]
        return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_result(
    files: list,   # List[FileResult]  — avoid circular import with string annotation
) -> Tuple[float, List[str], Dict[str, float]]:
    """
    Returns (overall_score 0–100, list_of_failure_strings, per_category_scores).
    """
    failures: List[str]        = []
    cat_scores: Dict[str, float] = {}

    # ── Roundtrip score ───────────────────────────────────────────────────────
    valid = [f for f in files if f.error is None]
    rt_ok = sum(1 for f in valid if f.roundtrip_ok)
    rt_score = rt_ok / len(valid) if valid else 0.0
    if rt_ok < len(valid):
        bad = [f.name for f in valid if not f.roundtrip_ok]
        failures.append('Roundtrip failed: ' + ', '.join(bad))

    # ── Per-category BPC scores ───────────────────────────────────────────────
    for cat, (excellent, acceptable, _) in CATEGORY_CRITERIA.items():
        cat_files = [f for f in valid if f.category == cat]
        if not cat_files:
            cat_scores[cat] = 1.0
            continue
        avg_bpc = sum(f.hermes_bpc for f in cat_files) / len(cat_files)
        s = _clamp01((acceptable - avg_bpc) / (acceptable - excellent))
        cat_scores[cat] = s
        if s < 0.30:
            failures.append(
                f'{cat} BPC too high: avg {avg_bpc:.3f} '
                f'(excellent <{excellent}, acceptable <{acceptable})'
            )

    # ── Weighted total ────────────────────────────────────────────────────────
    total_w = ROUNDTRIP_WEIGHT + sum(w for _, _, w in CATEGORY_CRITERIA.values())
    raw = rt_score * ROUNDTRIP_WEIGHT
    for cat, (_, _, w) in CATEGORY_CRITERIA.items():
        raw += cat_scores.get(cat, 1.0) * w
    score = raw / total_w * 100.0   # normalise to 0–100

    # ── Gzip-beat bonus ───────────────────────────────────────────────────────
    gzip_files = [f for f in valid if 'gzip' in f.reference_bpc]
    if gzip_files:
        beats = sum(1 for f in gzip_files if f.hermes_bpc < f.reference_bpc['gzip'])
        bonus_frac = beats / len(gzip_files)
        score = min(100.0, score + bonus_frac * GZIP_BEAT_BONUS * 100.0)

    return score, failures, cat_scores
