# benchmarks/benchmark.py
from __future__ import annotations

import gzip
import bz2
import lzma
import math
import os
import time
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Reference compressors (no external deps — stdlib only)
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_COMPRESSORS: Dict[str, callable] = {
    'gzip': lambda d: gzip.compress(d, compresslevel=9),
    'bz2':  lambda d: bz2.compress(d, compresslevel=9),
    'lzma': lambda d: lzma.compress(d, preset=9),
}

def _bpc(compressed: bytes, original: bytes) -> float:
    return len(compressed) * 8 / len(original)


# ─────────────────────────────────────────────────────────────────────────────
# Result data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FileResult:
    name:            str
    size:            int
    category:        str          # 'text' | 'structured' | 'binary' | 'incompressible'
    hermes_bpc:      float
    hermes_ratio:    float        # original / compressed size
    hermes_time_ms:  float
    roundtrip_ok:    bool
    reference_bpc:   Dict[str, float] = field(default_factory=dict)
    error:           Optional[str] = None

    @property
    def best_reference_bpc(self) -> float:
        return min(self.reference_bpc.values()) if self.reference_bpc else float('inf')

    @property
    def beats_best_reference(self) -> bool:
        return self.hermes_bpc < self.best_reference_bpc


@dataclass
class BenchmarkResult:
    iteration:     int
    timestamp:     str
    files:         List[FileResult]
    overall_score: float
    passed:        bool
    failures:      List[str]
    category_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def avg_hermes_bpc(self) -> float:
        valid = [f.hermes_bpc for f in self.files if f.error is None]
        return sum(valid) / len(valid) if valid else float('inf')

    @property
    def roundtrip_pass_rate(self) -> float:
        total = len(self.files)
        ok    = sum(1 for f in self.files if f.roundtrip_ok)
        return ok / total if total else 0.0

    @property
    def files_beating_gzip(self) -> int:
        return sum(
            1 for f in self.files
            if f.error is None and 'gzip' in f.reference_bpc
            and f.hermes_bpc < f.reference_bpc['gzip']
        )

    def summary_str(self) -> str:
        lines = [
            f'  Score : {self.overall_score:.1f}/100  '
            f'({"PASSED ✅" if self.passed else "FAILED ❌"})',
            f'  BPC   : {self.avg_hermes_bpc:.4f} avg',
            f'  RT    : {self.roundtrip_pass_rate*100:.0f}% roundtrip ok',
            f'  Beats gzip in {self.files_beating_gzip}/{len(self.files)} files',
        ]
        if self.failures:
            lines.append('  Failures:')
            for f in self.failures:
                lines.append(f'    • {f}')
        return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Test corpus
# ─────────────────────────────────────────────────────────────────────────────

def _build_test_cases() -> List[Tuple[str, bytes, str]]:
    """Returns list of (name, data_bytes, category)."""
    rng   = np.random.RandomState(42)
    cases = []

    # ── Text ─────────────────────────────────────────────────────────────────
    para = (
        "The quick brown fox jumps over the lazy dog. "
        "Neural byte-level compression models learn to predict the next byte "
        "given all preceding bytes, achieving near-entropy coding on structured data. "
        "HERMES combines selective state-space models with sparse mixture-of-experts "
        "attention to handle both short-range and long-range dependencies.\n"
    )
    cases += [
        ('text_eng_4k',   (para * 20)[:4096].encode(),   'text'),
        ('text_eng_16k',  (para * 80)[:16384].encode(),  'text'),
    ]

    src = (
        "import torch\nimport torch.nn as nn\nfrom typing import Optional\n\n"
        "class ByteModel(nn.Module):\n"
        "    def __init__(self, vocab: int = 256, d: int = 512) -> None:\n"
        "        super().__init__()\n"
        "        self.emb  = nn.Embedding(vocab, d)\n"
        "        self.head = nn.Linear(d, vocab, bias=False)\n\n"
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "        return self.head(self.emb(x))\n\n"
        "    def n_params(self) -> int:\n"
        "        return sum(p.numel() for p in self.parameters())\n\n"
    )
    cases += [
        ('source_py_4k',  (src * 20)[:4096].encode(),    'text'),
        ('source_py_16k', (src * 80)[:16384].encode(),   'text'),
    ]

    # ── Structured ────────────────────────────────────────────────────────────
    json_rows = [
        '{"id":%d,"bpc":%.4f,"label":"item_%04d","active":true}\n' % (i, i * 0.0001, i)
        for i in range(800)
    ]
    cases.append(('json_8k', ''.join(json_rows).encode()[:8192], 'structured'))

    xml_rows = ['<entry id="%d"><val>%.6f</val></entry>\n' % (i, i * 1e-6) for i in range(400)]
    cases.append(('xml_8k',  ''.join(xml_rows).encode()[:8192],  'structured'))

    # ── Binary ────────────────────────────────────────────────────────────────
    cases += [
        ('zeros_4k',         bytes(4096),                                        'binary'),
        ('zeros_16k',        bytes(16384),                                       'binary'),
        ('incremental_4k',   bytes(range(256)) * 16,                             'binary'),
        ('repeated_byte_4k', b'\xAB' * 4096,                                    'binary'),
    ]

    elf_hdr = b'\x7fELF\x02\x01\x01\x00' + bytes(8) + b'\x02\x00\x3e\x00\x01\x00\x00\x00'
    elf_body = rng.randint(0, 256, 4096 - len(elf_hdr), dtype=np.uint8).tobytes()
    cases.append(('elf_like_4k', elf_hdr + elf_body, 'binary'))

    # Try a real system binary
    for candidate in ('/bin/ls', '/usr/bin/python3', '/bin/bash', '/usr/bin/ls'):
        if os.path.exists(candidate):
            try:
                with open(candidate, 'rb') as fh:
                    data = fh.read(8192)
                cases.append((f'elf_{os.path.basename(candidate)}_8k', data, 'binary'))
                break
            except OSError:
                pass

    # ── Incompressible ────────────────────────────────────────────────────────
    cases += [
        ('random_4k',  rng.randint(0, 256, 4096,  dtype=np.uint8).tobytes(), 'incompressible'),
        ('random_16k', rng.randint(0, 256, 16384, dtype=np.uint8).tobytes(), 'incompressible'),
    ]

    # ── Mixed ─────────────────────────────────────────────────────────────────
    mixed = (
        para.encode() * 15
        + bytes(512)
        + rng.randint(0, 256, 512, dtype=np.uint8).tobytes()
        + src.encode() * 5
    )
    cases.append(('mixed_8k', mixed[:8192], 'structured'))

    return cases


# ─────────────────────────────────────────────────────────────────────────────
# Suite
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkSuite:
    """
    Runs HERMES on a fixed synthetic + real-file corpus and compares against
    gzip / bz2 / lzma. Produces a BenchmarkResult with per-file details.
    """

    def __init__(self, chunk_size: int = 4096):
        self.chunk_size  = chunk_size
        self._test_cases = _build_test_cases()

    def run(
        self,
        model:     torch.nn.Module,
        device:    torch.device,
        iteration: int = 0,
    ) -> BenchmarkResult:
        from coding.coder import compress, decompress
        from benchmarks.thresholds import score_result, PASS_THRESHOLD

        file_results: List[FileResult] = []
        failures:     List[str]        = []

        print(f'\n{"─"*60}')
        print(f' Benchmark  (iteration {iteration})')
        print(f'{"─"*60}')
        print(f'  {"Name":<24} {"Size":>6}  {"HERMES":>7}  {"gzip":>7}  {"bz2":>7}  {"lzma":>7}  RT')

        for name, data, category in self._test_cases:
            # Reference compressors
            ref_bpc: Dict[str, float] = {}
            for cname, cfn in REFERENCE_COMPRESSORS.items():
                try:
                    ref_bpc[cname] = _bpc(cfn(data), data)
                except Exception:
                    pass

            # HERMES compress + decompress
            error: Optional[str] = None
            hermes_bpc = float('inf')
            hermes_ratio = 0.0
            elapsed_ms = 0.0
            rt_ok = False

            try:
                t0           = time.perf_counter()
                compressed   = compress(data, model, device,
                                        chunk_size=self.chunk_size, verbose=False)
                elapsed_ms   = (time.perf_counter() - t0) * 1000
                decompressed = decompress(compressed, model, device,
                                          chunk_size=self.chunk_size, verbose=False)
                rt_ok        = decompressed == data
                hermes_bpc   = _bpc(compressed, data)
                hermes_ratio = len(data) / len(compressed)

                if not rt_ok:
                    failures.append(f'{name}: roundtrip FAILED')

            except Exception as exc:
                error = str(exc)
                failures.append(f'{name}: error — {exc}')

            rt_mark = '✅' if rt_ok else '❌'
            print(
                f'  {name:<24} {len(data):>5}B '
                f'  {hermes_bpc:>6.3f}'
                f'  {ref_bpc.get("gzip", 0):>6.3f}'
                f'  {ref_bpc.get("bz2", 0):>6.3f}'
                f'  {ref_bpc.get("lzma", 0):>6.3f}'
                f'  {rt_mark}'
            )

            file_results.append(FileResult(
                name=name, size=len(data), category=category,
                hermes_bpc=hermes_bpc, hermes_ratio=hermes_ratio,
                hermes_time_ms=elapsed_ms, roundtrip_ok=rt_ok,
                reference_bpc=ref_bpc, error=error,
            ))

        score, score_failures, cat_scores = score_result(file_results)
        failures.extend(score_failures)
        roundtrip_ok = all(f.roundtrip_ok for f in file_results if f.error is None)
        passed = (score >= PASS_THRESHOLD) and roundtrip_ok

        result = BenchmarkResult(
            iteration=iteration,
            timestamp=datetime.datetime.now().isoformat(timespec='seconds'),
            files=file_results,
            overall_score=score,
            passed=passed,
            failures=failures,
            category_scores=cat_scores,
        )

        print(f'\n{result.summary_str()}')
        return result
