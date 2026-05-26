# training/data_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# Stratified byte corpus for HERMES training.
#
# Sources:
#   Phase 1 (text):    HuggingFace wikitext-103-raw-v1  (streaming, no disk)
#                      HuggingFace codeparrot/github-code (streaming)
#   Phase 2 (binary):  System ELF binaries   (/usr/bin, /usr/lib)
#                      Python .pyc files      (site-packages)
#                      Silesia corpus         (downloaded to disk)
#                      Kaggle datasets        (if available)
#
# The StratifiedByteDataset interleaves samples across source categories
# so every training batch sees a mix of file types.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import glob
import random
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from hermes.format_sniffer import sniff


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_bytes(path: str, max_bytes: int = 50_000_000) -> bytes:
    with open(path, 'rb') as f:
        return f.read(max_bytes)


def _collect_system_binaries(max_per_dir: int = 300) -> List[str]:
    """Return paths to ELF/shared-lib files already on the machine."""
    # Non-recursive patterns only — recursive glob on /usr/lib can take
    # 30+ seconds on Kaggle's filesystem and get the kernel killed.
    patterns = [
        '/usr/bin/*',
        '/usr/lib/*.so*',
        '/usr/lib/x86_64-linux-gnu/*.so*',
        '/usr/lib/aarch64-linux-gnu/*.so*',
    ]
    paths: List[str] = []
    for pat in patterns:
        found = glob.glob(pat)
        random.shuffle(found)
        paths.extend(found[:max_per_dir])
    return [p for p in paths if os.path.isfile(p) and os.path.getsize(p) > 256]


def _collect_pyc(max_files: int = 500) -> List[str]:
    """Python .pyc byte-code files from the current env."""
    import site
    dirs = site.getsitepackages() if hasattr(site, 'getsitepackages') else []
    paths: List[str] = []
    for d in dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith('.pyc'):
                    paths.append(os.path.join(root, f))
    random.shuffle(paths)
    return paths[:max_files]


def download_silesia(dest_dir: str) -> List[str]:
    """Download subset of Silesia corpus. Returns list of downloaded paths."""
    import urllib.request
    os.makedirs(dest_dir, exist_ok=True)
    urls = {
        'dickens': 'https://sun.aei.polsl.pl/~sdeor/corpus/dickens',
        'mozilla': 'https://sun.aei.polsl.pl/~sdeor/corpus/mozilla',
        'mr':      'https://sun.aei.polsl.pl/~sdeor/corpus/mr',
        'nci':     'https://sun.aei.polsl.pl/~sdeor/corpus/nci',
        'ooffice': 'https://sun.aei.polsl.pl/~sdeor/corpus/ooffice',
        'osdb':    'https://sun.aei.polsl.pl/~sdeor/corpus/osdb',
        'reymont': 'https://sun.aei.polsl.pl/~sdeor/corpus/reymont',
        'samba':   'https://sun.aei.polsl.pl/~sdeor/corpus/samba',
        'sao':     'https://sun.aei.polsl.pl/~sdeor/corpus/sao',
        'xml':     'https://sun.aei.polsl.pl/~sdeor/corpus/xml',
    }
    paths: List[str] = []
    for name, url in urls.items():
        dest = os.path.join(dest_dir, name)
        if not os.path.exists(dest):
            try:
                print(f'  Downloading silesia/{name} …', end=' ', flush=True)
                urllib.request.urlretrieve(url, dest)
                print('OK')
            except Exception as e:
                print(f'SKIP ({e})')
                continue
        if os.path.exists(dest):
            paths.append(dest)
    return paths


def _stream_hf_text(dataset_name: str, split: str,
                    text_field: str, max_bytes: int,
                    config_name: str = None) -> bytes:
    """Stream text bytes from a HuggingFace dataset (no full download)."""
    try:
        from datasets import load_dataset
        kwargs = dict(split=split, streaming=True)
        if config_name:
            ds = load_dataset(dataset_name, config_name, **kwargs)
        else:
            ds = load_dataset(dataset_name, **kwargs)
        buf = bytearray()
        for sample in ds:
            text = sample.get(text_field, '') or ''
            buf.extend(text.encode('utf-8', errors='replace'))
            if len(buf) >= max_bytes:
                break
        return bytes(buf[:max_bytes])
    except Exception as e:
        print(f'  HF dataset {dataset_name} unavailable: {e}')
        return b''


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ChunkDataset(Dataset):
    """
    Yields (context_bytes, target_bytes, format_id) tuples from a list of
    raw byte buffers.

    Each sample is a random `seq_len`-byte window drawn from one of the
    source buffers.  Buffers are weighted by their byte count so that larger
    files contribute proportionally more samples.
    """

    def __init__(
        self,
        buffers:  List[bytes],
        seq_len:  int   = 2048,
        n_samples: int  = 50_000,
        seed:     int   = 42,
    ):
        self.seq_len   = seq_len
        self.n_samples = n_samples
        rng = random.Random(seed)

        # Filter empty / too-short buffers
        valid = [(b, sniff(b[:64])) for b in buffers if len(b) > seq_len + 1]
        if not valid:
            raise ValueError('No valid buffers (all shorter than seq_len).')

        # Weight by buffer size
        sizes   = [len(b) for b, _ in valid]
        total   = sum(sizes)
        weights = [s / total for s in sizes]

        # Pre-sample (buffer_idx, start_pos) pairs for __getitem__
        buf_indices = rng.choices(range(len(valid)), weights=weights, k=n_samples)
        self.samples: List[Tuple[bytes, int, int]] = []
        for bi in buf_indices:
            buf, fmt = valid[bi]
            start = rng.randint(0, len(buf) - seq_len - 1)
            self.samples.append((buf, start, fmt))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        buf, start, fmt = self.samples[idx]
        chunk = buf[start: start + self.seq_len + 1]
        x = torch.tensor(list(chunk[:-1]), dtype=torch.long)
        y = torch.tensor(list(chunk[1:]),  dtype=torch.long)
        return x, y, torch.tensor(fmt, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Corpus builders
# ─────────────────────────────────────────────────────────────────────────────

class CorpusBuilder:
    """Assembles training buffers for each curriculum phase."""

    def __init__(self, data_dir: str, max_bytes_per_source: int = 20_000_000):
        self.data_dir = data_dir
        self.max_b    = max_bytes_per_source
        os.makedirs(data_dir, exist_ok=True)

    # ── Phase 1: text ────────────────────────────────────────────────────────

    def build_text_buffers(self) -> List[bytes]:
        print('[Corpus] Building text corpus …')
        buffers: List[bytes] = []

        # wikitext-103-raw-v1  (config name required)
        wiki_path = os.path.join(self.data_dir, 'wikitext103.bin')
        if os.path.exists(wiki_path):
            buffers.append(_load_bytes(wiki_path, self.max_b))
            print(f'  wikitext: loaded from cache')
        else:
            data = _stream_hf_text('wikitext', 'train', 'text', self.max_b,
                                   config_name='wikitext-103-raw-v1')
            if data:
                with open(wiki_path, 'wb') as f:
                    f.write(data)
                buffers.append(data)
                print(f'  wikitext: {len(data)/1e6:.1f} MB')

        # Source code — use HuggingFaceFW/fineweb-edu (no loading script)
        code_path = os.path.join(self.data_dir, 'code.bin')
        if os.path.exists(code_path):
            buffers.append(_load_bytes(code_path, self.max_b))
            print(f'  code: loaded from cache')
        else:
            # Try several datasets in order of preference
            for ds_name, cfg, field in [
                ('roneneldan/TinyStories',    None,     'text'),
                ('HuggingFaceFW/fineweb-edu', 'sample-10BT', 'text'),
                ('ag_news',                   None,     'text'),
            ]:
                data = _stream_hf_text(ds_name, 'train', field,
                                       self.max_b, config_name=cfg)
                if data:
                    with open(code_path, 'wb') as f:
                        f.write(data)
                    buffers.append(data)
                    print(f'  code/text alt ({ds_name}): {len(data)/1e6:.1f} MB')
                    break

        # Fallback synthetic text
        if not buffers:
            print('  Using synthetic text fallback.')
            buffers.append(
                (b'The quick brown fox jumps over the lazy dog. ' * 200_000)
                [:self.max_b]
            )

        print(f'  Text corpus: {sum(len(b) for b in buffers)/1e6:.1f} MB '
              f'across {len(buffers)} source(s)')
        return buffers

    # ── Phase 2: binary ──────────────────────────────────────────────────────

    def build_binary_buffers(self) -> List[bytes]:
        print('[Corpus] Building binary corpus …')
        buffers: List[bytes] = []

        # System ELF binaries
        elf_paths = _collect_system_binaries(max_per_dir=200)
        elf_buf = bytearray()
        for p in elf_paths:
            try:
                elf_buf.extend(_load_bytes(p, 500_000))
                if len(elf_buf) > self.max_b:
                    break
            except OSError:
                pass
        if elf_buf:
            buffers.append(bytes(elf_buf[:self.max_b]))
            print(f'  ELF binaries: {len(elf_buf)/1e6:.1f} MB')

        # Python .pyc files
        pyc_paths = _collect_pyc(max_files=300)
        pyc_buf = bytearray()
        for p in pyc_paths:
            try:
                pyc_buf.extend(_load_bytes(p, 200_000))
                if len(pyc_buf) > self.max_b:
                    break
            except OSError:
                pass
        if pyc_buf:
            buffers.append(bytes(pyc_buf[:self.max_b]))
            print(f'  Python .pyc: {len(pyc_buf)/1e6:.1f} MB')

        # Silesia corpus
        sil_dir   = os.path.join(self.data_dir, 'silesia')
        sil_paths = download_silesia(sil_dir)
        sil_buf   = bytearray()
        for p in sil_paths:
            try:
                sil_buf.extend(_load_bytes(p, 5_000_000))
                if len(sil_buf) > self.max_b:
                    break
            except OSError:
                pass
        if sil_buf:
            buffers.append(bytes(sil_buf[:self.max_b]))
            print(f'  Silesia: {len(sil_buf)/1e6:.1f} MB')

        if not buffers:
            print('  No binary sources found — using random bytes fallback.')
            buffers.append(bytes(np.random.randint(0, 256,
                                                   size=1_000_000,
                                                   dtype=np.uint8).tobytes()))
        print(f'  Binary corpus: {sum(len(b) for b in buffers)/1e6:.1f} MB')
        return buffers


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(
    buffers:    List[bytes],
    seq_len:    int   = 2048,
    n_samples:  int   = 50_000,
    batch_size: int   = 8,
    val_frac:   float = 0.05,
    seed:       int   = 42,
    num_workers: int  = 0,
) -> Tuple[DataLoader, DataLoader]:

    ds    = ChunkDataset(buffers, seq_len=seq_len, n_samples=n_samples, seed=seed)
    n_val = max(1, int(len(ds) * val_frac))
    n_trn = len(ds) - n_val
    trn, val = random_split(ds, [n_trn, n_val],
                            generator=torch.Generator().manual_seed(seed))

    # persistent_workers=True can deadlock in Kaggle notebook kernels;
    # num_workers=0 avoids subprocess forking issues with CUDA.
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True, persistent_workers=False)
    return (DataLoader(trn, shuffle=True,  **kw),
            DataLoader(val, shuffle=False, **kw))
