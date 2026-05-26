# coding/coder.py
# ─────────────────────────────────────────────────────────────────────────────
# HERMES Streaming Entropy Coder
#
# Uses the `constriction` library (Rust-backed rANS, pip-installable):
#   pip install constriction
#
# Stream format  "HRM\x01":
#   [4]  magic:         b'HRM\x01'
#   [1]  format_id:     uint8
#   [8]  original_size: uint64 LE
#   [32] sha256:        bytes
#   [4]  n_chunks:      uint32 LE
#   [n_chunks × 4] chunk_sizes: uint32 LE each
#   [...] ANS-coded chunk payloads (concatenated)
#
# C++/Rust side must implement the same rANS codec.
# The constriction Rust crate can be used natively on the Rust side:
#   constriction = "0.4"   (in Cargo.toml)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import hashlib
import math
import struct
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

STREAM_MAGIC = b'HRM\x01'
PROB_BITS    = 14                   # 16384-entry CDF (good precision, fast)
PROB_SCALE   = 1 << PROB_BITS


# ─────────────────────────────────────────────────────────────────────────────
# CDF quantisation (deterministic — encoder == decoder)
# ─────────────────────────────────────────────────────────────────────────────

def logits_to_cdf(logits_row: np.ndarray) -> np.ndarray:
    """
    [256] float32 logits  →  [257] int64 CDF with counts summing to PROB_SCALE.

    Uses a single-shot rounding correction to avoid the iterative while-loop.
    Identical results on encoder and decoder sides given the same logits.
    """
    p = logits_row.astype(np.float64)
    # stable softmax
    p -= p.max()
    p  = np.exp(p)
    p /= p.sum()

    counts = np.floor(p * PROB_SCALE).astype(np.int64)
    counts = np.clip(counts, 1, None)            # every symbol ≥ 1 count

    deficit = PROB_SCALE - counts.sum()
    if deficit > 0:
        # Give remaining mass to highest-probability symbols (greedy)
        order = np.argsort(p - counts / PROB_SCALE)[::-1]
        for i in order[:deficit]:
            counts[i] += 1
    elif deficit < 0:
        # Remove excess from highest-count symbols
        order = np.argsort(counts)[::-1]
        for i in order[: -deficit]:
            counts[i] -= 1
            if counts[i] < 1:
                counts[i] = 1

    cdf = np.zeros(257, dtype=np.int64)
    cdf[1:] = np.cumsum(counts)
    return cdf


def batch_logits_to_cdfs(logits: np.ndarray) -> np.ndarray:
    """
    [T, 256] float32 logits  →  [T, 257] int64 CDFs.
    Vectorised softmax + floor rounding for speed.
    """
    p = logits.astype(np.float64)
    p -= p.max(axis=1, keepdims=True)
    p  = np.exp(p)
    p /= p.sum(axis=1, keepdims=True)

    counts = np.floor(p * PROB_SCALE).astype(np.int64).clip(1)

    # Fix rounding errors row-by-row
    deficit = PROB_SCALE - counts.sum(axis=1)             # [T]
    for i, d in enumerate(deficit):
        if d == 0:
            continue
        residual = p[i] - counts[i] / PROB_SCALE
        idx = np.argsort(residual)[::-1] if d > 0 else np.argsort(counts[i])[::-1]
        nd  = abs(int(d))
        for j in idx[:nd]:
            counts[i, j] += int(np.sign(d))
            if counts[i, j] < 1:
                counts[i, j] = 1

    cdfs = np.zeros((len(logits), 257), dtype=np.int64)
    cdfs[:, 1:] = np.cumsum(counts, axis=1)
    return cdfs


# ─────────────────────────────────────────────────────────────────────────────
# constriction-based ANS encoder/decoder
# ─────────────────────────────────────────────────────────────────────────────

def _encode_chunk(symbols: np.ndarray, cdfs: np.ndarray) -> bytes:
    """
    symbols: [T] uint8
    cdfs:    [T, 257] int64
    Returns raw bytes of the ANS-coded chunk.
    """
    import constriction

    coder = constriction.stream.stack.AnsCoder()
    T = len(symbols)
    for t in range(T - 1, -1, -1):        # ANS encodes in reverse
        sym = int(symbols[t])
        cdf = cdfs[t].tolist()
        model = constriction.stream.model.CustomModel(
            cdf[:-1],  # quantiles (lower CDF values for each symbol)
            cdf[1:],   # next quantiles (upper CDF values)
            PROB_BITS,
        )
        coder.encode_symbol(sym, model)
    return coder.get_compressed().tobytes()


def _decode_chunk(payload: bytes, cdfs: np.ndarray,
                  chunk_len: int) -> np.ndarray:
    """
    payload:    raw ANS bytes
    cdfs:       [T, 257] int64  (same as encoder)
    chunk_len:  number of bytes to decode
    Returns [T] uint8 decoded bytes.
    """
    import constriction

    compressed = np.frombuffer(payload, dtype=np.uint32)
    coder = constriction.stream.stack.AnsCoder(compressed)

    symbols = np.empty(chunk_len, dtype=np.uint8)
    for t in range(chunk_len):
        cdf = cdfs[t].tolist()
        model = constriction.stream.model.CustomModel(
            cdf[:-1], cdf[1:], PROB_BITS,
        )
        symbols[t] = coder.decode_symbol(model)
    return symbols


# ─────────────────────────────────────────────────────────────────────────────
# Fallback pure-Python rANS (no constriction dependency)
# ─────────────────────────────────────────────────────────────────────────────

_STATE_MIN = 1 << 16

class _PureANSEncoder:
    def __init__(self):
        self.state  = _STATE_MIN
        self.stream: List[int] = []

    def encode(self, symbol: int, cdf: np.ndarray):
        freq  = int(cdf[symbol + 1] - cdf[symbol])
        start = int(cdf[symbol])
        max_s = ((_STATE_MIN // PROB_SCALE) * freq) << 16
        while self.state >= max_s:
            self.stream.append(self.state & 0xFFFF)
            self.state >>= 16
        self.state = (self.state // freq) * PROB_SCALE + start + (self.state % freq)

    def flush(self) -> bytes:
        self.stream.append(self.state & 0xFFFF)
        self.stream.append((self.state >> 16) & 0xFFFF)
        return struct.pack(f'<{len(self.stream)}H', *self.stream)


class _PureANSDecoder:
    def __init__(self, data: bytes):
        words = list(struct.unpack(f'<{len(data)//2}H', data))
        self.state  = (words.pop() << 16) | words.pop()
        self.stream = words

    def decode(self, cdf: np.ndarray) -> int:
        slot = self.state & (PROB_SCALE - 1)
        symbol = int(np.searchsorted(cdf[1:], slot, side='right'))
        symbol = min(symbol, 255)
        freq  = int(cdf[symbol + 1] - cdf[symbol])
        start = int(cdf[symbol])
        self.state = freq * (self.state >> PROB_BITS) + (self.state & (PROB_SCALE - 1)) - start
        while self.state < _STATE_MIN:
            word = self.stream.pop() if self.stream else 0
            self.state = (self.state << 16) | word
        return symbol


# ─────────────────────────────────────────────────────────────────────────────
# High-level compress / decompress
# ─────────────────────────────────────────────────────────────────────────────

def _try_import_constriction() -> bool:
    try:
        import constriction  # noqa: F401
        return True
    except ImportError:
        return False


@torch.no_grad()
def compress(
    raw:        bytes,
    model:      torch.nn.Module,
    device:     torch.device,
    chunk_size: int  = 4096,
    verbose:    bool = True,
) -> bytes:
    """
    Compress `raw` bytes using HERMES model.

    The SSM hidden state (h_list) is carried across chunks so the model has
    full file-level context when predicting later bytes.
    """
    from hermes.format_sniffer import sniff

    model.eval()
    use_constriction = _try_import_constriction()

    n        = len(raw)
    fmt_id   = sniff(raw[:64])
    sha256   = hashlib.sha256(raw).digest()
    raw_arr  = np.frombuffer(raw, dtype=np.uint8)

    chunk_payloads: List[bytes] = []
    h_list = None

    for start in range(0, n, chunk_size):
        chunk      = raw_arr[start: start + chunk_size]
        chunk_len  = len(chunk)

        # Build model input (pad to chunk_size for batched inference)
        pad_len    = chunk_size - chunk_len
        x_np       = np.pad(chunk, (0, pad_len))
        x          = torch.tensor(x_np, dtype=torch.long).unsqueeze(0).to(device)
        fmt_t      = torch.tensor([fmt_id], dtype=torch.long).to(device)

        logits, h_list, _, _ = model(x, fmt_t, h_list=h_list,
                                     targets=None, training=False)

        # Detach SSM state before next chunk
        h_list = [h.detach() for h in h_list]

        # [1, chunk_size, 256] → [chunk_len, 256] numpy
        logits_np = logits[0, :chunk_len].float().cpu().numpy()
        cdfs      = batch_logits_to_cdfs(logits_np)

        if use_constriction:
            payload = _encode_chunk(chunk, cdfs)
        else:
            enc = _PureANSEncoder()
            for t in range(chunk_len - 1, -1, -1):
                enc.encode(int(chunk[t]), cdfs[t])
            payload = enc.flush()

        chunk_payloads.append(payload)

    # ── Build stream ──────────────────────────────────────────────────────────
    n_chunks    = len(chunk_payloads)
    chunk_sizes = [len(p) for p in chunk_payloads]
    header      = (STREAM_MAGIC
                   + bytes([fmt_id])
                   + struct.pack('<Q', n)
                   + sha256
                   + struct.pack('<I', n_chunks)
                   + struct.pack(f'<{n_chunks}I', *chunk_sizes))
    out = header + b''.join(chunk_payloads)

    if verbose:
        bpc = len(out) * 8 / max(n, 1)
        print(f'[compress] {n:,} B → {len(out):,} B  ({bpc:.3f} BPC)')
    return out


@torch.no_grad()
def decompress(
    compressed: bytes,
    model:      torch.nn.Module,
    device:     torch.device,
    chunk_size: int  = 4096,
    verbose:    bool = True,
) -> bytes:
    """Decompress a HERMES stream. Mirrors compress() exactly."""
    from hermes.format_sniffer import sniff

    # ── Parse header ──────────────────────────────────────────────────────────
    if compressed[:4] != STREAM_MAGIC:
        raise ValueError(f'Bad magic: expected {STREAM_MAGIC}, got {compressed[:4]}')

    offset   = 4
    fmt_id   = compressed[offset];          offset += 1
    n,       = struct.unpack_from('<Q', compressed, offset); offset += 8
    sha_orig = compressed[offset: offset + 32];              offset += 32
    n_chunks,= struct.unpack_from('<I', compressed, offset); offset += 4
    chunk_sizes = list(struct.unpack_from(f'<{n_chunks}I', compressed, offset))
    offset  += 4 * n_chunks

    use_constriction = _try_import_constriction()
    model.eval()

    fmt_t   = torch.tensor([fmt_id], dtype=torch.long).to(device)
    h_list  = None
    result  = bytearray()
    remaining = n

    for ci in range(n_chunks):
        chunk_len  = min(chunk_size, remaining)
        payload    = compressed[offset: offset + chunk_sizes[ci]]
        offset    += chunk_sizes[ci]

        # Build context from previous decoded bytes (or zeros for first chunk)
        if len(result) == 0:
            prev = np.zeros(chunk_size, dtype=np.uint8)
        else:
            prev_bytes = bytes(result[- chunk_size:])
            prev = np.frombuffer(prev_bytes.ljust(chunk_size, b'\x00'),
                                 dtype=np.uint8)

        x     = torch.tensor(prev, dtype=torch.long).unsqueeze(0).to(device)
        logits, h_list, _, _ = model(x, fmt_t, h_list=h_list,
                                     targets=None, training=False)
        h_list = [h.detach() for h in h_list]

        logits_np = logits[0, :chunk_len].float().cpu().numpy()
        cdfs      = batch_logits_to_cdfs(logits_np)

        if use_constriction:
            decoded = _decode_chunk(payload, cdfs, chunk_len)
        else:
            dec = _PureANSDecoder(payload)
            decoded = np.array([dec.decode(cdfs[t]) for t in range(chunk_len)],
                               dtype=np.uint8)

        result.extend(decoded.tobytes())
        remaining -= chunk_len

    raw = bytes(result)
    if hashlib.sha256(raw).digest() != sha_orig:
        raise ValueError('SHA-256 checksum mismatch — stream is corrupted.')
    if verbose:
        print(f'[decompress] Restored {n:,} bytes ✓')
    return raw
