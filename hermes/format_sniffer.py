# hermes/format_sniffer.py
# ─────────────────────────────────────────────────────────────────────────────
# Maps the first 16 bytes of a file to one of 32 format-class IDs.
# The ID is used as a learnable "format token" prepended to every sequence,
# letting HERMES learn format-specific byte distributions without an explicit
# format parser at inference time.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import List, Tuple

# (magic_bytes_prefix, format_id, human_label)
_RULES: List[Tuple[bytes, int, str]] = [
    # Executables / code
    (b'\x7fELF',           0,  'elf'),
    (b'MZ',                1,  'pe_exe'),
    (b'\xca\xfe\xba\xbe',  2,  'macho'),
    (b'\xfe\xed\xfa\xce',  2,  'macho'),
    (b'\xfe\xed\xfa\xcf',  2,  'macho'),
    (b'\x00asm',           3,  'wasm'),
    # Archives / containers
    (b'PK\x03\x04',        4,  'zip'),
    (b'PK\x05\x06',        4,  'zip'),
    (b'\x1f\x8b',          5,  'gzip'),
    (b'BZh',               6,  'bzip2'),
    (b'\xfd7zXZ\x00',      7,  'xz'),
    (b'7z\xbc\xaf\x27\x1c',8,  '7zip'),
    (b'Rar!',              9,  'rar'),
    (b'\x04\x22\x4d\x18', 10,  'lz4'),
    (b'\x28\xb5\x2f\xfd', 11,  'zstd'),
    # Images
    (b'\x89PNG\r\n\x1a\n', 12, 'png'),
    (b'\xff\xd8\xff',      13, 'jpeg'),
    (b'GIF8',              14, 'gif'),
    (b'BM',                15, 'bmp'),
    (b'RIFF',              16, 'riff'),   # WAV / AVI / WEBP
    (b'\x00\x00\x01\x00',  17, 'ico'),
    # Audio / Video
    (b'fLaC',              18, 'flac'),
    (b'OggS',              19, 'ogg'),
    (b'ID3',               20, 'mp3'),
    # Documents / structured
    (b'%PDF',              21, 'pdf'),
    (b'\xd0\xcf\x11\xe0',  22, 'ole2'),   # DOC, XLS, PPT (old Office)
    (b'PK\x03\x04',        23, 'ooxml'),  # DOCX, XLSX (also ZIP — covered above)
    (b'SQLite format 3',   24, 'sqlite'),
    # Data formats
    (b'PAR1',              25, 'parquet'),
    (b'ORC',               26, 'orc'),
    (b'\x4f\x62\x6a\x01',  27, 'avro'),
    # Source / text heuristics handled below
    # ID 28 = UTF-8 text (BOM or high ASCII ratio)
    # ID 29 = ASCII source code
    # ID 30 = binary (unrecognised, high entropy)
    # ID 31 = unknown / short file
]

NUM_FORMAT_CLASSES = 32
_UNKNOWN_ID        = 31
_BINARY_ID         = 30
_SOURCE_ID         = 29
_TEXT_ID           = 28

# Precompute max prefix length needed
_MAX_PREFIX = max(len(magic) for magic, _, _ in _RULES)


def sniff(data: bytes) -> int:
    """Return a format class ID in [0, 31] for the given raw bytes."""
    if len(data) < 4:
        return _UNKNOWN_ID

    prefix = data[:_MAX_PREFIX]

    # Check magic rules (first match wins)
    for magic, fid, _ in _RULES:
        if prefix[:len(magic)] == magic:
            return fid

    # Heuristic fallback: inspect byte distribution of first 512 bytes
    sample = data[:512]
    n = len(sample)
    printable = sum(0x20 <= b < 0x7f or b in (9, 10, 13) for b in sample)
    high       = sum(b > 0x7f for b in sample)

    if printable / n > 0.90:
        return _SOURCE_ID   # ASCII source / text
    if high / n > 0.30:
        return _TEXT_ID     # likely UTF-8 prose
    return _BINARY_ID       # unrecognised binary


def sniff_path(path: str) -> int:
    """Sniff format from a file path (reads only the first 64 bytes)."""
    try:
        with open(path, 'rb') as f:
            header = f.read(64)
        return sniff(header)
    except OSError:
        return _UNKNOWN_ID


def format_label(fid: int) -> str:
    """Human-readable label for a format ID."""
    labels = {fid: lbl for _, fid, lbl in _RULES}
    labels.update({_TEXT_ID: 'text', _SOURCE_ID: 'source',
                   _BINARY_ID: 'binary', _UNKNOWN_ID: 'unknown'})
    return labels.get(fid, f'fmt{fid}')
