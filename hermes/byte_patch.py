# hermes/byte_patch.py
# ─────────────────────────────────────────────────────────────────────────────
# Two-level byte ↔ patch tokenizer.
#
# Encoder: raw bytes [B, T]  →  patch tokens [B, T/P, d_model]
#   - Byte embedding (d_byte)
#   - Strided Conv1d (stride=patch_size) collapses P bytes into one patch token
#   - GroupNorm + GELU
#
# Decoder: patch tokens [B, T/P, d_model]  →  per-byte logits [B, T, 256]
#   - ConvTranspose1d expands back to byte resolution
#   - Per-position 2-layer MLP produces logits over 256 byte values
#
# The stride-4 patch reduces sequence length by 4× for the expensive SSM /
# attention layers, allowing seq_len=4096 bytes with only 1024 patch tokens.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class BytePatchEncoder(nn.Module):
    """
    [B, T] byte indices  →  [B, T // patch_size, d_model] patch tokens.

    Args:
        patch_size: number of bytes collapsed into one patch token (default 4).
        d_byte:     byte embedding dimension before the strided convolution.
        d_model:    output patch token dimension.
        vocab:      byte vocabulary size (256).
    """

    def __init__(self, patch_size: int = 4, d_byte: int = 64,
                 d_model: int = 512, vocab: int = 256):
        super().__init__()
        self.patch_size = patch_size

        self.byte_emb = nn.Embedding(vocab, d_byte)

        # Strided conv: collapses patch_size bytes into one token
        self.patch_conv = nn.Sequential(
            nn.Conv1d(d_byte, d_model,
                      kernel_size=patch_size, stride=patch_size, bias=False),
            nn.GroupNorm(min(8, d_model // 8), d_model),
            nn.GELU(),
        )

        # Residual projection if d_byte != d_model (always true here)
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.byte_emb.weight, std=0.02)
        for m in self.patch_conv.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T]  →  [B, T/patch_size, d_model]"""
        B, T = x.shape
        # Pad T to be divisible by patch_size
        pad = (self.patch_size - T % self.patch_size) % self.patch_size
        if pad:
            x = F.pad(x, (0, pad))

        e = self.byte_emb(x)                          # [B, T', d_byte]
        e = e.permute(0, 2, 1)                        # [B, d_byte, T']
        patches = self.patch_conv(e).permute(0, 2, 1) # [B, T'/P, d_model]
        return self.norm(patches)


class PatchByteDecoder(nn.Module):
    """
    [B, T/patch_size, d_model] patch tokens  →  [B, T, 256] per-byte logits.

    Uses transposed convolution for upsampling then a small per-position MLP
    to produce logits for each of the 256 byte values.
    """

    def __init__(self, patch_size: int = 4, d_model: int = 512,
                 d_mid: int = 256, vocab: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.vocab = vocab

        # Upsample: [B, d_model, T/P]  →  [B, d_mid, T]
        self.up_conv = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_mid,
                               kernel_size=patch_size, stride=patch_size,
                               bias=False),
            nn.GroupNorm(min(8, d_mid // 8), d_mid),
            nn.GELU(),
        )

        # Per-byte MLP: d_mid → vocab logits
        self.byte_head = nn.Sequential(
            nn.Linear(d_mid, d_mid),
            nn.GELU(),
            nn.Linear(d_mid, vocab),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.ConvTranspose1d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, patches: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        patches:    [B, S, d_model]  (S = T // patch_size)
        target_len: original T (to trim padding)
        Returns:    [B, T, 256] logits
        """
        h = patches.permute(0, 2, 1)                    # [B, d_model, S]
        h = self.up_conv(h).permute(0, 2, 1)            # [B, S*P, d_mid]
        h = h[:, :target_len, :]                        # trim padding
        logits = self.byte_head(h)                       # [B, T, 256]
        return logits
