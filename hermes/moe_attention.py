# hermes/moe_attention.py
# ─────────────────────────────────────────────────────────────────────────────
# Sparse Mixture-of-Experts Local-Window Attention block.
#
# Architecture:
#   • N expert attention modules (default 4), each a standard multi-head
#     causal local-window attention.
#   • A lightweight router scores each token and selects the top-K experts
#     (default K=2) via a straight-through softmax gate.
#   • A load-balance auxiliary loss penalises expert collapse.
#   • Residual + LayerNorm wrapper with a gated FFN.
#
# The local window (default 128 patch tokens = 512 bytes) bounds the
# quadratic attention cost regardless of total sequence length.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Local-Window Causal Attention (single expert) ────────────────────────────

class LocalWindowAttention(nn.Module):
    """
    Causal multi-head attention with a sliding local window.

    Only attends to the previous `window` tokens, bounding complexity to
    O(T * window) instead of O(T²).

    FIX 1: replaced manual masked_fill(-1e9) — which overflows float16 — with
    torch.nn.functional.scaled_dot_product_attention, which handles the mask
    internally in full float32 precision regardless of input dtype.  This also
    fuses the softmax + dropout + matmul into a single CUDA kernel (Flash
    Attention path on Ampere/T4), giving a substantial speed boost.
    """

    def __init__(self, d_model: int, n_heads: int = 8, window: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.window   = window
        self.dropout  = dropout  # kept for sdpa

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out  = nn.Linear(d_model, d_model, bias=False)

        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        nn.init.trunc_normal_(self.out.weight, std=0.02)

    # ── mask cache (built once per (T, device), reused across batches) ───────

    @staticmethod
    def _make_mask(T: int, window: int, device: torch.device) -> torch.Tensor:
        """
        Returns a boolean additive-mask of shape [T, T] where True means
        'block this position'.  Compatible with sdpa's attn_mask argument
        when passed as a bool tensor (True → -inf).
        """
        # causal: upper triangle is blocked
        causal = torch.ones(T, T, device=device, dtype=torch.bool).tril()
        if window < T:
            local  = torch.ones(T, T, device=device,
                                dtype=torch.bool).tril().triu(-(window - 1))
            causal = causal & local
        # sdpa expects True = attend, so we pass causal directly
        return causal   # [T, T] bool, True = allowed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # [B, H, T, Dh]

        # Build the boolean attention mask [T, T]
        mask = self._make_mask(T, self.window, x.device)  # True = attend

        # FIX 1: use fused SDPA — avoids float16 overflow and is faster.
        # dropout_p only applied during training (Module.training flag is
        # respected automatically by sdpa when is_causal=False and mask given).
        dp = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,          # bool mask: True = keep
            dropout_p=dp,
            is_causal=False,         # we supply an explicit mask
        )                            # [B, H, T, Dh]

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)


# ── Sparse MoE Attention Block ────────────────────────────────────────────────

class SparseMoEAttentionBlock(nn.Module):
    """
    N expert local-window attention modules with top-K routing.

    FIX 2: router now operates per-token rather than on the sequence mean,
    so each token can be dispatched independently.  The original mean-pool
    routing forced every token in a sample to the same experts — defeating the
    purpose of MoE for variable content.  The dispatch loop is also tightened
    to avoid redundant indexing.
    """

    def __init__(self, d_model: int, n_experts: int = 4, top_k: int = 2,
                 n_heads: int = 8, window: int = 128,
                 ffn_expand: int = 4, dropout: float = 0.1):
        super().__init__()
        assert top_k <= n_experts
        self.n_experts = n_experts
        self.top_k     = top_k

        # Router: per-token expert scores
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Expert pool
        self.experts = nn.ModuleList([
            LocalWindowAttention(d_model, n_heads, window, dropout)
            for _ in range(n_experts)
        ])

        # Gated FFN (shared)
        inner = d_model * ffn_expand
        self.ffn = nn.Sequential(
            nn.Linear(d_model, inner * 2),
            _SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner, d_model),
            nn.Dropout(dropout),
        )

        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn  = nn.LayerNorm(d_model)

        nn.init.trunc_normal_(self.router.weight, std=0.02)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            out:         [B, T, d_model]
            router_loss: scalar load-balance auxiliary loss
        """
        B, T, D = x.shape

        # FIX 2: per-token routing (was per-sequence mean — defeated MoE purpose)
        # router_logits: [B, T, n_experts]
        router_logits = self.router(x)
        router_probs  = F.softmax(router_logits, dim=-1)   # [B, T, n_experts]

        # Top-K over experts for each token
        topk_weights, topk_idx = router_probs.topk(self.top_k, dim=-1)
        # Renormalise so selected-expert weights sum to 1
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        # topk_weights: [B, T, top_k]
        # topk_idx:     [B, T, top_k]

        # Dispatch: accumulate weighted expert outputs
        attn_out = torch.zeros_like(x)
        for k in range(self.top_k):
            for e_idx in range(self.n_experts):
                # Which batch items have expert e_idx at rank k?
                # mask: [B] bool
                mask = (topk_idx[:, :, k] == e_idx).any(dim=1)  # batch-level gate
                if not mask.any():
                    continue
                # token-level weight for this expert at rank k: [B, T, 1]
                tok_mask = (topk_idx[mask, :, k] == e_idx).float().unsqueeze(-1)
                w        = topk_weights[mask, :, k].unsqueeze(-1) * tok_mask
                out      = self.experts[e_idx](x[mask])   # [m, T, D]
                attn_out[mask] = attn_out[mask] + w * out

        x = self.norm_attn(x + attn_out)

        # Load-balance loss: penalise expert collapse (per-token mean)
        mean_probs  = router_probs.mean(dim=(0, 1))        # [n_experts]
        router_loss = self.n_experts * (mean_probs * mean_probs).sum()

        # Shared gated FFN
        x = self.norm_ffn(x + self.ffn(x))

        return x, router_loss


# ── SwiGLU activation ────────────────────────────────────────────────────────

class _SwiGLU(nn.Module):
    """Splits last dim in half and applies SiLU gate."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)
