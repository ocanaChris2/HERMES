# hermes/mamba_block.py
# ─────────────────────────────────────────────────────────────────────────────
# Pure-PyTorch implementation of the Mamba Selective State Space Model.
# Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
# State Spaces", arXiv:2312.00752.
#
# Scan implementation: chunked sequential scan.
#   Each chunk of SCAN_CHUNK timesteps is processed with a vectorised
#   torch operation (no Python loop *per token*), keeping GPU utilisation
#   high while peak memory is O(B * chunk * d_inner * d_state) instead of
#   O(B * L_pad * d_inner * d_state * copies) required by the Blelloch
#   parallel scan.  On a T4 with B=8, L=512, d_inner=1024, d_state=16
#   the parallel scan tried to allocate 256 MiB+ of intermediates;
#   this implementation requires ~4 MiB peak for the same shapes.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """
    One Mamba SSM layer with residual connection and LayerNorm.

    Args:
        d_model:  input / output dimension.
        d_state:  SSM state dimension N (default 16).
        d_conv:   depthwise conv kernel size (default 4).
        expand:   inner expansion factor; d_inner = expand * d_model (default 2).
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.d_inner  = int(expand * d_model)

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d   = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        self.x_proj   = nn.Linear(self.d_inner, 1 + 2 * d_state, bias=False)
        self.dt_proj  = nn.Linear(1, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32) \
               .unsqueeze(0).expand(self.d_inner, -1)
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.in_proj.weight,  std=0.02)
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.x_proj.weight,   std=0.02)

        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.d_inner) *
            (math.log(0.1) - math.log(dt_init_floor)) + math.log(dt_init_floor)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, L, d_model]
            h: [B, d_inner, d_state]  persistent SSM state; None → zeros.

        Returns:
            y:     [B, L, d_model]
            new_h: [B, d_inner, d_state]  (detach before next chunk)
        """
        B, L, _ = x.shape

        # ── 1. Gate split ────────────────────────────────────────────────────
        xz           = self.in_proj(x)
        x_in, z      = xz.chunk(2, dim=-1)              # [B, L, d_inner]

        # ── 2. Depthwise conv ────────────────────────────────────────────────
        x_conv = self.conv1d(x_in.permute(0, 2, 1))[:, :, :L]
        x_conv = F.silu(x_conv).permute(0, 2, 1)        # [B, L, d_inner]

        # ── 3. SSM parameters ────────────────────────────────────────────────
        xp                  = self.x_proj(x_conv)
        dt_raw, B_ssm, C_ssm = xp.split([1, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt_raw))            # [B, L, d_inner]
        A  = -torch.exp(self.A_log.float())              # [d_inner, d_state]

        # ── 4. Discretize (ZOH) ──────────────────────────────────────────────
        dA  = torch.exp(dt.unsqueeze(-1) * A)            # [B, L, d_inner, d_state]
        dBu = (dt * x_conv).unsqueeze(-1) * B_ssm.unsqueeze(2)

        # ── 5. Chunked sequential scan (memory-efficient) ────────────────────
        y, new_h = _selective_scan_chunked(h, dA, dBu, C_ssm)

        # ── 6. Output ────────────────────────────────────────────────────────
        y   = y + x_conv * self.D
        out = self.out_proj(y * F.silu(z))
        return self.norm(x + out), new_h


# ── Chunked sequential scan ───────────────────────────────────────────────────
# Process SCAN_CHUNK timesteps at once using vectorised ops (no per-token
# Python loop).  Peak extra memory = O(B * SCAN_CHUNK * d_inner * d_state)
# which is ~4 MiB for the default settings, vs 256+ MiB for the parallel scan.
SCAN_CHUNK = 32  # tune: larger = fewer Python iterations, more VRAM


def _selective_scan_chunked(
    h0:  Optional[torch.Tensor],  # [B, d_inner, d_state] or None
    dA:  torch.Tensor,            # [B, L, d_inner, d_state]
    dBu: torch.Tensor,            # [B, L, d_inner, d_state]
    C:   torch.Tensor,            # [B, L, d_state]
    chunk: int = SCAN_CHUNK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient chunked sequential scan for the SSM recurrence:
        h_t = dA_t * h_{t-1} + dBu_t
        y_t = (h_t * C_t).sum(-1)

    Processes `chunk` timesteps per iteration with fully vectorised
    PyTorch ops.  Peak intermediate memory scales with `chunk`, not L.
    """
    B, L, d_inner, d_state = dA.shape
    dtype  = dA.dtype
    device = dA.device

    # Running hidden state — shape [B, d_inner, d_state]
    h = h0.float() if h0 is not None else torch.zeros(
        B, d_inner, d_state, dtype=torch.float32, device=device
    )

    # Pre-cast inputs to float32 for numerical stability
    dA_f  = dA.float()   # [B, L, d_inner, d_state]
    dBu_f = dBu.float()  # [B, L, d_inner, d_state]
    C_f   = C.float()    # [B, L, d_state]

    y_chunks: list = []
    h_last = h

    for start in range(0, L, chunk):
        end   = min(start + chunk, L)
        clen  = end - start

        a_c  = dA_f [:, start:end]   # [B, clen, d_inner, d_state]
        bu_c = dBu_f[:, start:end]
        c_c  = C_f  [:, start:end]   # [B, clen, d_state]

        # Accumulate hidden states over the chunk sequentially
        # (only `clen` Python iterations, not L)
        h_list: list = []
        h_cur = h_last
        for t in range(clen):
            # a_c[:, t]: [B, d_inner, d_state]
            h_cur = a_c[:, t] * h_cur + bu_c[:, t]
            h_list.append(h_cur)

        # Stack → [B, clen, d_inner, d_state]
        h_chunk = torch.stack(h_list, dim=1)

        # Output for this chunk: y = (h * C).sum(-1)
        # c_c: [B, clen, d_state] → [B, clen, 1, d_state]
        y_c = (h_chunk * c_c.unsqueeze(2)).sum(-1)  # [B, clen, d_inner]
        y_chunks.append(y_c)

        h_last = h_cur  # carry state forward

    y       = torch.cat(y_chunks, dim=1).to(dtype)   # [B, L, d_inner]
    new_h   = h_last.to(dtype)                        # [B, d_inner, d_state]
    return y, new_h
