# hermes/model.py
# ─────────────────────────────────────────────────────────────────────────────
# HERMES — Hierarchical Entropy Routing Model with Efficient State-spaces
#
# Full model assembly:
#
#   raw bytes [B,T]
#       │
#   BytePatchEncoder        → [B, T/P, d_model]   (stride-P conv)
#       │
#   FormatToken             → prepend 1 format embedding, seq = [B, T/P+1, d_model]
#       │
#   MambaBlock × n_mamba   → persistent SSM state h carried across chunks
#       │
#   SparseMoEBlock × n_moe → (+ EarlyExitGate after each block)
#       │
#   PatchByteDecoder        → [B, T, 256] per-byte logits
#
# Stateful inference (compression):
#   The SSM hidden states (h_list) are passed explicitly between chunk calls
#   so that information from earlier parts of the file propagates to later
#   predictions — this is the primary long-range adaptation mechanism
#   (replaces OTTA backprop at inference time).
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .byte_patch     import BytePatchEncoder, PatchByteDecoder
from .mamba_block    import MambaBlock
from .moe_attention  import SparseMoEAttentionBlock
from .early_exit     import EarlyExitGate
from .format_sniffer import NUM_FORMAT_CLASSES


class HERMES(nn.Module):
    """
    Universal adaptive byte compressor.

    Args:
        d_model:    patch token dimension.
        d_byte:     byte embedding dimension (before patch conv).
        d_state:    Mamba SSM state dim.
        patch_size: bytes per patch token (stride of encoder conv).
        n_mamba:    number of Mamba blocks.
        n_moe:      number of Sparse MoE attention blocks.
        n_experts:  experts per MoE block.
        top_k:      active experts per forward pass.
        n_heads:    attention heads per expert.
        window:     local attention window (patch tokens).
        dropout:    dropout rate.
        exit_threshold: early-exit confidence threshold during inference.
    """

    def __init__(
        self,
        d_model:        int   = 512,
        d_byte:         int   = 64,
        d_state:        int   = 16,
        patch_size:     int   = 4,
        n_mamba:        int   = 2,
        n_moe:          int   = 4,
        n_experts:      int   = 4,
        top_k:          int   = 2,
        n_heads:        int   = 8,
        window:         int   = 128,
        dropout:        float = 0.1,
        exit_threshold: float = 0.80,
        vocab:          int   = 256,
    ):
        super().__init__()

        self.patch_size     = patch_size
        self.d_model        = d_model
        self.n_mamba        = n_mamba
        self.n_moe          = n_moe
        self.vocab          = vocab
        self.exit_threshold = exit_threshold

        # ── Tokenizer ──────────────────────────────────────────────────────
        self.encoder = BytePatchEncoder(patch_size, d_byte, d_model, vocab)
        self.decoder = PatchByteDecoder(patch_size, d_model, d_model // 2, vocab)

        # ── Format embedding ───────────────────────────────────────────────
        self.format_emb = nn.Embedding(NUM_FORMAT_CLASSES, d_model)

        # ── Positional embedding (for patch tokens) ────────────────────────
        # Max 8192 patch tokens = 32 KB at patch_size=4
        self.pos_emb = nn.Embedding(8192, d_model)

        # ── Mamba SSM layers ────────────────────────────────────────────────
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, expand=2)
            for _ in range(n_mamba)
        ])

        # ── Sparse MoE + Early Exit layers ─────────────────────────────────
        self.moe_blocks = nn.ModuleList([
            SparseMoEAttentionBlock(d_model, n_experts, top_k,
                                    n_heads, window, dropout=dropout)
            for _ in range(n_moe)
        ])
        self.exit_gates = nn.ModuleList([
            EarlyExitGate(d_model, vocab, exit_threshold)
            for _ in range(n_moe)
        ])

        # ── Final norm ─────────────────────────────────────────────────────
        self.final_norm = nn.LayerNorm(d_model)

        self._init_weights()

    # ── weight init ────────────────────────────────────────────────────────────

    def _init_weights(self):
        nn.init.trunc_normal_(self.format_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_emb.weight,    std=0.02)

    # ── parameter count ────────────────────────────────────────────────────────

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config(self) -> Dict:
        return dict(
            d_model=self.d_model, d_state=self.mamba_layers[0].d_state,
            patch_size=self.patch_size, n_mamba=self.n_mamba, n_moe=self.n_moe,
            n_experts=self.moe_blocks[0].n_experts,
            top_k=self.moe_blocks[0].top_k,
            n_heads=self.moe_blocks[0].experts[0].n_heads,
            window=self.moe_blocks[0].experts[0].window,
            exit_threshold=self.exit_threshold,
            vocab=self.vocab,
        )

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:          torch.Tensor,               # [B, T] byte indices
        format_id:  torch.Tensor,               # [B]    format class ID
        h_list:     Optional[List[torch.Tensor]] = None,   # Mamba states
        targets:    Optional[torch.Tensor]        = None,  # [B, T] for aux loss
        training:   bool                          = True,
    ) -> Tuple[torch.Tensor,               # logits [B, T, 256]
               List[torch.Tensor],         # new h_list (updated Mamba states)
               torch.Tensor,              # total auxiliary loss
               List[torch.Tensor]]:       # exit logits per block (may be empty)
        """
        Args:
            x:         [B, T]  raw byte indices.
            format_id: [B]     format tag from magic-byte sniffer.
            h_list:    list of [B, d_inner, d_state] Mamba states — one per
                       Mamba layer. Pass None to start from zeros (new file).
                       Pass detached tensors from previous chunk for continuity.
            targets:   [B, T] byte targets — needed for auxiliary losses.
            training:  disables early exit when True.

        Returns:
            logits:      [B, T, 256]
            new_h_list:  updated Mamba states (detach before next chunk call)
            aux_loss:    scalar auxiliary loss (router + exit probes)
            exit_logits: intermediate logits from exit gates (for distillation)
        """
        B, T = x.shape

        # ── 1. Encode bytes → patches ───────────────────────────────────────
        patches = self.encoder(x)                           # [B, S, d_model]
        S = patches.shape[1]

        # ── 2. Prepend format token ─────────────────────────────────────────
        fmt_tok = self.format_emb(format_id).unsqueeze(1)  # [B, 1, d_model]
        patches = torch.cat([fmt_tok, patches], dim=1)     # [B, S+1, d_model]
        S1 = patches.shape[1]

        # ── 3. Add positional embedding ─────────────────────────────────────
        pos = torch.arange(S1, device=x.device)
        patches = patches + self.pos_emb(pos).unsqueeze(0)

        # ── 4. Mamba SSM layers (stateful) ──────────────────────────────────
        if h_list is None:
            h_list = [None] * self.n_mamba

        new_h_list: List[torch.Tensor] = []
        for i, mamba in enumerate(self.mamba_layers):
            patches, new_h = mamba(patches, h_list[i])
            new_h_list.append(new_h)

        # ── 5. Sparse MoE blocks + early exit ──────────────────────────────
        aux_loss     = torch.tensor(0.0, device=x.device)
        exit_logits: List[torch.Tensor] = []

        # Targets for exit gates are at patch resolution (subsample)
        # We align patch index i → byte index i*patch_size
        patch_targets: Optional[torch.Tensor] = None
        if targets is not None:
            # Take the first byte of each patch as the patch-level target
            patch_targets = targets[:, ::self.patch_size]
            # Trim to S (patch count without the format token)
            patch_targets = patch_targets[:, :S]
            # Pad by 1 to account for format token prepended at position 0
            patch_targets = F.pad(patch_targets, (1, 0), value=0)

        for i, (moe, gate) in enumerate(zip(self.moe_blocks, self.exit_gates)):
            patches, router_loss = moe(patches)
            aux_loss = aux_loss + 0.01 * router_loss

            g_logits, conf, exit_aux = gate(patches, patch_targets)
            exit_logits.append(g_logits)
            if targets is not None:
                aux_loss = aux_loss + 0.05 * exit_aux

            # Early exit during inference only
            if not training and i < self.n_moe - 1:
                if gate.should_exit(conf):
                    break  # skip remaining MoE blocks

        # ── 6. Decode patches → per-byte logits ────────────────────────────
        # Remove the format token before decoding
        patches = self.final_norm(patches[:, 1:, :])        # [B, S, d_model]
        logits  = self.decoder(patches, T)                  # [B, T, 256]

        return logits, new_h_list, aux_loss, exit_logits
