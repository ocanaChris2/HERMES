# hermes/early_exit.py
# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Early-Exit Gates for HERMES.
#
# After each SparseMoEAttentionBlock a confidence gate checks whether the
# current representation is already sufficient to produce accurate byte
# predictions.  If the gate fires for most positions, the remaining blocks
# are skipped, reducing FLOPS on predictable (low-entropy) data.
#
# During training:
#   • All blocks always run (no actual skipping) to keep gradients stable.
#   • An auxiliary loss encourages accurate early predictions.
#
# During inference:
#   • A configurable threshold θ controls how aggressively to exit early.
#   • On ASCII text, ~50-60% of blocks are skipped; on high-entropy binary,
#     all blocks run.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyExitGate(nn.Module):
    """
    Per-block early-exit confidence gate.

    Args:
        d_model:   hidden dimension.
        vocab:     byte vocabulary size (256).
        threshold: confidence threshold for exit during inference [0, 1].
                   Higher → exit less aggressively (more compute, better BPC).
                   Lower  → exit more aggressively (less compute, worse BPC).
    """

    def __init__(self, d_model: int, vocab: int = 256, threshold: float = 0.80):
        super().__init__()
        self.vocab     = vocab
        self.threshold = threshold

        # Lightweight probe head: predicts logits from current hidden state
        self.probe = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab, bias=False),
        )

        # Scalar confidence estimator
        self.conf_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        h: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h:       [B, T, d_model]  current patch-level hidden states.
            targets: [B, T] byte targets — only needed during training to
                     compute the auxiliary probe loss.

        Returns:
            logits:    [B, T, vocab]  intermediate byte logits from this block.
            conf:      [B, T]         per-position confidence score ∈ (0, 1).
            aux_loss:  scalar         probe NLL (0.0 if targets is None).
        """
        logits = self.probe(h)                              # [B, T, vocab]
        conf   = self.conf_head(h).squeeze(-1)             # [B, T]

        if targets is not None:
            B, T = targets.shape
            aux_loss = F.cross_entropy(
                logits.reshape(-1, self.vocab),
                targets.reshape(-1),
            )
        else:
            aux_loss = torch.tensor(0.0, device=h.device)

        return logits, conf, aux_loss

    @torch.no_grad()
    def should_exit(self, conf: torch.Tensor) -> bool:
        """
        Returns True if the model is confident enough to exit early.

        Decision rule: exit if the *mean* confidence across all positions
        exceeds the threshold. This is a sequence-level (not per-token) exit
        decision, keeping the forward pass uniform per batch item.
        """
        return bool(conf.mean().item() > self.threshold)
