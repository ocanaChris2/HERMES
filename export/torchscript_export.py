# export/torchscript_export.py
# ─────────────────────────────────────────────────────────────────────────────
# Exports the trained HERMES model to TorchScript (.pt) for use with:
#   • LibTorch (C++) — torch::jit::load("hermes_model.pt")
#   • tch-rs   (Rust) — tch::CModule::load("hermes_model.pt")
#
# The Mamba SSM hidden state (h_list) is exposed as explicit tensor args
# so C++/Rust code can manage the state across chunk calls without needing
# any Python runtime.
#
# Usage:
#   from export.torchscript_export import export_hermes
#   export_hermes(model, 'hermes_model.pt', device)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class HERMESInferenceWrapper(nn.Module):
    """
    Thin wrapper around HERMES for TorchScript export.

    Exposes a single `infer_chunk` method that:
      1. Accepts a chunk of bytes + format_id + a flat list of SSM state tensors
      2. Returns logits [T, 256] + updated flat state list

    The state is flattened to List[Tensor] because TorchScript does not
    support nested Optional[List[Optional[Tensor]]].

    C++ call pattern:
        auto [logits, new_states] = wrapper.forward(x, fmt, states);
        // states: vector<at::Tensor> of length n_mamba,
        //   each shape [1, d_inner, d_state]
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model   = model
        self.n_mamba = model.n_mamba

    def forward(
        self,
        x:       torch.Tensor,        # [1, T]   byte indices
        fmt_id:  torch.Tensor,        # [1]       format class ID
        h_flat:  List[torch.Tensor],  # n_mamba × [1, d_inner, d_state] or []
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            logits:  [T, 256]            per-byte logits
            h_flat:  updated SSM states  (pass back on next chunk call)
        """
        # Reconstruct h_list from flat list
        h_list: Optional[List[torch.Tensor]]
        if len(h_flat) == 0:
            h_list = None
        else:
            h_list = list(h_flat)

        T = x.shape[1]
        logits, new_h_list, _, _ = self.model(
            x, fmt_id, h_list=h_list, targets=None, training=False
        )

        logits_out  = logits[0]          # [T, 256] — remove batch dim
        h_flat_out  = list(new_h_list)   # List[Tensor]

        return logits_out, h_flat_out


def export_hermes(
    model:     nn.Module,
    save_path: str,
    device:    torch.device,
    chunk_size: int = 4096,
    patch_size: int = 4,
) -> str:
    """
    Trace + script the HERMES model and save as a TorchScript archive.

    Args:
        model:      trained HERMES instance (EMA weights already applied).
        save_path:  destination .pt file path.
        device:     inference device for tracing.
        chunk_size: byte chunk size used during compression.
        patch_size: must match model.patch_size.

    Returns:
        Absolute path to the saved .pt file.
    """
    model.eval()
    model.to(device)

    wrapper = HERMESInferenceWrapper(model)
    wrapper.eval()

    # Dummy inputs for tracing
    seq_len = chunk_size
    x_dummy   = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    fmt_dummy = torch.zeros(1, dtype=torch.long, device=device)
    h_dummy: List[torch.Tensor] = []   # empty → model starts from zero state

    print(f'[export] Scripting HERMES wrapper …')

    # Try torch.jit.script first (preferred — handles control flow)
    try:
        scripted = torch.jit.script(wrapper)
        print('  torch.jit.script ✓')
    except Exception as e:
        print(f'  torch.jit.script failed ({e}), falling back to trace …')
        scripted = torch.jit.trace(
            wrapper, (x_dummy, fmt_dummy, h_dummy),
            strict=False,
        )
        print('  torch.jit.trace ✓')

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    scripted.save(save_path)
    size_mb = os.path.getsize(save_path) / 1e6
    print(f'[export] Saved → {save_path}  ({size_mb:.1f} MB)')

    # Quick smoke test
    with torch.no_grad():
        logits_out, h_out = scripted(x_dummy, fmt_dummy, h_dummy)
        assert logits_out.shape == (seq_len, 256), \
            f'Unexpected logits shape: {logits_out.shape}'
        assert len(h_out) == model.n_mamba, \
            f'Unexpected h_list length: {len(h_out)}'
    print(f'[export] Smoke test passed ✓  '
          f'(logits {tuple(logits_out.shape)}, '
          f'{len(h_out)} SSM states)')

    return os.path.abspath(save_path)


# ─────────────────────────────────────────────────────────────────────────────
# C++ / Rust integration notes (printed as comments in the exported model)
# ─────────────────────────────────────────────────────────────────────────────

CPP_EXAMPLE = '''
// ── LibTorch C++ example ──────────────────────────────────────────────────
// #include <torch/script.h>
//
// auto model = torch::jit::load("hermes_model.pt");
// model.eval();
//
// // Initial state: empty vector (model resets to zeros)
// std::vector<torch::jit::IValue> states;
//
// for (auto& chunk : chunks) {
//     auto x   = torch::from_blob(chunk.data(), {1, (long)chunk.size()},
//                                 torch::kInt64).to(device);
//     auto fmt = torch::tensor({format_id}, torch::kInt64).to(device);
//
//     auto inputs = c10::impl::GenericList(torch::TensorType::get());
//     for (auto& s : states) inputs.push_back(s.toTensor());
//
//     auto out     = model.forward({x, fmt, inputs}).toTuple();
//     auto logits  = out->elements()[0].toTensor();   // [T, 256]
//     auto new_st  = out->elements()[1].toList();     // updated states
//
//     states.clear();
//     for (size_t i = 0; i < new_st.size(); ++i)
//         states.push_back(new_st.get(i).toTensor().detach());
//
//     // Use logits to build CDFs and call your Rust/C++ rANS coder
// }
'''

RUST_EXAMPLE = '''
// ── tch-rs Rust example ───────────────────────────────────────────────────
// use tch::{CModule, Tensor, Kind, Device};
//
// let model = CModule::load("hermes_model.pt")?;
//
// let mut states: Vec<Tensor> = vec![];
//
// for chunk in &chunks {
//     let x   = Tensor::of_slice(chunk).unsqueeze(0).to_kind(Kind::Int64);
//     let fmt = Tensor::of_slice(&[format_id as i64]);
//     let st  = tch::IValue::TensorList(states.clone());
//
//     let out    = model.forward_is(&[IValue::Tensor(x),
//                                    IValue::Tensor(fmt), st])?;
//     let logits = out.0;          // Tensor [T, 256]
//     states     = out.1;          // Vec<Tensor> updated states
//     // rANS encode using `constriction` crate with logits-derived CDFs
// }
'''

if __name__ == '__main__':
    print(CPP_EXAMPLE)
    print(RUST_EXAMPLE)
