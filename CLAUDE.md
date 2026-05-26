# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

HERMES (Hierarchical Entropy Routing Model with Efficient State-spaces) is a neural adaptive byte compressor. It operates directly on raw bytes (vocabulary size 256) and uses arithmetic coding (rANS) with model-predicted probability distributions to achieve compression. The metric throughout is **BPC (bits per byte)** — lower is better.

## Running training

```bash
# Full two-phase training (designed for Kaggle T4 GPU, ~13-14 hours)
python hermes_train.py

# Programmatic control
python -c "from hermes_train import train_hermes; train_hermes(p1_epochs=10, p2_epochs=15)"
```

On Kaggle, use `kaggle_cell.py` as the setup cell before training — it copies files, installs deps (`constriction`, `datasets`), and sets CUDA memory env vars.

## Dependencies

```bash
pip install torch constriction datasets huggingface_hub numpy matplotlib
```

`constriction` is the Rust-backed rANS coder. A pure-Python fallback exists in `coding/coder.py` but is much slower.

## Compress / decompress a file

```python
from hermes import HERMES, sniff
from coding.coder import compress, decompress
import torch

model = HERMES()  # load weights separately
device = torch.device('cuda')
model.load_state_dict(torch.load('hermes_output/hermes_ema.pt')['model'])
model.to(device).eval()

compressed   = compress(open('file.bin', 'rb').read(), model, device)
decompressed = decompress(compressed, model, device)
```

## Architecture overview

The data flow through the model is:

```
raw bytes [B, T]
    │
BytePatchEncoder (hermes/byte_patch.py)
    │  Byte embedding (d_byte=64) → strided Conv1d (stride=patch_size=4)
    │  → [B, T/4, d_model=512]
    │
format token prepended  (hermes/format_sniffer.py sniffs first 64 bytes)
    │  → [B, T/4+1, d_model]
    │
MambaBlock × 2  (hermes/mamba_block.py)
    │  Pure-PyTorch selective SSM with chunked scan (memory-efficient)
    │  SSM hidden states h_list are carried across chunk calls at inference
    │
SparseMoEAttentionBlock × 4  (hermes/moe_attention.py)
  + EarlyExitGate after each  (hermes/early_exit.py)
    │  Local-window attention (window=128 patch tokens), 4 experts, top-2 routing
    │  Exit gates skip remaining blocks on low-entropy (predictable) data
    │
PatchByteDecoder (hermes/byte_patch.py)
    │  ConvTranspose1d → per-position MLP
    └→ [B, T, 256]  per-byte logits
```

**Stateful inference**: `h_list` (Mamba SSM states) is passed explicitly between chunk calls so information from earlier parts of a file propagates to later predictions. Always detach before passing to the next chunk:  `h_list = [h.detach() for h in h_list]`.

## Training pipeline

`training/trainer.py` — `HERMESTrainer`:
- Phase 1 (text): wikitext-103 + code, `lr=2e-4`, 15 epochs
- Phase 2 (binary): ELF binaries + `.pyc` + Silesia corpus, `lr=5e-5`, 25 epochs
- AMP (float16) + gradient accumulation (`accum_steps=8`, effective batch=32)
- EMA weights (decay=0.9995) used for inference/export
- **OTTA meta-loss**: during training, randomly resets SSM state mid-sequence and penalises slow recovery — forces the SSM to be a fast in-context adapter

`training/data_pipeline.py` — `CorpusBuilder` streams from HuggingFace (no full download) and falls back to local system binaries + Silesia corpus for Phase 2.

## Loss function

Three components (in `training/trainer.py::compute_loss`):
1. **Main NLL** on byte targets
2. **Router load-balance loss** × 0.01 per MoE block (prevents expert collapse)
3. **Early-exit distillation** (KL from early-exit logits to final logits × 0.2)

## Entropy coding (coding/coder.py)

Stream format: `HRM\x01` magic + format_id + original_size (uint64) + sha256 + chunk_sizes + ANS-coded payloads.

Logits → CDF quantisation uses `PROB_BITS=14` (16384-entry CDF). The `batch_logits_to_cdfs` function vectorises softmax and floor rounding with a per-row rounding correction to guarantee `sum(counts) == PROB_SCALE` deterministically on both encoder and decoder.

## TorchScript export (export/torchscript_export.py)

Exports `HERMESInferenceWrapper` — a thin wrapper that flattens `h_list` to `List[Tensor]` (TorchScript doesn't support `Optional[List[Optional[Tensor]]]`). Used by LibTorch (C++) or tch-rs (Rust) for native inference without Python. The file contains C++ and Rust usage examples in `CPP_EXAMPLE` / `RUST_EXAMPLE` constants.

## Key design constraints

- **Chunked SSM scan** (`SCAN_CHUNK=32` in `mamba_block.py`): avoids the parallel Blelloch scan which needs 256+ MiB intermediates on T4. Peak memory is ~4 MiB for default shapes.
- **Local window attention** (`window=128` patch tokens = 512 bytes): bounds quadratic attention cost; uses `F.scaled_dot_product_attention` to avoid float16 overflow and enable Flash Attention on Ampere/T4.
- **Per-token MoE routing** (not per-sequence): each token independently picks top-2 experts — the original per-sequence mean routing defeated the purpose of MoE for variable content.
- **patch_size=4**: reduces sequence length 4× for the expensive SSM/attention layers, enabling 4 KB samples with only 1024 patch tokens.
