# HERMES on Google Colab

The Colab counterpart of the repo's Kaggle workflow (`../kaggle_cell.py`). Same
model, same two-phase training + benchmark loop — adapted for Colab's runtime,
Secrets, and Google Drive.

This folder contains:

| File | Use |
|---|---|
| `hermes_colab.ipynb` | **Recommended.** Ready-to-open notebook with separate setup / train / inspect / charts / compress cells. |
| `colab_cell.py` | Single self-contained cell (paste-and-run) — the direct analog of `kaggle_cell.py`, with auto project-acquisition, Drive persistence, and a hardware switch. |

Nothing in `hermes_train.py` or the model code is Colab-specific; the corpus is
built from HuggingFace streaming + on-machine binaries + the Silesia download,
all of which work identically on Colab.

---

## Option A — the notebook (recommended)

1. Open <https://colab.research.google.com> → **File → Open notebook → GitHub**,
   paste `ocanaChris2/HERMES`, and pick `colab/hermes_colab.ipynb`.
   *(Or **Upload** this `.ipynb` directly.)*
2. **Runtime → Change runtime type → Hardware accelerator: GPU** (T4 is fine).
3. Run the cells top to bottom. Cell 1 clones the repo, installs deps, mounts
   Drive, and wires HuggingFace auth; cell 2 trains.

## Option B — the single paste cell

1. New Colab notebook → set the GPU runtime (as above).
2. Open `colab/colab_cell.py`, copy **all** of it into one cell, and run.
   Edit the small **Config** block at the top first if needed:

   ```python
   HERMES_ACCELERATOR = 'auto'    # 'auto' | 'gpu' | 'tpu' | 'cpu'
   USE_DRIVE          = True      # persist checkpoints to Drive (resume-friendly)
   PROJECT_SOURCE     = 'auto'    # 'auto' | 'git' | 'drive_zip' | 'upload' | 'local'
   P1_EPOCHS, P2_EPOCHS = 10, 15  # full training = 15, 25
   ```

   `PROJECT_SOURCE='auto'` tries `git clone` first, then a `hermes_project.zip`
   in your Drive, then an interactive upload.

---

## HuggingFace token (optional)

The corpus builder falls back to public datasets, so a token is **not** required.
To use one (faster / gated datasets):

1. Click the **🔑 Secrets** icon in Colab's left sidebar.
2. **+ Add new secret** → Name: `HF`, Value: your `hf_...` token.
3. Toggle **Notebook access** on for this notebook.

Both the notebook and `colab_cell.py` read it via `google.colab.userdata.get('HF')`.

---

## Persistence & resuming (Google Drive)

Colab wipes `/content` on disconnect and caps sessions at ~12 h, so a full run
won't finish in one sitting. With `USE_DRIVE = True`:

- `<project>/hermes_output` is symlinked to `MyDrive/HERMES/hermes_output`, so
  every checkpoint, export, and chart is written straight to Drive.
- On reconnect, re-run **setup** then **train** — `resume=True` detects the
  Phase-1/Phase-2 checkpoints on Drive and continues from there.

To start fresh, delete `MyDrive/HERMES/hermes_output` (or set `USE_DRIVE=False`
for a throwaway ephemeral run).

> Writing checkpoints directly to a mounted Drive is convenient but slower than
> local disk. For a one-shot run that you expect to finish before any timeout,
> `USE_DRIVE=False` is faster.

---

## Time estimates (single T4)

| Config | Approx. wall-clock |
|---|---|
| `p1=10, p2=15` (session budget) | ~9 h |
| `p1=15, p2=25` (full training) | ~13–14 h |
| Each benchmark + finetune iteration | ~30–60 min |

A free-tier T4 cannot complete the full run in one session — rely on Drive +
`resume`, or shorten the epoch counts.

---

## Out of memory?

The defaults target a single 15 GiB T4. If you still OOM (or use a smaller GPU),
lower these in `hermes_train.py`:

```python
BATCH_SIZE  = 2     # default 4
P1_SEQ_LEN  = 512   # default 1024
P2_SEQ_LEN  = 1024  # default 2048
```

---

## Colab vs Kaggle — what changed

| Concern | Kaggle (`../kaggle_cell.py`) | Colab (here) |
|---|---|---|
| Project files | Added as a Dataset at `/kaggle/input/...` | `git clone` (default), Drive zip, or upload |
| Secrets | `kaggle_secrets.UserSecretsClient` | `google.colab.userdata` |
| Output dir | `/kaggle/working/hermes_output` (auto-persisted) | `<project>/hermes_output` → symlinked to Drive |
| Accelerators | GPU T4×2 **or** TPU v5e-8 | Single GPU (T4 free; L4/V100/A100 Pro); TPU is best-effort |
| Persistence | Kernel output | Google Drive (`USE_DRIVE`) |

TPU on Colab is offered only as a best-effort path (`HERMES_ACCELERATOR='tpu'`):
`torch_xla` isn't reliably preinstalled and is tightly version-coupled to torch.
**GPU is the supported Colab path.**
