# HERMES
**Hierarchical Entropy Routing Model with Efficient State-spaces**

A neural adaptive byte compressor that operates directly on raw bytes (vocabulary 256) using arithmetic coding (rANS) guided by model-predicted probability distributions. Metric: **BPC (bits per byte)** — lower is better.

---

## Where to run

- **Kaggle** (GPU T4 / TPU v5e-8) — the step-by-step below, or paste [`kaggle_cell.py`](kaggle_cell.py) into one cell.
- **Google Colab** (GPU T4) — see [`colab/README.md`](colab/README.md): open [`colab/hermes_colab.ipynb`](colab/hermes_colab.ipynb) or paste [`colab/colab_cell.py`](colab/colab_cell.py).

---

## Kaggle Notebook — Step-by-Step

### Step 1 — Upload the project as a Kaggle Dataset

1. Go to **kaggle.com → Datasets → New Dataset**
2. Upload `hermes_project.zip`
3. Name it **`hermes-project`** (slug will be `<your-username>/hermes-project`)
4. Set visibility to **Private** and click **Create**

---

### Step 2 — Create a new Kaggle Notebook

1. Go to **Notebooks → New Notebook**
2. In **Settings (right panel)**:
   - Accelerator: **GPU T4 x2** (or GPU T4 single)
   - Internet: **On**
3. Click **Add data** → search for `hermes-project` → add it

---

### Step 3 — Cell 1: Setup environment

```python
import os, sys, shutil, subprocess, gc
import torch

# ── CUDA cleanup from previous runs ──────────────────────────────────────────
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# ── Memory & threading flags ──────────────────────────────────────────────────
os.environ['TOKENIZERS_PARALLELISM']  = 'false'
os.environ['OMP_NUM_THREADS']         = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ── HuggingFace auth (optional) ───────────────────────────────────────────────
# Add a secret named "HF" in Add-ons → Secrets with your HF token.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret("HF")
    from huggingface_hub import login
    login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
    print("HuggingFace: authenticated ✅")
except Exception:
    print("HuggingFace: no token found, using public datasets only")

# ── Install dependencies ──────────────────────────────────────────────────────
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'constriction', 'datasets', 'huggingface_hub'])

# ── Copy project files to working directory ───────────────────────────────────
INPUT   = '/kaggle/input/datasets/christianocanab/hermes-project'
WORKING = '/kaggle/working'

print("Copying project files...")
for item in os.listdir(INPUT):
    src = os.path.join(INPUT, item)
    dst = os.path.join(WORKING, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
print("Files copied ✅")

# ── Clear stale module cache (important after kernel restart) ─────────────────
for key in list(sys.modules):
    if any(x in key for x in ('hermes', 'training', 'coding', 'export', 'benchmarks')):
        del sys.modules[key]

# ── Set working directory and Python path ─────────────────────────────────────
os.chdir(WORKING)
if WORKING not in sys.path:
    sys.path.insert(0, WORKING)

print("Setup complete ✅")
print(f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "No GPU found!")
```

---

### Step 4 — Cell 2: Train + Benchmark loop

```python
from hermes_train import train_hermes

model = train_hermes(
    p1_epochs=10,           # Phase 1 (text)  — reduce for shorter sessions
    p2_epochs=15,           # Phase 2 (binary) — full = 25
    resume=True,            # resume from checkpoint if interrupted
    benchmark_loop=True,    # run iterative benchmark + retrain after training
    target_score=88.0,      # 88 = perfect quality, 62 = minimum pass
    max_bench_iters=None,   # None = unlimited (stops when passed or stagnated)
)
```

> **Time estimates on T4:**
> | Config | Time |
> |---|---|
> | p1=10, p2=15 (session budget) | ~9 h |
> | p1=15, p2=25 (full training) | ~13–14 h |
> | Each benchmark+finetune iteration | ~30–60 min |

---

### Step 5 — Cell 3: Inspect results

```python
import os

OUT = '/kaggle/working/hermes_output'

# List all output files
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(os.path.join(OUT, f))
    print(f'  {f:<45} {size/1024:>8.1f} KB')
```

Output files you'll find:

| File | Description |
|---|---|
| `hermes_ema.pt` | EMA model weights (best for inference) |
| `hermes_model.pt` | TorchScript export |
| `training_curve.png` | Train / val BPC over epochs |
| `benchmark_report_iter000.png` | 5-panel benchmark dashboard (per iteration) |
| `benchmark_iteration_history.png` | Score + BPC trend across retrain iterations |
| `final_dashboard.png` | Combined training + benchmark summary |
| `checkpoints/best.pt` | Best checkpoint by validation BPC |
| `benchmark_best.pt` | Best checkpoint by benchmark score |

---

### Step 6 — Cell 4: Display charts inline

```python
from IPython.display import Image, display

OUT = '/kaggle/working/hermes_output'

for chart in ['training_curve.png',
              'benchmark_report_iter000.png',
              'final_dashboard.png']:
    path = os.path.join(OUT, chart)
    if os.path.exists(path):
        print(f'\n── {chart} ──')
        display(Image(filename=path))
```

---

### Step 7 — Cell 5 (optional): Compress / decompress a file

```python
from hermes_train import compress_file, decompress_file

# Compress any file
compress_file(
    '/kaggle/input/some-dataset/file.bin',
    '/kaggle/working/file.hrm',
    model,
)

# Decompress it back
decompress_file(
    '/kaggle/working/file.hrm',
    '/kaggle/working/file_restored.bin',
    model,
)
```

---

## Resuming after a kernel restart

Just re-run **Cell 1** (setup) then **Cell 2** (train) with `resume=True`. The trainer detects existing checkpoints and skips completed phases automatically.

---

## Reducing memory usage

If you hit OOM errors, reduce in `hermes_train.py`:

```python
BATCH_SIZE  = 2   # default 4
P1_SEQ_LEN  = 512 # default 1024
P2_SEQ_LEN  = 1024 # default 2048
```

---

## Architecture summary

```
raw bytes [B, T]
  BytePatchEncoder  →  [B, T/4, 512]
  MambaBlock × 2    →  selective SSM (chunked scan, memory-efficient)
  SparseMoEAttention × 4  →  local window=128, top-2 of 4 experts
  EarlyExitGate after each block
  PatchByteDecoder  →  [B, T, 256] per-byte logits
```

BPC metric flows through rANS (arithmetic coding) via `constriction`.
