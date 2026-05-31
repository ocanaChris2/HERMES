# ═══════════════════════════════════════════════════════════════════════════════
# HERMES — Single-cell Kaggle pipeline
# Paste the entire contents of this file into one Kaggle notebook cell.
#
# Prerequisites:
#   1. Add dataset "hermes-project" (your upload of hermes_project.zip)
#   2. Accelerator: GPU T4 x2  OR  TPU v5e-8  (see flag below)
#   3. Internet: On
#   4. (Optional) Add a Kaggle Secret named "HF" with your HuggingFace token
#
# ── Accelerator switch ────────────────────────────────────────────────────────
# Change this ONE value to switch between backends:
#
#   'auto'  — GPU if CUDA available, then TPU, then CPU
#   'gpu'   — NVIDIA GPU (T4 / P100)       — set Kaggle accelerator to GPU
#   'tpu'   — Google TPU v5e-8             — set Kaggle accelerator to TPU v5e-8
#   'cpu'   — CPU only (smoke-test)
#
HERMES_ACCELERATOR = 'auto'   # ← change me
# ═══════════════════════════════════════════════════════════════════════════════

import os, sys, shutil, subprocess, gc

os.environ['HERMES_ACCELERATOR'] = HERMES_ACCELERATOR

import torch

W = 62   # banner width

def _banner(title: str = ''):
    print('═' * W)
    if title:
        print(f'  {title}')

def _row(label: str, value: str, ok: bool = True):
    mark = '✅' if ok else '⚠ '
    print(f'  {mark}  {label:<8}  {value}')

_banner('HERMES setup')

# ── 1. Hardware detection & cleanup ───────────────────────────────────────────
gc.collect()

_on_tpu = False
_on_gpu = False

# GPU check first (fastest path, no extra imports)
if HERMES_ACCELERATOR in ('gpu', 'auto') and torch.cuda.is_available():
    _on_gpu = True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _mem = torch.cuda.get_device_properties(0).total_memory // 2**20
    _row('accel', f"GPU  {torch.cuda.get_device_name(0)}  ({_mem} MB)")

# TPU check — only attempt on actual Kaggle TPU VMs (ISTPUVM=1 is set by the Docker image).
# torch_xla is locked inside the Kaggle TPU image and cannot be pip-installed externally.
_is_tpu_vm = os.environ.get('ISTPUVM') == '1'
if not _on_gpu and HERMES_ACCELERATOR in ('tpu', 'auto'):
    if HERMES_ACCELERATOR == 'tpu' and not _is_tpu_vm:
        raise RuntimeError(
            "HERMES_ACCELERATOR=tpu but this is not a Kaggle TPU VM (ISTPUVM is not set).\n"
            "Fix: Settings → Accelerator → TPU v5e-8, then re-run.\n"
            "Or change HERMES_ACCELERATOR to 'gpu' or 'auto'."
        )
    if _is_tpu_vm:
        try:
            import torch_xla.core.xla_model as xm
            _xla_dev   = xm.xla_device()
            _xla_cores = xm.get_xla_supported_devices()
            _n         = len(_xla_cores)
            _row('accel', f"TPU v5e-8  {_n} cores  ({_xla_cores[0]} … {_xla_cores[-1]})")
            _on_tpu = True
        except Exception as e:
            if HERMES_ACCELERATOR == 'tpu':
                raise RuntimeError(f"XLA device unreachable on TPU VM: {e}")
            # auto mode: fall through to CPU

if not _on_gpu and not _on_tpu:
    _row('accel',
         "none — CPU only (very slow)\n"
         "         → Settings → Accelerator → GPU T4 or TPU v5e-8",
         ok=False)

# ── 2. Environment flags ───────────────────────────────────────────────────────
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS']        = '1'
if _on_gpu:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ── 3. HuggingFace auth (optional) ────────────────────────────────────────────
_hf_ok = False
try:
    from kaggle_secrets import UserSecretsClient
    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret("HF")
    from huggingface_hub import login
    login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
    _hf_ok = True
except Exception:
    pass
_row('HF', 'authenticated' if _hf_ok else 'no token — public datasets only', ok=_hf_ok)

# ── 4. Install dependencies ────────────────────────────────────────────────────
# Note: on TPU VMs all packages (torch_xla, tensorflow, libtpu) are locked in the
# Docker image — do not attempt to install/uninstall them.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'constriction', 'datasets', 'huggingface_hub'])
_row('deps', 'constriction  datasets  huggingface_hub')

# ── 5. Copy project files ──────────────────────────────────────────────────────
INPUT   = '/kaggle/input/datasets/christianocanab/hermes-project'
WORKING = '/kaggle/working'

if os.path.exists(INPUT):
    _copied = 0
    for item in os.listdir(INPUT):
        src = os.path.join(INPUT, item)
        dst = os.path.join(WORKING, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        _copied += 1
    _row('files', f'{WORKING}  ({_copied} items)')
else:
    raise FileNotFoundError(
        f"Dataset not found at {INPUT}\n"
        "Make sure you added 'hermes-project' via Add Data in the notebook."
    )

print('═' * W)

# ── 6. Clear stale module cache ────────────────────────────────────────────────
for key in list(sys.modules):
    if any(x in key for x in ('hermes', 'training', 'coding', 'export', 'benchmarks')):
        del sys.modules[key]

os.chdir(WORKING)
if WORKING not in sys.path:
    sys.path.insert(0, WORKING)

# ── 7. Train + benchmark loop ──────────────────────────────────────────────────
print()

from hermes_train import train_hermes

model = train_hermes(
    p1_epochs      = 10,     # Phase 1 text    (full = 15)
    p2_epochs      = 15,     # Phase 2 binary  (full = 25)
    resume         = True,   # skip phases that already have checkpoints
    benchmark_loop = True,   # retrain until benchmark passes
    target_score   = 88.0,   # 88 = perfect, 62 = minimum pass
    max_bench_iters= None,   # None = unlimited (stops on pass or stagnation)
)

# ── 8. Display output charts ───────────────────────────────────────────────────
from IPython.display import Image, display

OUT = '/kaggle/working/hermes_output'
charts = [
    'training_curve.png',
    'benchmark_report_iter000.png',
    'benchmark_iteration_history.png',
    'final_dashboard.png',
]
for chart in charts:
    path = os.path.join(OUT, chart)
    if os.path.exists(path):
        print(f"\n── {chart} ──")
        display(Image(filename=path))

# ── 9. Summary ────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  All outputs saved to:", OUT)
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(os.path.join(OUT, f))
    print(f"  {f:<45} {size/1024:>8.1f} KB")
print("═" * 60)
