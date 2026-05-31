# ═══════════════════════════════════════════════════════════════════════════════
# HERMES — Single-cell Google Colab pipeline
# Paste the entire contents of this file into one Colab notebook cell and run it.
#
# This is the Colab counterpart of ../kaggle_cell.py.  It clones (or unzips) the
# project, installs deps, optionally mounts Google Drive for persistent
# checkpoints, then runs the full train + benchmark loop.
#
# Prerequisites
#   1. Runtime → Change runtime type → Hardware accelerator: GPU  (T4 is fine;
#      Colab Pro L4 / V100 / A100 also work).  The model is sized for a single
#      15 GiB T4, so no config changes are needed on the free tier.
#   2. (Optional) Secrets (🔑 icon, left sidebar): add one named "HF" holding a
#      HuggingFace token and enable "Notebook access" — lets the corpus builder
#      pull gated/faster datasets.  Without it, public datasets are used.
#   3. (Optional) Mount Google Drive (USE_DRIVE below) so checkpoints survive
#      Colab's idle/12-hour disconnects and `resume=True` can pick up where it
#      left off in a later session.
#
# ── Config — edit these, then run ──────────────────────────────────────────────
HERMES_ACCELERATOR = 'auto'    # 'auto' | 'gpu' | 'tpu' | 'cpu'  (GPU recommended)
USE_DRIVE          = True      # mount Google Drive for persistent outputs/resume
PROJECT_SOURCE     = 'auto'    # 'auto' | 'git' | 'drive_zip' | 'upload' | 'local'

GIT_URL   = 'https://github.com/ocanaChris2/HERMES.git'      # used by 'git'
DRIVE_ZIP = '/content/drive/MyDrive/hermes_project.zip'      # used by 'drive_zip'
DRIVE_OUT = '/content/drive/MyDrive/HERMES/hermes_output'    # persistent outputs

P1_EPOCHS       = 10      # Phase 1 text   (full training = 15)
P2_EPOCHS       = 15      # Phase 2 binary (full training = 25)
TARGET_SCORE    = 88.0    # 88 = perfect quality, 62 = minimum pass
MAX_BENCH_ITERS = None    # None = unlimited (stops on pass / perfect / stagnation)
# ═══════════════════════════════════════════════════════════════════════════════

import os, sys, gc, glob, shutil, zipfile, subprocess

os.environ['HERMES_ACCELERATOR'] = HERMES_ACCELERATOR   # honoured by hermes_train

import torch

W = 62   # banner width

def _banner(title: str = ''):
    print('═' * W)
    if title:
        print(f'  {title}')

def _row(label: str, value: str, ok: bool = True):
    mark = '✅' if ok else '⚠ '
    print(f'  {mark}  {label:<8}  {value}')

_banner('HERMES — Google Colab setup')

# ── 0. Colab detection ─────────────────────────────────────────────────────────
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except Exception:
    IN_COLAB = False
if not IN_COLAB:
    _row('env', 'not a Colab runtime — continuing in local/compat mode', ok=False)

# ── 1. Hardware detection & cleanup ────────────────────────────────────────────
gc.collect()
_on_gpu = _on_tpu = False

if HERMES_ACCELERATOR in ('gpu', 'auto') and torch.cuda.is_available():
    _on_gpu = True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _mem = torch.cuda.get_device_properties(0).total_memory // 2**20
    _row('accel', f"GPU  {torch.cuda.get_device_name(0)}  ({_mem} MB)")

# TPU on Colab is best-effort: torch_xla is not reliably preinstalled and is
# tightly version-coupled to torch.  GPU is the supported path here.
if not _on_gpu and HERMES_ACCELERATOR in ('tpu', 'auto'):
    try:
        import torch_xla.core.xla_model as xm
        _cores = xm.get_xla_supported_devices()
        _row('accel', f"TPU  {len(_cores)} core(s)  ({_cores[0]} … {_cores[-1]})")
        _on_tpu = True
    except Exception as e:
        if HERMES_ACCELERATOR == 'tpu':
            raise RuntimeError(
                f"HERMES_ACCELERATOR=tpu but torch_xla is unavailable ({e}).\n"
                "On Colab, prefer GPU: Runtime → Change runtime type → GPU, then "
                "set HERMES_ACCELERATOR='gpu' (or 'auto')."
            )

if not _on_gpu and not _on_tpu:
    _row('accel',
         "none — CPU only (very slow)\n"
         "         → Runtime → Change runtime type → GPU (T4)",
         ok=False)

# ── 2. Environment flags ───────────────────────────────────────────────────────
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS']        = '1'
if _on_gpu:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ── 3. Mount Google Drive (optional, for persistence) ──────────────────────────
_drive_ok = False
if USE_DRIVE and IN_COLAB:
    try:
        from google.colab import drive
        if not os.path.ismount('/content/drive'):
            drive.mount('/content/drive')
        _drive_ok = os.path.isdir('/content/drive/MyDrive')
    except Exception as e:
        _row('drive', f'mount failed ({e}) — outputs will be ephemeral', ok=False)
if _drive_ok:
    _row('drive', 'mounted  /content/drive/MyDrive')
elif USE_DRIVE:
    _row('drive', 'unavailable — outputs ephemeral (lost on disconnect)', ok=False)

# ── 4. Install dependencies ────────────────────────────────────────────────────
# Colab ships torch / numpy / matplotlib; we only add the compressor + datasets.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'constriction', 'datasets', 'huggingface_hub'])
_row('deps', 'constriction  datasets  huggingface_hub')

# ── 5. HuggingFace auth (optional) ─────────────────────────────────────────────
_hf_ok = False
try:
    _tok = None
    if IN_COLAB:
        from google.colab import userdata
        _tok = userdata.get('HF')         # raises if secret missing / not granted
    _tok = _tok or os.environ.get('HF_TOKEN')
    if _tok:
        os.environ['HF_TOKEN'] = _tok
        from huggingface_hub import login
        login(token=_tok, add_to_git_credential=False)
        _hf_ok = True
except Exception:
    pass
_row('HF', 'authenticated' if _hf_ok else 'no token — public datasets only',
     ok=_hf_ok)

# ── 6. Acquire project files ───────────────────────────────────────────────────
def _proj_root(path: str):
    """Return the dir containing hermes_train.py at `path` or one level below."""
    if path and os.path.isfile(os.path.join(path, 'hermes_train.py')):
        return path
    for d in sorted(glob.glob(os.path.join(path or '.', '*'))):
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, 'hermes_train.py')):
            return d
    return None

def _unzip(zip_path: str, dest: str) -> str:
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return dest

def _acquire(source: str):
    if source == 'local':
        return _proj_root(os.getcwd())

    if source in ('git', 'auto'):
        target = '/content/HERMES'
        root = _proj_root(target)                       # already cloned?
        if root:
            return root
        try:
            subprocess.check_call(
                ['git', 'clone', '--depth', '1', GIT_URL, target])
            root = _proj_root(target)
            if root:
                return root
        except Exception as e:
            if source == 'git':
                raise RuntimeError(f"git clone failed: {e}")
            _row('files', f'git clone failed ({e}) — trying zip/upload', ok=False)

    if source in ('drive_zip', 'auto'):
        if _drive_ok and os.path.isfile(DRIVE_ZIP):
            _unzip(DRIVE_ZIP, '/content/hermes')
            root = _proj_root('/content/hermes')
            if root:
                return root
        elif source == 'drive_zip':
            raise FileNotFoundError(
                f"PROJECT_SOURCE='drive_zip' but no zip at {DRIVE_ZIP}. "
                "Upload hermes_project.zip to your Drive (or fix DRIVE_ZIP).")

    if source in ('upload', 'auto') and IN_COLAB:
        from google.colab import files
        print('  Upload hermes_project.zip …')
        for name in files.upload():
            if name.endswith('.zip'):
                _unzip(name, '/content/hermes')
                root = _proj_root('/content/hermes')
                if root:
                    return root
    return None

PROJ = _acquire(PROJECT_SOURCE)
if not PROJ:
    raise FileNotFoundError(
        "Could not locate hermes_train.py. Set PROJECT_SOURCE to 'git', "
        "'drive_zip', or 'upload' and re-run.")
os.chdir(PROJ)
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
_row('files', PROJ)

# ── 7. Redirect outputs to Drive (persistence + cross-session resume) ──────────
# hermes_train.py writes to <project>/hermes_output when not on Kaggle; we make
# that a symlink into Drive so checkpoints survive disconnects.
LOCAL_OUT = os.path.join(PROJ, 'hermes_output')
if _drive_ok:
    os.makedirs(DRIVE_OUT, exist_ok=True)
    if os.path.islink(LOCAL_OUT):
        if os.path.realpath(LOCAL_OUT) != os.path.realpath(DRIVE_OUT):
            os.unlink(LOCAL_OUT)
            os.symlink(DRIVE_OUT, LOCAL_OUT)
    elif os.path.isdir(LOCAL_OUT):                 # migrate any local outputs
        for item in os.listdir(LOCAL_OUT):
            shutil.move(os.path.join(LOCAL_OUT, item),
                        os.path.join(DRIVE_OUT, item))
        os.rmdir(LOCAL_OUT)
        os.symlink(DRIVE_OUT, LOCAL_OUT)
    elif not os.path.exists(LOCAL_OUT):
        os.symlink(DRIVE_OUT, LOCAL_OUT)
    _row('output', f'{DRIVE_OUT}  (persistent)')
else:
    _row('output', f'{LOCAL_OUT}  (ephemeral)', ok=USE_DRIVE is False)

# ── 8. Clear stale module cache (important after a runtime restart) ─────────────
for key in list(sys.modules):
    if any(x in key for x in ('hermes', 'training', 'coding', 'export', 'benchmarks')):
        del sys.modules[key]

print('═' * W)

# ── 9. Train + benchmark loop ──────────────────────────────────────────────────
print()
from hermes_train import train_hermes

model = train_hermes(
    p1_epochs       = P1_EPOCHS,
    p2_epochs       = P2_EPOCHS,
    resume          = True,            # skip phases that already have checkpoints
    benchmark_loop  = True,            # retrain until benchmark passes
    target_score    = TARGET_SCORE,
    max_bench_iters = MAX_BENCH_ITERS,
)

# ── 10. Display output charts ──────────────────────────────────────────────────
from IPython.display import Image, display

OUT = os.path.join(PROJ, 'hermes_output')
for chart in ['training_curve.png',
              'benchmark_report_iter000.png',
              'benchmark_iteration_history.png',
              'final_dashboard.png']:
    path = os.path.join(OUT, chart)
    if os.path.exists(path):
        print(f"\n── {chart} ──")
        display(Image(filename=path))

# ── 11. Summary ────────────────────────────────────────────────────────────────
print("\n" + "═" * W)
print("  All outputs saved to:", os.path.realpath(OUT))
if os.path.isdir(OUT):
    for f in sorted(os.listdir(OUT)):
        fp = os.path.join(OUT, f)
        if os.path.isfile(fp):
            print(f"  {f:<45} {os.path.getsize(fp)/1024:>8.1f} KB")
print("═" * W)
