import os, sys, shutil, subprocess, gc
import torch

# ── 1. CUDA cleanup from previous runs ────────────────────────────────────────
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# ── 2. Environment flags ───────────────────────────────────────────────────────
os.environ['TOKENIZERS_PARALLELISM']  = 'false'
os.environ['OMP_NUM_THREADS']         = '1'
os.environ['PYTORCH_ALLOC_CONF']      = 'expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ── 3. HuggingFace authentication ─────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret("HF")
except Exception:
    pass

if 'HF_TOKEN' in os.environ:
    from huggingface_hub import login
    login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
    print("HuggingFace: authenticated ✅")

# ── 4. Install dependencies ────────────────────────────────────────────────────
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'constriction', 'datasets', 'huggingface_hub'])

# ── 5. Copy project files to working directory ────────────────────────────────
INPUT   = '/kaggle/input/datasets/christianocanab/hermes-project'
WORKING = '/kaggle/working'

print("Copying project files...")
if os.path.exists(INPUT):
    for item in os.listdir(INPUT):
        s = os.path.join(INPUT, item)
        d = os.path.join(WORKING, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    print("  Done ✅")
else:
    print(f"  ⚠️  Dataset not found at {INPUT}")

# ── 6. Clear stale module cache ───────────────────────────────────────────────
for k in list(sys.modules):
    if any(x in k for x in ('hermes', 'training', 'coding', 'export')):
        del sys.modules[k]

# ── 7. Set working directory and Python path ──────────────────────────────────
os.chdir(WORKING)
if WORKING not in sys.path:
    sys.path.insert(0, WORKING)

# ── 8. Start training ─────────────────────────────────────────────────────────
print("\nStarting Training...")
from hermes_train import train_hermes
model = train_hermes()
