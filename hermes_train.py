# hermes_train.py
# ─────────────────────────────────────────────────────────────────────────────
# HERMES — Kaggle Notebook Entry Point
#
# Run this cell in a Kaggle notebook (GPU T4 or TPU v5e-8):
#
#   import subprocess, sys
#   subprocess.check_call([sys.executable, '-m', 'pip', 'install',
#                          '-q', 'constriction', 'datasets'])
#   exec(open('hermes_train.py').read())
#
# Or simply:  model = train_hermes()
#
# ── Accelerator selection ────────────────────────────────────────────────────
# Set the HERMES_ACCELERATOR environment variable before importing to choose
# which hardware backend to use.  Accepted values:
#
#   auto  (default) — GPU if CUDA is available, else TPU if XLA is reachable,
#                     else CPU.
#   gpu             — Force CUDA GPU (raises if unavailable).
#   tpu             — Force TPU via torch_xla (raises if torch_xla missing or
#                     no XLA device is reachable).
#   cpu             — Force CPU (useful for quick smoke-tests).
#
# Example (set before launching the notebook cell):
#   import os; os.environ['HERMES_ACCELERATOR'] = 'tpu'
#
# ── Estimated wall-clock times ───────────────────────────────────────────────
# T4  GPU:      Phase 1 ~5 h, Phase 2 ~8 h,  total ~13–14 h
# TPU v5e-8:    Phase 1 ~2 h, Phase 2 ~4 h,  total  ~6–7 h  (estimated)
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, math, time, warnings, subprocess
warnings.filterwarnings('ignore')

# ── Install optional deps (safe to re-run) ───────────────────────────────────
def _pip_install(*pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])

try:
    import constriction
except ImportError:
    print('Installing constriction …')
    _pip_install('constriction')

try:
    import datasets as _ds  # noqa
except ImportError:
    print('Installing datasets …')
    _pip_install('datasets')

# ── Std imports ───────────────────────────────────────────────────────────────
import torch

# Kaggle reuses the C++ PyTorch extension across kernel restarts: the operator
# registry (C++ side) already has entries from the previous run while
# sys.modules is empty, so both Library.define and Library.impl raise on
# re-registration. Patch both to swallow those specific errors, import the
# module that triggers them, then restore the originals.
import torch.library as _tl, sys as _sys
_orig_define = _tl.Library.define
_orig_impl   = _tl.Library.impl
_DUP_MSGS = ("Duplicate registration", "already a kernel registered from python")
def _safe_define(self, schema, *args, **kwargs):
    try:
        return _orig_define(self, schema, *args, **kwargs)
    except RuntimeError as e:
        if not any(m in str(e) for m in _DUP_MSGS):
            raise
def _safe_impl(self, op_name, fn, *args, **kwargs):
    try:
        return _orig_impl(self, op_name, fn, *args, **kwargs)
    except RuntimeError as e:
        if not any(m in str(e) for m in _DUP_MSGS):
            raise
_tl.Library.define = _safe_define
_tl.Library.impl   = _safe_impl
# Remove cached module so it re-executes under the patched methods
_sys.modules.pop("torch.export.custom_ops", None)
import torch.export.custom_ops
_tl.Library.define = _orig_define
_tl.Library.impl   = _orig_impl
del _tl, _sys, _orig_define, _orig_impl, _safe_define, _safe_impl, _DUP_MSGS

import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Project imports ───────────────────────────────────────────────────────────
# __file__ is undefined when run via exec() in a Jupyter cell
try:
    _proj_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _proj_dir = os.getcwd()
sys.path.insert(0, _proj_dir)

from hermes                  import HERMES, sniff
from training.data_pipeline  import CorpusBuilder, build_loaders
from training.trainer        import HERMESTrainer
from coding.coder            import compress, decompress
from export.torchscript_export import export_hermes
from benchmarks              import BenchmarkSuite, BenchmarkDrivenRetrainer, BenchmarkVisualizer

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Tune these to fit your Kaggle session time budget.
# For a 9-hour session: P1_EPOCHS=10, P2_EPOCHS=15
# For full training:   P1_EPOCHS=15, P2_EPOCHS=25

SEED = 42

# ── Accelerator flag ──────────────────────────────────────────────────────────
# Override with:  os.environ['HERMES_ACCELERATOR'] = 'tpu' | 'gpu' | 'cpu'
# or set ACCELERATOR directly here for a hard-coded default.
ACCELERATOR = os.environ.get('HERMES_ACCELERATOR', 'auto').lower()


def _resolve_device() -> torch.device:
    """Return the torch.device to use, honouring HERMES_ACCELERATOR."""
    if ACCELERATOR == 'gpu':
        if not torch.cuda.is_available():
            raise RuntimeError("HERMES_ACCELERATOR=gpu but no CUDA device found.")
        return torch.device('cuda')

    if ACCELERATOR == 'tpu':
        try:
            import torch_xla.core.xla_model as xm  # noqa: PLC0415
            dev = xm.xla_device()
            return dev
        except ImportError:
            raise RuntimeError(
                "HERMES_ACCELERATOR=tpu but torch_xla is not installed. "
                "On Kaggle TPU notebooks it should be pre-installed."
            )
        except Exception as exc:
            raise RuntimeError(f"HERMES_ACCELERATOR=tpu: XLA device unavailable — {exc}")

    if ACCELERATOR == 'cpu':
        return torch.device('cpu')

    # auto: GPU → TPU → CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        import torch_xla.core.xla_model as xm  # noqa: PLC0415
        return xm.xla_device()
    except Exception:
        pass
    return torch.device('cpu')


DEVICE = _resolve_device()

# Paths (Kaggle working directory)
OUT_DIR       = '/kaggle/working/hermes_output'
CKPT_DIR      = os.path.join(OUT_DIR, 'checkpoints')
DATA_DIR      = os.path.join(OUT_DIR, 'data')
MODEL_PT      = os.path.join(OUT_DIR, 'hermes_model.pt')
EMA_PT        = os.path.join(OUT_DIR, 'hermes_ema.pt')

# Fall back to local paths when not on Kaggle
if not os.path.exists('/kaggle'):
    OUT_DIR  = os.path.join(os.path.dirname(__file__), 'hermes_output')
    CKPT_DIR = os.path.join(OUT_DIR, 'checkpoints')
    DATA_DIR = os.path.join(OUT_DIR, 'data')
    MODEL_PT = os.path.join(OUT_DIR, 'hermes_model.pt')
    EMA_PT   = os.path.join(OUT_DIR, 'hermes_ema.pt')

for d in [OUT_DIR, CKPT_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

# Model hyper-params (22 M params target, T4-safe)
MODEL_CFG = dict(
    d_model        = 512,
    d_byte         = 64,
    d_state        = 16,
    patch_size     = 4,
    n_mamba        = 2,
    n_moe          = 4,
    n_experts      = 4,
    top_k          = 2,
    n_heads        = 8,
    window         = 128,
    dropout        = 0.1,
    exit_threshold = 0.80,
)

# Training hyper-params  (T4 / 15 GiB safe)
# Chunked scan is memory-efficient; smaller batch + shorter seqlen give the
# GPU more headroom for backward activations and AMP buffers.
P1_SEQ_LEN    = 1024    # Phase 1: text  — 1024 tokens × patch_size=4 = 4 KB/sample
P2_SEQ_LEN    = 2048    # Phase 2: binary — 2048 tokens = 8 KB/sample
BATCH_SIZE    = 4       # micro-batch; effective batch = 4 × ACCUM_STEPS = 32
ACCUM_STEPS   = 8       # gradient accumulation steps
P1_SAMPLES    = 20_000  # samples per Phase 1 epoch
P2_SAMPLES    = 15_000  # samples per Phase 2 epoch
P1_EPOCHS     = 15
P2_EPOCHS     = 25
P1_LR         = 2e-4
P2_LR         = 5e-5
CHUNK_SIZE    = 4096    # compression chunk size (bytes)


# ── Main ──────────────────────────────────────────────────────────────────────

def train_hermes(
    p1_epochs:         int   = P1_EPOCHS,
    p2_epochs:         int   = P2_EPOCHS,
    resume:            bool  = True,
    benchmark_loop:    bool  = True,   # run iterative benchmark+retrain after training
    max_bench_iters:   int | None = None,  # None = unlimited (until pass/perfect/stagnation)
    target_score:      float = 88.0,   # PERFECT_THRESHOLD by default
) -> HERMES:

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Build model ──────────────────────────────────────────────────────────
    model = HERMES(**MODEL_CFG).to(DEVICE)

    # ── Startup banner ───────────────────────────────────────────────────────
    W = 62
    if DEVICE.type == 'cuda':
        _hw = (f"GPU  {torch.cuda.get_device_name(0)}"
               f"  ({torch.cuda.get_device_properties(0).total_memory // 2**20} MB)")
    elif str(DEVICE).startswith('xla'):
        try:
            import torch_xla.core.xla_model as xm  # noqa: PLC0415
            _cores = xm.get_xla_supported_devices()
            _hw = f"TPU v5e-8  {len(_cores)} cores  ({_cores[0]} … {_cores[-1]})"
        except Exception:
            _hw = f"XLA  {DEVICE}"
    else:
        _hw = 'CPU'
    print('═' * W)
    print(f'  HERMES  {model.n_params():,} params  ·  {_hw}')
    print('═' * W)

    trainer = HERMESTrainer(
        model, CKPT_DIR, DEVICE,
        lr=P1_LR, accum_steps=ACCUM_STEPS,
        otta_prob=0.3,
    )

    # ── Corpus ───────────────────────────────────────────────────────────────
    builder = CorpusBuilder(DATA_DIR, max_bytes_per_source=25_000_000)

    # ── Phase 1: Text ────────────────────────────────────────────────────────
    p1_ckpt = os.path.join(CKPT_DIR, 'text_latest.pt')
    if resume and os.path.exists(p1_ckpt):
        # Skipping Phase 1 training must still restore the Phase 1 weights into
        # the (otherwise randomly-initialised) model + EMA — otherwise Phase 2
        # would train from scratch on binary data.
        print('\n[Phase 1] Checkpoint found — restoring weights, skipping to Phase 2')
        _ck = torch.load(p1_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(_ck['model'])
        try:
            trainer.ema_model.load_state_dict(_ck['ema_model'])
        except Exception as _e:
            print(f'  (EMA state not restored: {_e}; EMA will re-seed from model)')
        model.to(DEVICE)
        del _ck
    else:
        print('\n' + '═'*60)
        print(' Phase 1 — Text corpus (wikitext + GitHub code)')
        print('═'*60)
        text_bufs = builder.build_text_buffers()
        trn1, val1 = build_loaders(
            text_bufs, seq_len=P1_SEQ_LEN,
            n_samples=P1_SAMPLES, batch_size=BATCH_SIZE,
        )
        trainer.run_phase(trn1, val1, n_epochs=p1_epochs,
                          lr=P1_LR, phase_name='text')

    # ── Phase 2: Binary ───────────────────────────────────────────────────────
    print('\n' + '═'*60)
    print(' Phase 2 — Binary corpus (ELF + pyc + Silesia)')
    print('═'*60)
    bin_bufs = builder.build_binary_buffers()
    trn2, val2 = build_loaders(
        bin_bufs, seq_len=P2_SEQ_LEN,
        n_samples=P2_SAMPLES, batch_size=BATCH_SIZE,
    )
    p2_ckpt = os.path.join(CKPT_DIR, 'binary_latest.pt')
    trainer.run_phase(trn2, val2, n_epochs=p2_epochs,
                      lr=P2_LR, phase_name='binary',
                      resume_path=p2_ckpt if resume else None)

    # ── Phase 3: EMA calibration + export ───────────────────────────────────
    print('\n' + '═'*60)
    print(' Phase 3 — Export')
    print('═'*60)
    trainer.export_ema(EMA_PT)

    # Load EMA weights into model
    ckpt = torch.load(EMA_PT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(DEVICE).eval()

    # TorchScript export (best-effort). A scripting/tracing failure must not
    # abort the run: hermes_ema.pt above is the canonical inference artifact,
    # and the benchmark loop below still needs to run.
    try:
        export_hermes(model, MODEL_PT, DEVICE, chunk_size=CHUNK_SIZE,
                      patch_size=MODEL_CFG['patch_size'])
    except Exception as e:
        print(f'  ⚠️  TorchScript export failed ({type(e).__name__}: {e})')
        print(f'      Continuing — EMA weights are saved at {EMA_PT}')

    # ── Roundtrip tests ──────────────────────────────────────────────────────
    print('\n' + '═'*60)
    print(' Roundtrip verification')
    print('═'*60)
    _run_roundtrip_tests(model)

    # ── Training curve ───────────────────────────────────────────────────────
    _plot_history(trainer.history, os.path.join(OUT_DIR, 'training_curve.png'))

    # ── Benchmark-driven retrain loop ─────────────────────────────────────────
    if benchmark_loop:
        print('\n' + '═'*60)
        print(' Benchmark + Iterative Retrain Loop')
        print('═'*60)
        retrainer = BenchmarkDrivenRetrainer(
            model          = model,
            trainer        = trainer,
            corpus_builder = builder,
            device         = DEVICE,
            out_dir        = OUT_DIR,
            chunk_size     = CHUNK_SIZE,
            target_score   = target_score,
        )
        model = retrainer.run(max_iterations=max_bench_iters)

        # Final combined dashboard
        BenchmarkVisualizer().plot_training_dashboard(
            trainer.history,
            retrainer.history,
            os.path.join(OUT_DIR, 'final_dashboard.png'),
        )

    print(f'\nAll outputs in: {OUT_DIR}')
    return model


def _run_roundtrip_tests(model: HERMES):
    test_cases = [
        ('ASCII text',   b'Hello, HERMES! ' * 200),
        ('Binary zeros', bytes(1024)),
        ('Random bytes', bytes(np.random.randint(0, 256, 1024, dtype=np.uint8))),
        ('Source code',  b'def compress(x):\n    return model(x)\n' * 100),
    ]
    all_ok = True
    for name, data in test_cases:
        try:
            compressed   = compress(data, model, DEVICE,
                                    chunk_size=CHUNK_SIZE, verbose=False)
            decompressed = decompress(compressed, model, DEVICE,
                                      chunk_size=CHUNK_SIZE, verbose=False)
            ok = decompressed == data
            bpc = len(compressed) * 8 / len(data)
            status = '✅' if ok else '❌'
            print(f'  {status} {name:<18} | {len(data):>6} B → '
                  f'{len(compressed):>6} B  ({bpc:.3f} BPC)')
            if not ok:
                all_ok = False
        except Exception as e:
            print(f'  ❌ {name}: {e}')
            all_ok = False

    if all_ok:
        print('\n  All roundtrip tests passed ✅')
    else:
        print('\n  ⚠️  Some roundtrip tests FAILED — check model state')


def _plot_history(history: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    if history['train_bpc']:
        ax.plot(history['train_bpc'], label='Train BPC', linewidth=2)
    if history['val_bpc']:
        ax.plot(history['val_bpc'], label='Val BPC',  linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Bits per byte (BPC)')
    ax.set_title('HERMES Training Curve')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Training curve saved → {save_path}')


# ── Quick-compress CLI (test a file after training) ───────────────────────────

def compress_file(src: str, dst: str, model: HERMES):
    with open(src, 'rb') as f:
        raw = f.read()
    out = compress(raw, model, DEVICE, chunk_size=CHUNK_SIZE)
    with open(dst, 'wb') as f:
        f.write(out)
    print(f'{src} → {dst}')


def decompress_file(src: str, dst: str, model: HERMES):
    with open(src, 'rb') as f:
        comp = f.read()
    raw = decompress(comp, model, DEVICE, chunk_size=CHUNK_SIZE)
    with open(dst, 'wb') as f:
        f.write(raw)
    print(f'{src} → {dst}')


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = train_hermes()
