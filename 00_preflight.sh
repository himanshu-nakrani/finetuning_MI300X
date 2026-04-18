#!/usr/bin/env bash
# 00_preflight.sh — 10-min environment sanity + 100-step smoke train
# Run BEFORE burning real GPU hours. Catches ROCm/driver/HF/disk issues.
# Budget: ~0.5 GPU-hr (smoke train dominates).
#
# Usage:
#   bash scripts/00_preflight.sh 2>&1 | tee logs/preflight_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/finetune}"
HF_CACHE="${SCRATCH}/cache"
LOG_DIR="${SCRATCH}/logs"
OUT_DIR="${SCRATCH}/outputs/preflight"

# bnb on ROCm shells out to `rocminfo` to detect arch/warp. In tenant containers
# that binary is often missing; hardcode for MI300X (gfx942) to silence warnings.
export BNB_ROCM_ARCH="${BNB_ROCM_ARCH:-gfx942}"

echo "=========================================="
echo " MI300X Bootcamp — Preflight"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="

# -------- 1. Directories --------
echo "[1/8] Creating scratch dirs at ${SCRATCH}"
mkdir -p "${SCRATCH}"/{models,data,outputs,logs,cache}
mkdir -p "${OUT_DIR}"

# -------- 2. Disk space --------
echo "[2/8] Disk space on scratch:"
df -h "${SCRATCH}" | tee -a "${LOG_DIR}/preflight_disk.txt"
AVAIL_GB=$(df -BG "${SCRATCH}" | awk 'NR==2 {gsub("G",""); print $4}')
if [ "${AVAIL_GB}" -lt 500 ]; then
  echo "ERROR: < 500 GB free on ${SCRATCH}. Need ~1 TB headroom for 70B runs." >&2
  exit 1
fi
echo "OK: ${AVAIL_GB} GB free."

# -------- 3. ROCm / GPU --------
echo "[3/8] ROCm + GPU:"
if ! command -v rocm-smi >/dev/null 2>&1; then
  echo "ERROR: rocm-smi not found. Is ROCm installed?" >&2
  exit 1
fi
rocm-smi --showproductname || true
rocm-smi --showmeminfo vram | tee "${LOG_DIR}/preflight_vram.txt"

VRAM_MB=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR==2 {print int($2/1024/1024)}' || echo 0)
echo "Detected VRAM (MB): ${VRAM_MB}"
if [ "${VRAM_MB}" -lt 180000 ]; then
  echo "WARN: VRAM < 180 GB. Expected 192 GB on MI300X. Continuing anyway."
fi

# -------- 4. Python + torch --------
echo "[4/8] Python + torch:"
python - <<'PY'
import sys, platform
print("python:", sys.version.split()[0], platform.platform())
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.version.hip: ", torch.version.hip)

# Reject CUDA wheels on a ROCm box — common footgun (pip install torch grabs +cuXXX from PyPI).
if "+cu" in torch.__version__ or (torch.version.cuda and not torch.version.hip):
    raise SystemExit(
        "ERROR: this is a CUDA build of torch (" + torch.__version__ + ").\n"
        "Reinstall the ROCm wheel:\n"
        "  pip uninstall -y torch torchvision torchaudio pynvml triton\n"
        "  pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch\n"
        "(or:  pip install \"unsloth[rocm]\"  to let unsloth pin the stack)"
    )

if not torch.version.hip:
    raise SystemExit("ERROR: torch has no HIP runtime; not a ROCm build.")

if not torch.cuda.is_available():
    raise SystemExit(
        "torch.cuda.is_available() is False even though this is a ROCm build.\n"
        "Check: rocm-smi sees the GPU, HSA_OVERRIDE_GFX_VERSION is unset, "
        "and the GPU is not in a low-power state (rocm-smi --setperflevel high)."
    )

print("device:", torch.cuda.get_device_name(0))
print("device_count:", torch.cuda.device_count())
free, total = torch.cuda.mem_get_info()
print(f"mem free/total (GB): {free/1e9:.1f} / {total/1e9:.1f}")
PY

# -------- 5. Key packages --------
echo "[5/8] Required packages:"
python - <<'PY'
# Import unsloth FIRST so its monkey-patches land before transformers/peft/trl
# (otherwise unsloth prints a warning and some kernel fast-paths are skipped).
required_first = ["unsloth"]
required = ["transformers", "datasets", "peft", "trl", "accelerate",
            "wandb", "bitsandbytes"]
optional = ["vllm"]      # not needed until Day 2 (synth gen)

missing_req = []
for m in required_first + required:
    try:
        mod = __import__(m)
        print(f"  OK       {m} {getattr(mod,'__version__','?')}")
    except Exception as e:
        missing_req.append(m)
        print(f"  MISSING  {m}: {e}")

for m in optional:
    try:
        mod = __import__(m)
        print(f"  OK opt   {m} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  WARN opt {m}: not installed (only needed Day 2+)")

if missing_req:
    raise SystemExit(f"Install missing required: pip install {' '.join(missing_req)}")
PY

# -------- 6. HF auth --------
echo "[6/8] HuggingFace auth:"
python - <<'PY'
from huggingface_hub import whoami
try:
    info = whoami()
    print("HF user:", info.get("name"))
except Exception as e:
    raise SystemExit(f"HF not logged in. Run: huggingface-cli login\n{e}")
PY

# -------- 7. Wandb (optional but wanted) --------
echo "[7/8] Wandb:"
python - <<'PY'
import os, wandb
if not os.environ.get("WANDB_API_KEY") and not os.path.exists(os.path.expanduser("~/.netrc")):
    print("WARN: wandb not logged in. Run: wandb login")
else:
    print("OK: wandb credentials present")
PY

# -------- 8. Smoke train (100 steps, tiny LoRA on 8B) --------
echo "[8/8] Smoke train (100 steps, Llama-3.1-8B QLoRA, 500 samples)..."
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export WANDB_MODE="${WANDB_MODE:-offline}"   # offline for preflight

python scripts/_smoke_train.py \
    --output_dir "${OUT_DIR}" \
    --max_steps 100 2>&1 | tee "${LOG_DIR}/preflight_smoke.log"

echo ""
echo "=========================================="
echo " PREFLIGHT PASSED"
echo " Artifacts: ${OUT_DIR}"
echo " Logs:      ${LOG_DIR}/preflight_*"
echo "=========================================="
