#!/usr/bin/env bash
# install_mi300x.sh — one-shot installer for the MI300X / ROCm 6.2 stack.
#
# Idempotent. Safe to re-run. Creates a venv at ./.venv if not present,
# wipes any CUDA torch / known-broken packages, then installs the pinned
# ROCm stack from requirements-mi300x.txt.
#
# Usage:
#   bash install_mi300x.sh                 # create/use ./.venv
#   VENV=/path/to/venv bash install_mi300x.sh
#   SKIP_VENV=1 bash install_mi300x.sh     # install into the active env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="${SCRIPT_DIR}/requirements-mi300x.txt"
VENV="${VENV:-${SCRIPT_DIR}/.venv}"

if [ ! -f "${REQ}" ]; then
  echo "ERROR: ${REQ} not found." >&2
  exit 1
fi

# -------- venv --------
if [ -z "${SKIP_VENV:-}" ]; then
  if [ ! -d "${VENV}" ]; then
    echo "[venv] creating ${VENV}"
    python3.12 -m venv "${VENV}" || python3 -m venv "${VENV}"
  fi
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  echo "[venv] active: $(command -v python)"
fi

python -m pip install --upgrade pip wheel

# -------- nuke known-bad packages from any prior install --------
echo "[clean] removing CUDA / incompatible leftovers"
pip uninstall -y \
    torch torchvision torchaudio \
    pytorch-triton pytorch-triton-rocm triton \
    pynvml \
    torchao xformers llmcompressor \
    || true   # ok if not installed

# -------- main install (order in the file matters) --------
echo "[install] pip install -r ${REQ}"
pip install -r "${REQ}"

# -------- env --------
if ! grep -q "BNB_ROCM_ARCH" ~/.bashrc 2>/dev/null; then
  echo 'export BNB_ROCM_ARCH=gfx942' >> ~/.bashrc
  echo "[env] appended BNB_ROCM_ARCH=gfx942 to ~/.bashrc"
fi
export BNB_ROCM_ARCH=gfx942

# -------- verify --------
echo "[verify] sanity check"
python - <<'PY'
import torch, torchvision
assert "+rocm" in torch.__version__, f"torch is not the ROCm build: {torch.__version__}"
assert torch.version.hip,            f"torch.version.hip is None: {torch.__version__}"
assert torch.cuda.is_available(),    "torch.cuda.is_available() False on ROCm"
print("torch:      ", torch.__version__)
print("hip:        ", torch.version.hip)
print("torchvision:", torchvision.__version__)
print("device:     ", torch.cuda.get_device_name(0))

# Quick bf16 matmul to prove kernels load
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
y = x @ x; torch.cuda.synchronize()
print("bf16 matmul peak MB:", torch.cuda.max_memory_allocated() // 1024**2)

# Confirm the ecosystem libs import cleanly (this is where torchao etc. used to fail)
import transformers, datasets, peft, trl, accelerate, bitsandbytes, unsloth
print("transformers:", transformers.__version__)
print("trl:         ", trl.__version__)
print("peft:        ", peft.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
print("unsloth:     ", unsloth.__version__)
print("OK")
PY

echo ""
echo "=========================================="
echo " MI300X stack installed."
echo " venv:  ${VENV}"
echo " next:  source ${VENV}/bin/activate"
echo "=========================================="
