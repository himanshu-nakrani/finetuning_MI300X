#!/bin/bash
# 01_reasoning_booster_setup.sh - Environment setup for MI300X Reasoning Booster
# Run this on the Digital Ocean MI300X droplet

set -euo pipefail

echo "=========================================="
echo " MI300X Reasoning Booster - Environment Setup"
echo "=========================================="

# Create project directory
mkdir -p /scratch/reasoning_booster
cd /scratch/reasoning_booster

echo "[1/6] Creating virtual environment..."
python3 -m venv unsloth_env
source unsloth_env/bin/activate

echo "[2/6] Detecting ROCm version..."
ROCM_TAG="$(amd-smi version 2>/dev/null | awk -F'ROCm version: ' 'NF>1{split($2,a,"."); print "rocm"a[1]"."a[2]}' || echo rocm7.0)"
echo "Detected ROCm tag: $ROCM_TAG"

echo "[3/6] Installing PyTorch for ROCm..."
pip install --upgrade pip wheel
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$ROCM_TAG" --upgrade --force-reinstall

echo "[4/6] Installing Unsloth (AMD native)..."
pip install --no-deps unsloth unsloth-zoo
pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git
pip install "unsloth[amd] @ git+https://github.com/unslothai/unsloth"

echo "[5/6] Installing additional tools..."
pip install synthetic-data-kit vllm trl accelerate wandb lm-eval datasets peft bitsandbytes

echo "[6/6] Verifying installation..."
python << 'PY'
import torch
import transformers
import trl
import peft

print(f"PyTorch version: {torch.__version__}")
print(f"ROCm/HIP: {torch.version.hip}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Transformers: {transformers.__version__}")
print(f"TRL: {trl.__version__}")
print(f"PEFT: {peft.__version__}")

# Quick bf16 matmul test
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print("bf16 matmul test: PASSED")
print("OK")
PY

echo ""
echo "=========================================="
echo " Environment setup complete!"
echo " Next: Run 02_start_vllm.sh to start vLLM server"
echo "=========================================="
