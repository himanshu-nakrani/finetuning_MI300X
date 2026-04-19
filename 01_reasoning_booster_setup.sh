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

echo "[1/5] Creating virtual environment..."
python3 -m venv unsloth_env
source unsloth_env/bin/activate

echo "[2/5] Installing MI300X ROCm stack using proven requirements..."
python -m pip install --upgrade pip wheel

# Remove any CUDA packages that might interfere
echo "[clean] Removing CUDA packages if present..."
pip uninstall -y torch torchvision torchaudio pytorch-triton triton pynvml torchao xformers llmcompressor || true

# Install ROCm PyTorch and core dependencies
echo "[install] Installing PyTorch ROCm 6.2..."
pip install --extra-index-url https://download.pytorch.org/whl/rocm6.2 torch==2.5.1+rocm6.2 torchvision==0.20.1+rocm6.2

echo "[install] Installing HuggingFace stack..."
pip install "transformers>=4.45,<4.60" "datasets>=2.20" "accelerate>=1.0" "peft>=0.13" "trl>=0.12" "huggingface_hub>=0.25"

echo "[install] Installing Unsloth..."
pip install --no-deps unsloth unsloth-zoo
pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git
pip install "unsloth[amd] @ git+https://github.com/unslothai/unsloth"

echo "[install] Installing quantization and tracking..."
pip install "bitsandbytes>=0.44" wandb pyyaml

echo "[install] Installing data pipeline and eval tools..."
pip install datasketch text-dedup lm-eval

echo "[install] Installing vLLM for data generation..."
pip install vllm

# Set ROCm architecture for bitsandbytes
echo "[3/5] Setting ROCm environment variables..."
export BNB_ROCM_ARCH=gfx942
if ! grep -q "BNB_ROCM_ARCH" ~/.bashrc 2>/dev/null; then
    echo 'export BNB_ROCM_ARCH=gfx942' >> ~/.bashrc
fi

echo "[4/5] Verifying installation..."
python << 'PY'
import torch
import transformers
import trl
import peft

print(f"PyTorch version: {torch.__version__}")
print(f"ROCm/HIP: {torch.version.hip}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Verify ROCm build
assert "+rocm" in torch.__version__, f"ERROR: torch is not ROCm build: {torch.__version__}"
assert torch.version.hip, f"ERROR: torch.version.hip is None: {torch.__version__}"
assert torch.cuda.is_available(), "ERROR: torch.cuda.is_available() False on ROCm"

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
