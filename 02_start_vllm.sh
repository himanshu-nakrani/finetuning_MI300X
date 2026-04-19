#!/bin/bash
# 02_start_vllm.sh - Start vLLM server for synthetic data generation
# Run this on the Digital Ocean MI300X droplet

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
source .venv/bin/activate

echo "=========================================="
echo " Starting vLLM server on port 8001"
echo "=========================================="

vllm serve unsloth/Llama-3.3-70B-Instruct \
  --port 8001 \
  --max-model-len 48000 \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 1
