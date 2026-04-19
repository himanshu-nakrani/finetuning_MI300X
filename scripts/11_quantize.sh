#!/usr/bin/env bash
# 11_quantize.sh — merge a LoRA adapter into its base, convert to GGUF Q4_K_M
# via llama.cpp so the final headline model is consumable on any laptop.
#
# ~1.5 GPU-hr total:
#   - merge (GPU-assisted):     ~10 min for 72B
#   - convert hf -> gguf f16:   ~5 min  (CPU)
#   - quantize f16 -> Q4_K_M:   ~5 min  (CPU)
#   - upload:                   depends on HF bandwidth
#
# Usage:
#   bash scripts/11_quantize.sh \
#       Qwen/Qwen2.5-72B-Instruct \
#       /scratch/finetune/outputs/p4_dpo/adapter \
#       himanshunakrani9/qwen2.5-72b-reasoning-dpo-gguf-q4
#
# Arg 1: base model (HF id or local path)
# Arg 2: adapter path
# Arg 3: HF repo id for the resulting GGUF upload (or "-" to skip push)

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <base> <adapter> <hub_repo_id|->" >&2
  exit 1
fi

BASE="$1"
ADAPTER="$2"
HUB_REPO="$3"

SCRATCH="${SCRATCH:-/scratch/finetune}"
MERGED_DIR="${SCRATCH}/models/p4_merged"
GGUF_DIR="${SCRATCH}/models/p4_gguf"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/tmp/llama.cpp}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${MERGED_DIR}" "${GGUF_DIR}"

echo "=========================================="
echo " Quantize pipeline"
echo "   base:     ${BASE}"
echo "   adapter:  ${ADAPTER}"
echo "   merged:   ${MERGED_DIR}"
echo "   gguf:     ${GGUF_DIR}"
echo "   hub:      ${HUB_REPO}"
echo "=========================================="

# ---------- 1. Merge adapter into base (reuses _merge_adapter.py) ----------
if [ ! -f "${MERGED_DIR}/config.json" ]; then
  echo "[1/4] merging adapter → ${MERGED_DIR}"
  python "${SCRIPT_DIR}/_merge_adapter.py" \
      --base    "${BASE}" \
      --adapter "${ADAPTER}" \
      --output  "${MERGED_DIR}" \
      --dtype   bfloat16 \
      --device_map auto
else
  echo "[1/4] reusing merged at ${MERGED_DIR}"
fi

# ---------- 2. Clone + build llama.cpp (CPU tools only) ----------
if [ ! -d "${LLAMA_CPP_DIR}/.git" ]; then
  echo "[2/4] cloning llama.cpp → ${LLAMA_CPP_DIR}"
  git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "${LLAMA_CPP_DIR}"
fi

if [ ! -x "${LLAMA_CPP_DIR}/llama-quantize" ] && \
   [ ! -x "${LLAMA_CPP_DIR}/build/bin/llama-quantize" ]; then
  echo "[2/4] building llama.cpp CPU tools"
  (
    cd "${LLAMA_CPP_DIR}"
    cmake -B build -DGGML_CUDA=OFF -DGGML_HIPBLAS=OFF
    cmake --build build -j "$(nproc)" --config Release \
          --target llama-quantize llama-cli
  )
fi

QUANTIZE_BIN="${LLAMA_CPP_DIR}/build/bin/llama-quantize"
[ -x "${QUANTIZE_BIN}" ] || QUANTIZE_BIN="${LLAMA_CPP_DIR}/llama-quantize"

# llama.cpp's convert script also needs its Python deps
pip install -q --upgrade "gguf>=0.9" "sentencepiece>=0.2" "protobuf>=3.20"

# ---------- 3. Convert HF → GGUF f16, then quantize to Q4_K_M ----------
F16="${GGUF_DIR}/model.f16.gguf"
Q4="${GGUF_DIR}/model.Q4_K_M.gguf"

if [ ! -f "${F16}" ]; then
  echo "[3/4] HF → GGUF f16"
  python "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
      "${MERGED_DIR}" --outfile "${F16}" --outtype f16
else
  echo "[3/4] reusing f16 at ${F16}"
fi

if [ ! -f "${Q4}" ]; then
  echo "[4/4] quantize f16 → Q4_K_M"
  "${QUANTIZE_BIN}" "${F16}" "${Q4}" Q4_K_M
else
  echo "[4/4] reusing Q4_K_M at ${Q4}"
fi

echo "--- final file sizes ---"
ls -lh "${F16}" "${Q4}"

# ---------- 5. Upload to HF (optional) ----------
if [ "${HUB_REPO}" != "-" ]; then
  echo "[push] uploading ${Q4} → ${HUB_REPO}"
  # Stage with README so the HF repo isn't empty
  STAGE="${GGUF_DIR}/_hf_stage"
  mkdir -p "${STAGE}"
  cp -f "${Q4}" "${STAGE}/"
  cat > "${STAGE}/README.md" <<EOF
# ${HUB_REPO}

GGUF Q4_K_M quantization of the P4 reasoning-DPO model.

- Base: \`${BASE}\`
- Adapter: ${ADAPTER}
- Pipeline: SFT → reasoning-SFT → DPO → merge → GGUF Q4_K_M via llama.cpp

Run locally:

\`\`\`bash
llama-cli -m model.Q4_K_M.gguf -p "Solve: If a train..."
\`\`\`
EOF
  huggingface-cli upload "${HUB_REPO}" "${STAGE}" \
      --commit-message "GGUF Q4_K_M of P4 reasoning-DPO"
fi

echo ""
echo "=========================================="
echo " Quantize OK"
echo "   f16:     ${F16}"
echo "   Q4_K_M:  ${Q4}"
[ "${HUB_REPO}" != "-" ] && echo "   hub:     https://huggingface.co/${HUB_REPO}"
echo "=========================================="
