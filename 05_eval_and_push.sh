#!/bin/bash
# 05_eval_and_push.sh - Evaluate models and push to Hugging Face
# Run on Digital Ocean MI300X droplet after GRPO training

set -euo pipefail

cd /scratch/reasoning_booster
source .venv/bin/activate

echo "=========================================="
echo " Evaluation and Shipping"
echo "=========================================="

mkdir -p results

echo "[1/4] Evaluating GRPO model on GSM8K..."
lm_eval --model hf \
  --model_args pretrained=./reasoning_booster_grpo_merged \
  --tasks gsm8k \
  --batch_size 8 \
  --output_path results/gsm8k_grpo.json

echo "[2/4] Evaluating GRPO model on HumanEval..."
lm_eval --model hf \
  --model_args pretrained=./reasoning_booster_grpo_merged \
  --tasks humaneval \
  --batch_size 8 \
  --output_path results/humaneval_grpo.json

echo "[3/4] Evaluating baseline model..."
lm_eval --model hf \
  --model_args pretrained=unsloth/Llama-3.3-70B-Instruct \
  --tasks gsm8k,humaneval \
  --batch_size 8 \
  --output_path results/baseline.json

echo "[4/4] Pushing to Hugging Face..."
python << 'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import os

# Get HF username from environment or prompt
hf_username = os.getenv("HF_USERNAME", input("Enter your Hugging Face username: "))

api = HfApi()

# Push merged model
print("Pushing merged model...")
model = AutoModelForCausalLM.from_pretrained("./reasoning_booster_grpo_merged")
tokenizer = AutoTokenizer.from_pretrained("./reasoning_booster_grpo_merged")

model.push_to_hub(f"{hf_username}/mi300x-reasoning-booster")
tokenizer.push_to_hub(f"{hf_username}/mi300x-reasoning-booster")

# Push LoRA adapter
print("Pushing LoRA adapter...")
model.push_to_hub(f"{hf_username}/mi300x-reasoning-booster-lora")

# Push dataset
print("Pushing dataset...")
api.upload_folder(
    folder_path="./data/final",
    repo_id=f"{hf_username}/reasoning-booster-synthetic-data",
    repo_type="dataset"
)

print("All artifacts pushed to Hugging Face!")
PY

echo ""
echo "=========================================="
echo " Evaluation and shipping complete!"
echo " Results saved to results/"
echo " Next: Create Gradio demo and update GitHub"
echo "=========================================="
