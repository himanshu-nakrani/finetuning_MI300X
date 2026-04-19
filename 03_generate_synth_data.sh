#!/bin/bash
# 03_generate_synth_data.sh - Generate synthetic reasoning data
# Run this on the Digital Ocean MI300X droplet AFTER vLLM server is running

set -euo pipefail

cd /scratch/reasoning_booster
source .venv/bin/activate

echo "=========================================="
echo " Synthetic Data Generation"
echo "=========================================="

# Create directories
mkdir -p data/input data/parsed data/generated data/curated data/final

echo "[1/4] Downloading seed data (GSM8K)..."
wget -O data/input/gsm8k_train.jsonl https://huggingface.co/datasets/gsm8k/resolve/main/main/train.jsonl

echo "[2/4] Ingesting and parsing data..."
synthetic-data-kit ingest data/input --verbose
synthetic-data-kit -c config.yaml create data/parsed --type cot --num-pairs 20 --verbose

echo "[3/4] Generating synthetic examples (this will take 4-5 hours)..."
synthetic-data-kit -c config.yaml generate data/generated --num-samples 5000 --verbose

echo "[4/4] Curating high-quality examples (this will take 1-2 hours)..."
synthetic-data-kit -c config.yaml curate data/generated --threshold 7.0 --verbose

echo "[5/5] Saving in fine-tuning format..."
synthetic-data-kit save-as data/curated --format ft --output data/final --verbose

echo ""
echo "=========================================="
echo " Synthetic data generation complete!"
echo " Output: data/final/"
echo " Next: Run 04_train_sft.py"
echo "=========================================="
