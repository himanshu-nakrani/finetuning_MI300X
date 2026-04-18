# MI300X 70B Fine-Tuning Bootcamp

50 GPU-hours on a single AMD Instinct MI300X (192 GB). 8 shippable models covering SFT → DPO → GRPO, tool-calling, and vision LoRA on Llama-3.1/3.2.

**Start here:** `[PLAN.md](./PLAN.md)` — the full end-to-end plan (budget, projects, data, eval, risks, deliverables).

## Quick start

```bash
# 1. Install ROCm torch, then core deps (vLLM + Gradio are separate files — see requirements.txt)
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch
pip install -r requirements.txt
# Optional: pip install -r requirements-vllm.txt && pip install -r requirements-demo.txt
# (or: pip install "unsloth[rocm]" then -r requirements.txt)

# 2. Auth
huggingface-cli login
wandb login

# 3. Preflight (env sanity + 100-step smoke train, ~30 min, ~0.5 GPU-hr)
export SCRATCH=/scratch/finetune
bash scripts/00_preflight.sh
```

If preflight prints `PREFLIGHT PASSED`, proceed to the start sequence in `PLAN.md` §17.

## Layout

See `PLAN.md` §14. Top-level:

- `configs/` — YAML configs per project
- `scripts/` — runnable launchers (`00_preflight.sh` → `12_serve_bench.py`)
- `data/` — dedup + decontamination pipeline
- `eval/` — benchmark runners
- `rewards/` — GRPO reward functions + unit tests
- `results/` — benchmark tables, ablation reports
- `demo/` — Gradio app

## Status

- Plan drafted (`PLAN.md`)
- Preflight script (`scripts/00_preflight.sh`, `scripts/_smoke_train.py`)
- **Day 1 ready:**
  - `data/prepare_base_mix.py` (+ `dedup.py`, `decontaminate.py`)
  - `scripts/01_8b_qlora.py` + `configs/p1_8b_qlora.yaml` — P1 (8B QLoRA warm-up)
  - `scripts/02_70b_sft.py` + `configs/p2_70b_sft.yaml` — P2 (70B LoRA SFT)
- P3 — Reasoning-Booster SFT
- P4 — DPO
- P5 — GRPO (headline)
- P6 — LoRA rank ablation
- P7 — Tool-calling / JSON
- P8 — Vision LoRA
- Quantization + serving bench + demo

