# MI300X 50-Hour Fine-Tuning Bootcamp — End-to-End Plan

**Hardware:** AMD Instinct MI300X (192 GB VRAM), 5 TB NVMe scratch
**Window:** 5 calendar days, 50 GPU-hour credits (hard deadline)
**Wall-clock:** 5 active days (~10 hrs/day), **zero buffer** — every hour is allocated
**Date authored:** April 2026
**Goal:** Ship 8 fine-tuned models + full pipeline + portfolio in exactly 50 GPU-hours across 5 calendar days, covering ~85% of what the April-2026 applied fine-tuning job market asks for.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Market Context — What "Enough" Means in April 2026](#2-market-context)
3. [Skills Coverage Matrix](#3-skills-coverage-matrix)
4. [GPU Budget Ledger (50.0 hrs, line-by-line)](#4-gpu-budget-ledger)
5. [Day-by-Day Execution Plan](#5-day-by-day-execution-plan)
6. [Project Specifications (P1–P8)](#6-project-specifications)
7. [Data Pipeline Specifications](#7-data-pipeline-specifications)
8. [Evaluation Protocol](#8-evaluation-protocol)
9. [Concurrency Rules (Why This Fits in 50 hrs)](#9-concurrency-rules)
10. [Risk Register & Overrun/Under-run Playbook](#10-risk-register)
11. [Parallel Reading List](#11-parallel-reading-list)
12. [Deliverables & Portfolio Artifacts](#12-deliverables)
13. [Resume / LinkedIn Copy](#13-resume-copy)
14. [Repository Layout](#14-repository-layout)
15. [Environment Setup (Copy-Paste)](#15-environment-setup)
16. [What You Will NOT Learn (Be Honest)](#16-what-you-wont-learn)
17. [Start Sequence](#17-start-sequence)

---

## 1. Executive Summary

The industry standard for applied LLM fine-tuning in April 2026 is:

> **Synthetic data → SFT (LoRA/QLoRA) → DPO → GRPO (RLVR)** on Llama-3.1/3.3 70B, using **Unsloth + TRL + vLLM** on ROCm or CUDA.

Most engineers never touch 70B. A single MI300X with 192 GB VRAM lets you do it solo — batch sizes that crush H100s in memory-bound regimes, no multi-node headaches.

**This plan burns all 50 GPU-hours** across 8 shippable projects, overlapping reading/data-gen with training so wall-clock stays inside 5 active days.

---

## 2. Market Context

**Hot in April 2026:**

- GRPO / RLVR (DeepSeek-R1 lineage) — verifiable-reward RL.
- Tool-calling + JSON schema adherence fine-tunes.
- FP8 quantization + vLLM LoRA hot-swap serving.
- Synthetic data with dedup + benchmark decontamination.
- Vision-language LoRA on Llama-3.2-Vision / Qwen-VL.

**Cooling / commodity:**

- Plain Alpaca-style SFT on 7B (table stakes).
- Classic RLHF with separately-trained reward models (DPO/GRPO replaced most of it).

**Required tool fluency:**

- Unsloth, TRL (SFT/DPO/GRPO Trainer), PEFT, vLLM, SGLang (awareness), llm-compressor, datasets, wandb, HF Hub.

---

## 3. Skills Coverage Matrix


| Skill                                            | Covered In         | Depth      |
| ------------------------------------------------ | ------------------ | ---------- |
| LoRA / QLoRA / DoRA                              | P1, P2, P6         | Hands-on   |
| Full SFT                                         | P2, P3, P7         | Hands-on   |
| Chat templates + loss masking                    | Day 1 reading + P1 | Hands-on   |
| Synthetic data gen (self-instruct / evol)        | Day 2              | Hands-on   |
| Dedup (MinHash) + decontamination                | Day 2              | Hands-on   |
| DPO                                              | P4                 | Hands-on   |
| GRPO / RLVR                                      | P5                 | Hands-on   |
| Verifiable rewards (math/code)                   | P5                 | Hands-on   |
| LLM-as-judge eval                                | Day 3              | Hands-on   |
| Benchmarks: GSM8K, HumanEval, MATH-500, MT-Bench | Days 2–4           | Hands-on   |
| Tool-calling / JSON schema                       | P7                 | Hands-on   |
| Vision-language LoRA                             | P8                 | Hands-on   |
| Quantization: FP8, GGUF Q4_K_M                   | Day 5              | Hands-on   |
| vLLM serving + LoRA hot-swap                     | Day 5              | Hands-on   |
| Ablation methodology                             | P6                 | Hands-on   |
| FSDP / ZeRO / TP / PP                            | Reading only       | Conceptual |
| RoPE scaling (YaRN/NTK)                          | Reading only       | Conceptual |
| Reward modeling (Bradley-Terry)                  | Reading only       | Conceptual |
| MoE architectures                                | Reading only       | Conceptual |
| Licensing / PII / safety                         | Day 5              | Conceptual |


---

## 4. GPU Budget Ledger

Target: **50.0 GPU-hours**.


| #   | Run                                                        | GPU hrs | Cum  |
| --- | ---------------------------------------------------------- | ------- | ---- |
| 0   | Pre-flight sanity + smoke train                            | 0.5     | 0.5  |
| 1   | **P1:** Llama-3.1-8B QLoRA on Alpaca                       | 1.5     | 2.0  |
| 2   | **P2:** Llama-3.1-70B LoRA SFT on 20k instruct mix         | 6.0     | 8.0  |
| 3   | Synthetic data gen #1 (15k reasoning/code pairs via vLLM)  | 2.5     | 10.5 |
| 4   | **P3:** 70B SFT on synthetic reasoning (Reasoning-Booster) | 6.5     | 17.0 |
| 5   | Preference-pair gen (2 completions × 5k prompts + judge)   | 2.5     | 19.5 |
| 6   | **P4:** 70B DPO on Reasoning-Booster                       | 5.5     | 25.0 |
| 7   | Eval sweep A (P1–P4 full matrix)                           | 1.5     | 26.5 |
| 8   | GRPO dry-run (reward fn, 200 steps)                        | 1.0     | 27.5 |
| 9   | **P5:** 70B GRPO (RLVR), ~1.7k steps                       | 9.0     | 36.5 |
| 10  | Eval sweep B (P5 + ablations)                              | 1.5     | 38.0 |
| 11  | **P6:** Ablation — LoRA rank 64 vs 32 (1 epoch)            | 3.0     | 41.0 |
| 12  | **P7:** Tool-calling / JSON SFT LoRA on 70B                | 4.0     | 45.0 |
| 13  | **P8:** Llama-3.2-11B-Vision LoRA                          | 3.0     | 48.0 |
| 14  | Quantization (FP8 + GGUF for P4/P5/P7)                     | 1.0     | 49.0 |
| 15  | vLLM serving benchmarks + LoRA hot-swap                    | 1.0     | 50.0 |


---

## 5. Day-by-Day Execution Plan

### Day 1 — Foundation + 70B First Contact (GPU 8.0 hrs)

- 0.5 pre-flight
- 1.5 P1 (8B warm-up)
- 6.0 P2 (70B LoRA SFT)
- **Reading:** chat templates, QLoRA paper, Unsloth AMD docs, FSDP blog
- **Ship:** HF repos #1 (8B), #2 (70B-base-SFT)

### Day 2 — Synthetic Data + Reasoning-Booster (GPU 9.0 hrs)

- 2.5 synth gen #1
- 6.5 P3 (70B SFT on synth)
- **Reading:** Tulu 3 data section, Llama 3 post-training, RoPE scaling
- **Ship:** synth dataset on HF, HF repo #3 (Reasoning-Booster-SFT)

### Day 3 — DPO + Eval Infra (GPU 9.5 hrs)

- 2.5 preference-pair gen
- 5.5 P4 (DPO)
- 1.5 Eval sweep A
- **Reading:** DPO paper §4, reward modeling basics, LLM-as-judge
- **Ship:** HF repo #4 (Reasoning-Booster-DPO), `benchmarks.md` v1

### Day 4 — GRPO Main Event (GPU 11.5 hrs)

- 1.0 GRPO dry-run
- 9.0 P5 GRPO
- 1.5 Eval sweep B
- **Reading:** DeepSeek-R1, reward hacking, KL control
- **Ship:** HF repo #5 (Reasoning-Booster-GRPO) — headline model

### Day 5 — Breadth + Productionize (GPU 12.0 hrs)

- 3.0 P6 ablation
- 4.0 P7 tool-calling/JSON
- 3.0 P8 vision LoRA
- 1.0 quantization
- 1.0 vLLM serving benchmarks
- **Non-GPU (done in parallel with training):** Gradio app, GitHub polish, LinkedIn post, HF collection
- **Ship:** HF repos #6, #7, #8 + demo Space + GitHub repo + benchmark report

---

## 6. Project Specifications

### P1 — Llama-3.1-8B QLoRA Warm-up

- **Base:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Data:** Alpaca-cleaned (~50k)
- **Config:** QLoRA NF4, rank=16, alpha=32, dropout=0.05, 3 epochs, bs=16, LR 2e-4 cosine
- **Tool:** Unsloth
- **Eval:** MMLU subset (500), 50-prompt vibe check
- **Output:** Merged HF repo + adapter repo

### P2 — Llama-3.1-70B LoRA SFT (Base Mix)

- **Base:** `meta-llama/Meta-Llama-3.1-70B-Instruct`
- **Data:** 20k curated from Tulu-3 mix (chat + reasoning + code), deduped, decontaminated against GSM8K/HumanEval/MATH
- **Config:** LoRA rank=32, alpha=64, all-linear targets, bs=64 grad_accum=2, 2 epochs, LR 2e-4 cosine, bf16, grad checkpointing
- **Tool:** Unsloth
- **Save:** every 30 min

### P3 — Reasoning-Booster SFT

- **Base:** P2 merged model (or Instruct + P2 adapter)
- **Data:** 15k synthetic reasoning/code pairs (see §7)
- **Config:** rank=32, 3 epochs, LR 2e-4 → 2e-5, bs=32 grad_accum=4
- **Eval:** GSM8K-250, HumanEval, MATH-500 subset

### P4 — Reasoning-Booster DPO

- **Base:** P3 adapter (continue LoRA, don't restart)
- **Data:** 5k preference pairs (see §7)
- **Tool:** TRL `DPOTrainer`
- **Config:** beta=0.1, LR 5e-7, 1 epoch, bs=8 grad_accum=8

### P5 — Reasoning-Booster GRPO (Headline)

- **Base:** P4 adapter
- **Data:** 2–3k prompts with verifiable answers (GSM8K-train, MATH, MBPP slice)
- **Tool:** TRL `GRPOTrainer` or Unsloth GRPO
- **Config:** group_size=8, LR 1e-6, max_new_tokens=1024, KL beta tuned in dry-run
- **Reward:** exact-match (math), unit-test pass (code), format tags (`<think>…</think><answer>…</answer>`)
- **Steps:** ~1700 (9.0 hr hard cap)

### P6 — LoRA Rank Ablation

- **Base:** Instruct 70B
- **Data:** same 20k as P2
- **Runs:** rank=64 (1 epoch) — compare vs P2's rank=32
- **Output:** `results/ablation_rank.md`

### P7 — Tool-Calling / JSON Schema LoRA

- **Base:** Instruct 70B
- **Data:** Glaive-function-calling + 2k synthetic tool-call traces
- **Config:** rank=32, 2 epochs, LR 1e-4, bs=32
- **Eval:** JSON schema adherence %, argument correctness, hallucinated-tool rate

### P8 — Vision-Language LoRA

- **Base:** `meta-llama/Llama-3.2-11B-Vision-Instruct`
- **Data:** 5k from LLaVA-Instruct-150k subset
- **Config:** LoRA rank=16 on language tower, vision encoder frozen, 1 epoch
- **Eval:** 50 hand-picked VQA prompts, before/after side-by-side

---

## 7. Data Pipeline Specifications

### 7.1 Base Mix (P2)

- Source: Tulu-3-SFT-mix
- Filter: length 64–4096 tokens, English-only, deduped (MinHash Jaccard ≥ 0.8 dropped)
- Decontaminate: n-gram overlap against GSM8K/HumanEval/MATH test sets
- Target: 20k examples

### 7.2 Synthetic Reasoning/Code (P3)

- **Engine:** vLLM serving merged P2 at 0.0.0.0:8000, tensor_parallel=1
- **Seeds:** GSM8K-train prompts, MBPP-train, evol-instruct rewrite prompts
- **Generation:** temp 0.7, top_p 0.95, 2 completions/prompt, reject short or non-compliant
- **Post:** MinHash dedup, length filter, target 15k final

### 7.3 Preference Pairs (P4)

- **Prompts:** 5k unseen reasoning/code prompts
- **Completions:** 2 per prompt at temp 0.9 from P3
- **Ranking:**
  - Math → ground-truth exact-match (chosen = correct, rejected = incorrect)
  - Code → unit-test pass/fail
  - General → Llama-3.1-70B-Instruct as judge (pairwise)
- **Output:** JSONL `{prompt, chosen, rejected}`

### 7.4 GRPO Prompts (P5)

- **Source:** GSM8K-train (1k), MATH (1k), MBPP-train (500)
- **Format:** system prompt enforces `<think>…</think><answer>…</answer>` schema
- **Reward fn:** unit-tested on 50 manual cases before launch (CPU only, 2 hrs Day 4)

---

## 8. Evaluation Protocol

**Benchmarks run on every model (P1–P8 where applicable):**


| Benchmark             | Split                                         | Used For            |
| --------------------- | --------------------------------------------- | ------------------- |
| GSM8K                 | test (250-sample + full on headline)          | Math reasoning      |
| HumanEval             | full, pass@1                                  | Code                |
| MATH-500              | full                                          | Hard math           |
| MT-Bench              | full (judge: GPT-4 or Llama-3.1-70B-Instruct) | Chat quality        |
| AlpacaEval 2 (subset) | 200 prompts                                   | Pairwise preference |
| Custom regression     | 50 hand-written prompts                       | Domain sanity       |
| Tool-call schema (P7) | 200 synthetic tool prompts                    | JSON validity %     |
| VQA subset (P8)       | 50 prompts                                    | Vision              |


**Output:** `results/benchmarks.md` with a single matrix table: rows = models, columns = benchmarks.

---

## 9. Concurrency Rules

Three rules keep GPU utilization >95%:

1. **Training ↔ Reading overlap.** All ~8 hrs of papers/docs happen during training runs. Zero wall-clock "studying."
2. **Data gen on same GPU between trainings.** vLLM cold-start <2 min on MI300X; not a separate resource.
3. **Batched eval.** Two sweeps total (A after DPO, B after GRPO) — pay startup cost twice, not 8 times.

---

## 10. Risk Register


| Risk                       | Prob | Impact | Mitigation                                                            |
| -------------------------- | ---- | ------ | --------------------------------------------------------------------- |
| ROCm env break             | Med  | High   | Pre-flight script at hr 0; pin Unsloth+ROCm versions                  |
| GRPO reward curve flat     | Med  | High   | Dry-run 200 steps first; 9.0 hr hard cap on P5; kill-and-restart rule |
| OOM at 70B bs=64           | Low  | Med    | Grad checkpointing on, bs fallback 32                                 |
| Synthetic data low-quality | Med  | Med    | MinHash dedup + length filter + 100-sample manual review before P3    |
| HF push fails              | Low  | Low    | Local backup on `/scratch/` every save                                |
| Run overrun                | Med  | High   | **No reserve** — cut P8 first, then P6, then shrink P5 to 6.5 hrs     |


**Under-run additions (if fast):**

- P9: self-play preference gen (P5 vs P4) → retrain DPO, 3–4 hrs
- P10: speculative decoding draft model (1B) trained on synth, 2 hrs

---

## 11. Parallel Reading List

Read *while GPU burns*. Total ~8 hrs.

1. Unsloth AMD/MI300X docs — 1 hr (Day 1)
2. TRL docs: SFT/DPO/GRPO Trainer — 2 hrs (Days 1–4, chunked)
3. Llama 3 paper (data + post-training only) — 1 hr (Day 2)
4. Tulu 3 paper (full post-training) — 1.5 hrs (Day 2)
5. DeepSeek-R1 paper — 1 hr (Day 4)
6. DPO paper §4 — 0.5 hr (Day 3)
7. QLoRA paper (NF4 + double quant) — 0.5 hr (Day 1)
8. PyTorch FSDP official blog — 0.5 hr (Day 1)

---

## 12. Deliverables

**8 HF model repos (all with model cards, configs, wandb links):**

1. `llama-3.1-8b-alpaca-qlora`
2. `llama-3.1-70b-instruct-mix-sft`
3. `llama-3.1-70b-reasoning-booster-sft`
4. `llama-3.1-70b-reasoning-booster-dpo`
5. `llama-3.1-70b-reasoning-booster-grpo` ← headline
6. `llama-3.1-70b-toolcalling-json`
7. `llama-3.2-11b-vision-lora`
8. Quantized variants repo (FP8 + GGUF)

**2 HF datasets:** synthetic reasoning (15k), preference pairs (5k).

**1 HF collection** grouping all above.

**1 GitHub repo** `MI300X-70B-Fine-Tuning-Bootcamp` (see §14).

**1 Gradio demo** (local + Space config, even if Space runs cpu-only as reference).

**1 LinkedIn post** with benchmark matrix as hook.

---

## 13. Resume / LinkedIn Copy

**Resume bullets:**

- *Fine-tuned Llama-3.1-70B on a single AMD MI300X (192 GB) across SFT → DPO → GRPO, improving GSM8K by X% and HumanEval pass@1 by Y% over the Instruct baseline.*
- *Built end-to-end synthetic data pipeline (vLLM self-instruct + MinHash dedup + benchmark decontamination) producing 15k reasoning pairs and 5k preference pairs.*
- *Implemented RLVR with verifiable math/code rewards using TRL `GRPOTrainer`; ablated KL-beta, group size, and LoRA rank.*
- *Shipped FP8 and GGUF-Q4_K_M quantized variants served via vLLM with hot-swappable LoRA adapters; measured throughput at Z tok/s.*
- *Extended pipeline to tool-calling (JSON schema adherence) and vision-language (Llama-3.2-11B-Vision).*

**LinkedIn hook:** "I burned 50 GPU-hours on an AMD MI300X and came out with 8 fine-tuned models, a full synthetic→SFT→DPO→GRPO pipeline, and benchmarks that beat the Instruct baseline by X%/Y%/Z% on GSM8K/HumanEval/MATH-500. Full repo + configs + wandb links in comments."

---

## 14. Repository Layout

```
MI300X-70B-Fine-Tuning-Bootcamp/
├── README.md
├── PLAN.md                          # this file
├── configs/
│   ├── p1_8b_qlora.yaml
│   ├── p2_70b_sft.yaml
│   ├── p3_reasoning_sft.yaml
│   ├── p4_dpo.yaml
│   ├── p5_grpo.yaml
│   ├── p6_ablation_rank64.yaml
│   ├── p7_toolcalling.yaml
│   └── p8_vision.yaml
├── scripts/
│   ├── 00_preflight.sh
│   ├── 01_8b_qlora.py
│   ├── 02_70b_sft.py
│   ├── 03_synth_gen.py
│   ├── 04_reasoning_sft.py
│   ├── 05_pref_gen.py
│   ├── 06_dpo.py
│   ├── 07_grpo.py
│   ├── 08_ablation.py
│   ├── 09_toolcalling.py
│   ├── 10_vision.py
│   ├── 11_quantize.sh
│   └── 12_serve_bench.py
├── data/
│   ├── dedup.py                     # MinHash
│   ├── decontaminate.py             # n-gram overlap
│   ├── synth_seeds/
│   └── README.md
├── eval/
│   ├── run_gsm8k.py
│   ├── run_humaneval.py
│   ├── run_math500.py
│   ├── run_mtbench.py
│   ├── run_toolcall_schema.py
│   ├── run_vqa.py
│   └── regression_prompts.jsonl
├── rewards/
│   ├── math_reward.py
│   ├── code_reward.py
│   ├── format_reward.py
│   └── tests/
├── results/
│   ├── benchmarks.md
│   ├── ablation_rank.md
│   └── serving_bench.md
├── demo/
│   └── gradio_app.py
└── requirements.txt
```

---

## 15. Environment Setup

```bash
# Scratch dir
mkdir -p /scratch/finetune/{models,data,outputs,logs,cache}
export HF_HOME=/scratch/finetune/cache
export TRANSFORMERS_CACHE=/scratch/finetune/cache
export WANDB_DIR=/scratch/finetune/logs

# Deps (ROCm MI300X)
pip install "unsloth[rocm]" trl peft datasets accelerate bitsandbytes wandb vllm
pip install text-dedup datasketch llm-compressor
pip install lm-eval

# Auth
huggingface-cli login
wandb login

# Sanity
rocm-smi --showmeminfo vram
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
df -h /scratch
```

**Pre-flight smoke train (0.5 GPU-hr):** 100-step dummy LoRA on 8B with 500 Alpaca examples — catches ROCm/driver/HF/disk issues before committing real hours.

---

## 16. What You Will NOT Learn

Do **not** claim these on your resume — interviewers will catch it:

- Pre-training from scratch (needs multi-node, weeks).
- Real multi-node distributed debugging (NCCL hangs, topology).
- Production RLHF with a separately-trained reward model at scale (you did RLVR).
- TB-scale data pipelines (Ray/Spark).
- Custom CUDA/ROCm/Triton kernels.

This is acceptable. The role in demand is **"applied fine-tuning engineer,"** not pre-training researcher.

---

## 17. Start Sequence

Execute in this order. Each script is generated **after** the previous run completes so parameters can be tuned to actual results.

1. `00_preflight.sh` — 10-min env + smoke train (0.5 GPU-hr)
2. `01_8b_qlora.py` — P1 (1.5 GPU-hr)
3. `02_70b_sft.py` — P2 (6.0 GPU-hr)
4. `03_synth_gen.py` — synth data (2.5 GPU-hr)
5. `04_reasoning_sft.py` — P3 (6.5 GPU-hr)
6. `05_pref_gen.py` — preference pairs (2.5 GPU-hr)
7. `06_dpo.py` — P4 (5.5 GPU-hr)
8. Eval sweep A (1.5 GPU-hr)
9. `07_grpo.py` dry-run (1.0 GPU-hr)
10. `07_grpo.py` full (9.0 GPU-hr)
11. Eval sweep B (1.5 GPU-hr)
12. `08_ablation.py` — P6 (3.0 GPU-hr)
13. `09_toolcalling.py` — P7 (4.0 GPU-hr)
14. `10_vision.py` — P8 (3.0 GPU-hr)
15. `11_quantize.sh` (1.0 GPU-hr)
16. `12_serve_bench.py` (1.0 GPU-hr)
17. Portfolio polish — **non-GPU only**, done in parallel with training on Days 3–5.

**Total: 50.0 GPU-hours. No reserve.**

---

## Appendix A — Quick Reference Commands

```bash
# Monitor VRAM during training
watch -n 2 rocm-smi --showmeminfo vram

# Tail wandb-less logs
tail -f /scratch/finetune/logs/*.log

# Push adapter to HF
huggingface-cli upload <user>/<repo> /scratch/finetune/outputs/<run>/

# Merge LoRA adapter
python -c "from peft import PeftModel; from transformers import AutoModelForCausalLM; \
  m = AutoModelForCausalLM.from_pretrained('<base>', torch_dtype='bfloat16'); \
  m = PeftModel.from_pretrained(m, '<adapter>').merge_and_unload(); \
  m.save_pretrained('<out>')"

# Serve with vLLM + LoRA hot-swap
vllm serve <base> --enable-lora --lora-modules p4=/path/to/p4 p5=/path/to/p5

# Quick GSM8K eval via lm-eval
lm_eval --model hf --model_args pretrained=<model> --tasks gsm8k --batch_size 8
```

---

## Appendix B — Kill Switches

Stop and reassess if any of these trigger:

- **P5 GRPO:** reward curve flat for 300 consecutive steps → kill, lower LR 2×, raise KL beta to 0.2, restart once (within the 9.0 hr cap — no extension).
- **Any run:** VRAM > 185 GB → reduce bs by half, don't push it.
- **Synthetic data:** manual review of 100 samples shows > 20% garbage → regenerate with stricter prompts, don't train on it.
- **Overrun clock (no buffer):** if cumulative GPU-hrs exceed `planned + 5%` at end of any day → drop P8 immediately, then P6 at next checkpoint. No negotiation.

---

**End of plan.** Ready to execute. Reply `go` to start with `00_preflight.sh`.