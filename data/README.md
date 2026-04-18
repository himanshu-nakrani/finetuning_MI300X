# data/

Data prep utilities used across the bootcamp.

## Day 1 — base mix for P2 (70B SFT)

```bash
# One command, ~20-40 min CPU+net (no GPU):
python data/prepare_base_mix.py --target 20000

# Output:
#   /scratch/finetune/data/tulu3_filtered.jsonl
#   /scratch/finetune/data/tulu3_dedup.jsonl
#   /scratch/finetune/data/tulu3_clean.jsonl
#   /scratch/finetune/data/base_mix_20k.jsonl   <-- consumed by P2
```

Pipeline:

1. Stream `allenai/tulu-3-sft-mixture` from HF.
2. Length-filter (64..4096 token-equivalents) + cheap English heuristic.
3. **MinHash dedup** (`dedup.py`) at Jaccard 0.8 — drops near-duplicates.
4. **Decontamination** (`decontaminate.py`) — drop any record sharing a 13-gram
   with GSM8K-test, HumanEval, or MATH-test.
5. Reservoir-sample down to 20k.

Each step writes an intermediate JSONL so reruns can skip completed work.

## Standalone tools

```bash
python data/dedup.py         --input X.jsonl --output Y.jsonl --threshold 0.8
python data/decontaminate.py --input X.jsonl --output Y.jsonl --n 13
```

Both expect/emit `{"messages": [{"role": ..., "content": ...}, ...]}` per line
(or `{"text": "..."}` as a fallback).
