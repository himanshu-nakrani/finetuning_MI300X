"""
_merge_adapter.py — merge a LoRA adapter into its base model and save fp16/bf16
weights ready for vLLM serving.

One-shot utility used at the start of Day 2 (merge P2 → serve with vLLM for
synthetic data gen) and again Day 5 (merge P5 for the final headline model
before quantization).

Usage:
  # Merge P2 (Day 2):
  python scripts/_merge_adapter.py \
      --base    Qwen/Qwen2.5-72B-Instruct \
      --adapter /scratch/finetune/outputs/p2_70b_sft/adapter \
      --output  /scratch/finetune/models/p2_merged \
      --dtype   bfloat16

  # Optionally also push to HF:
  python scripts/_merge_adapter.py ... --hub_repo_id <user>/qwen2.5-72b-mix-sft-merged
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _common import push_to_hub, setup_env_dirs  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF model id of base model")
    ap.add_argument("--adapter", required=True, help="Local path or HF id of adapter")
    ap.add_argument("--output", required=True, help="Where to save merged weights")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--device_map", default="auto",
                    help="'auto' (use GPU) or 'cpu' (slower, no VRAM, safer if GPU busy)")
    ap.add_argument("--hub_repo_id", default=None)
    ap.add_argument("--no_push", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    setup_env_dirs()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    t0 = time.time()
    print(f"[merge] base    = {args.base}")
    print(f"[merge] adapter = {args.adapter}")
    print(f"[merge] output  = {out}  ({args.dtype}, device_map={args.device_map})")

    print("[merge] loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(args.base)

    print("[merge] attaching adapter...")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("[merge] merge_and_unload (this is the slow step)...")
    model = model.merge_and_unload()

    print(f"[merge] saving merged weights → {out}")
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(str(out))

    dt = (time.time() - t0) / 60.0
    print(f"[merge] done in {dt:.1f} min")
    print(f"[merge] disk usage:")
    import subprocess
    subprocess.run(["du", "-sh", str(out)], check=False)

    if args.hub_repo_id and not args.no_push:
        print(f"[merge] pushing → {args.hub_repo_id}")
        push_to_hub(local_dir=out, repo_id=args.hub_repo_id,
                    commit_message="Merged adapter into base model")

    print("[merge] OK. To serve with vLLM:")
    print(f"  vllm serve {out} --max-model-len 4096 --dtype {args.dtype} \\")
    print(f"      --gpu-memory-utilization 0.92 --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
