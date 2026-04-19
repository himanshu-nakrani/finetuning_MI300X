"""
run_benchmarks.py — evaluate a model (+ optional adapter) on GSM8K-250 and
HumanEval, write a single-row benchmark report to results/.

Single entry-point by design: takes one --model (local path or HF id) and an
optional --adapter, runs both benchmarks with identical decoding params, and
appends a markdown row to results/benchmarks.md.

For Project A, we run it 4 times (base, P2, P3, P4) to produce the headline
benchmark matrix. For Project B, we run it on each ablation run.

Usage:
  python eval/run_benchmarks.py \
      --model   Qwen/Qwen2.5-72B-Instruct \
      --name    baseline

  python eval/run_benchmarks.py \
      --model   Qwen/Qwen2.5-72B-Instruct \
      --adapter /scratch/finetune/outputs/p4_dpo/adapter \
      --name    p4_dpo

Outputs:
  results/eval_<name>_gsm8k.jsonl      # per-example predictions
  results/eval_<name>_humaneval.jsonl
  results/benchmarks.md                # appended row

Budget: ~1 hr per 72B model eval at batch_size=8 (GSM8K-250 + HumanEval-164).
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from _common import setup_env_dirs  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model id or local path (merged model)")
    ap.add_argument("--adapter", default=None,
                    help="Optional PEFT adapter to apply on top of --model")
    ap.add_argument("--name", required=True,
                    help="Short name for this eval (e.g. 'p4_dpo')")
    ap.add_argument("--gsm8k_n", type=int, default=250,
                    help="Number of GSM8K-test questions to evaluate on")
    ap.add_argument("--humaneval", action="store_true", default=True,
                    help="Run HumanEval (default: on; use --no-humaneval to skip)")
    ap.add_argument("--no-humaneval", dest="humaneval", action="store_false")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# -----------------------------------------------------------------------------
# Model loader (handles merged-model, base+adapter, and HF-id cases)
# -----------------------------------------------------------------------------
def load_model(model_path: str, adapter_path: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] base = {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    m = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True,
    )
    if adapter_path:
        from peft import PeftModel
        print(f"[load] adapter = {adapter_path}")
        m = PeftModel.from_pretrained(m, adapter_path)
    m.eval()
    return m, tok


def generate_batch(model, tok, prompts: list[str], max_new_tokens: int,
                   max_model_len: int = 2048) -> list[str]:
    import torch
    rendered = [tok.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tok(rendered, return_tensors="pt", padding=True,
              truncation=True, max_length=max_model_len - max_new_tokens)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    in_len = enc["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # deterministic for eval
            temperature=1.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.batch_decode(out[:, in_len:], skip_special_tokens=True)


# -----------------------------------------------------------------------------
# GSM8K
# -----------------------------------------------------------------------------
_GSM_ANS_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def gsm8k_extract_answer(text: str) -> str | None:
    m = list(_GSM_ANS_RE.finditer(text[-500:]))
    if not m:
        return None
    return m[-1].group(0).replace(",", "").rstrip(".0") or "0"


def run_gsm8k(model, tok, n: int, batch_size: int, max_new_tokens: int,
              out_jsonl: Path):
    from datasets import load_dataset
    import random as _r
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = _r.Random(42)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    examples = [ds[i] for i in idx]

    correct = 0
    t0 = time.time()
    with out_jsonl.open("w") as fout:
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            prompts = [
                ex["question"].strip() + "\n\nSolve step-by-step and put the "
                "final numeric answer on a new line after '####'."
                for ex in batch
            ]
            outs = generate_batch(model, tok, prompts, max_new_tokens)
            for ex, out in zip(batch, outs):
                gt = ex["answer"].split("####")[-1].strip().replace(",", "") \
                    .rstrip(".0") or "0"
                pred = gsm8k_extract_answer(out) or ""
                ok = pred == gt
                correct += int(ok)
                fout.write(json.dumps({
                    "question": ex["question"], "gold": gt,
                    "pred": pred, "correct": ok, "response": out,
                }) + "\n")
            done = min(i + batch_size, len(examples))
            acc = correct / max(done, 1)
            rate = done / max(time.time() - t0, 1)
            print(f"[gsm8k] {done}/{len(examples)}  acc={acc:.3f}  "
                  f"{rate:.2f}/s", flush=True)

    acc = correct / len(examples)
    print(f"[gsm8k] FINAL {correct}/{len(examples)} = {acc:.3f}")
    return acc, len(examples)


# -----------------------------------------------------------------------------
# HumanEval
# -----------------------------------------------------------------------------
def _run_code(code: str, test: str, entry_point: str, timeout_s: int = 10) -> bool:
    src = (code + "\n" + test + "\n" + f"check({entry_point})\n")
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
            tf.write(src); tf.flush()
            r = subprocess.run(
                [sys.executable, tf.name],
                capture_output=True, timeout=timeout_s,
            )
        return r.returncode == 0
    except Exception:
        return False


_FENCED_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_python(text: str) -> str:
    m = _FENCED_RE.search(text)
    return (m.group(1) if m else text).strip()


def run_humaneval(model, tok, batch_size: int, max_new_tokens: int,
                  out_jsonl: Path):
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    passed = 0
    t0 = time.time()
    with out_jsonl.open("w") as fout:
        for i in range(0, len(ds), batch_size):
            batch = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
            prompts = [
                "Complete the following Python function. Only return the "
                "function body and any needed imports in a fenced code block.\n\n"
                + ex["prompt"]
                for ex in batch
            ]
            outs = generate_batch(model, tok, prompts, max_new_tokens)
            for ex, out in zip(batch, outs):
                code = extract_python(out)
                # HumanEval wants ex["prompt"] + candidate, then ex["test"]
                full_code = ex["prompt"] + "\n" + code
                ok = _run_code(full_code, ex["test"], ex["entry_point"])
                passed += int(ok)
                fout.write(json.dumps({
                    "task_id": ex["task_id"], "pass": ok,
                    "completion": code,
                }) + "\n")
            done = min(i + batch_size, len(ds))
            acc = passed / max(done, 1)
            rate = done / max(time.time() - t0, 1)
            print(f"[humaneval] {done}/{len(ds)}  pass@1={acc:.3f}  "
                  f"{rate:.2f}/s", flush=True)

    acc = passed / len(ds)
    print(f"[humaneval] FINAL {passed}/{len(ds)} = {acc:.3f}")
    return acc, len(ds)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def append_row(name: str, model: str, adapter: str | None,
               gsm8k: tuple[float, int] | None,
               humaneval: tuple[float, int] | None):
    md = RESULTS_DIR / "benchmarks.md"
    if not md.exists():
        md.write_text(
            "# Benchmark matrix\n\n"
            "| run | base model | adapter | GSM8K | HumanEval pass@1 | timestamp |\n"
            "|---|---|---|---|---|---|\n"
        )
    g = f"{gsm8k[0]:.3f} (n={gsm8k[1]})" if gsm8k else "—"
    h = f"{humaneval[0]:.3f} (n={humaneval[1]})" if humaneval else "—"
    row = (f"| `{name}` | `{model}` | "
           f"`{adapter or '—'}` | {g} | {h} | "
           f"{time.strftime('%Y-%m-%d %H:%M')} |\n")
    with md.open("a") as f:
        f.write(row)
    print(f"[results] appended to {md}")


def main() -> None:
    args = parse_args()
    setup_env_dirs()
    model, tok = load_model(args.model, args.adapter)

    gsm_result = None
    he_result = None

    print("\n=== GSM8K ===")
    gsm_result = run_gsm8k(
        model, tok, n=args.gsm8k_n, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        out_jsonl=RESULTS_DIR / f"eval_{args.name}_gsm8k.jsonl",
    )

    if args.humaneval:
        print("\n=== HumanEval ===")
        he_result = run_humaneval(
            model, tok, batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            out_jsonl=RESULTS_DIR / f"eval_{args.name}_humaneval.jsonl",
        )

    append_row(args.name, args.model, args.adapter, gsm_result, he_result)
    print("[done]")


if __name__ == "__main__":
    main()
