"""
decontaminate.py — n-gram overlap removal vs benchmark test sets.

Drops any training record whose text shares an n-gram (default 13) with any
prompt or answer in GSM8K-test, HumanEval, or MATH-test. This is the standard
"benchmark contamination" check (Llama-3 / Tulu-3 style).

Usage:
  python data/decontaminate.py \
      --input  /scratch/finetune/data/raw_tulu3_dedup.jsonl \
      --output /scratch/finetune/data/raw_tulu3_clean.jsonl \
      --benchmarks gsm8k humaneval math \
      --n 13
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _try_load(candidates: list[tuple[str, ...]]):
    """Try (path, *args) tuples in order; return the first that loads, else None."""
    last_err = None
    for spec in candidates:
        path, *rest = spec
        try:
            ds = load_dataset(path, *rest, split="test")
            print(f"[decon]   loaded: {path} {' '.join(rest)}")
            return ds
        except Exception as e:
            last_err = e
            print(f"[decon]   miss:   {path} {' '.join(rest)}  ({type(e).__name__})")
    print(f"[decon]   ALL CANDIDATES FAILED. last error: {last_err}")
    return None


def collect_benchmark_ngrams(names: list[str], n: int) -> set[tuple[str, ...]]:
    grams: set[tuple[str, ...]] = set()
    skipped: list[str] = []

    for name in names:
        if name == "gsm8k":
            ds = _try_load([
                ("openai/gsm8k", "main"),
                ("gsm8k", "main"),
            ])
            if ds is None:
                skipped.append(name); continue
            for ex in ds:
                grams |= ngrams(tokenize(ex["question"] + " " + ex["answer"]), n)

        elif name == "humaneval":
            ds = _try_load([
                ("openai_humaneval",),
                ("openai/openai_humaneval",),
            ])
            if ds is None:
                skipped.append(name); continue
            for ex in ds:
                grams |= ngrams(
                    tokenize(ex["prompt"] + " " + ex.get("canonical_solution", "")), n
                )

        elif name == "math":
            # Hendrycks MATH has bounced around HF. Try the live mirrors in order.
            ds = _try_load([
                ("HuggingFaceH4/MATH-500",),         # 500-sample eval slice (still useful for decon)
                ("qwedsacf/competition_math",),      # community mirror of full MATH
                ("EleutherAI/hendrycks_math", "all"),
                ("lighteval/MATH",),
                ("hendrycks/competition_math",),     # original (often gated/dead)
            ])
            if ds is None:
                skipped.append(name); continue
            for ex in ds:
                # Field names differ across mirrors; try common ones.
                problem = ex.get("problem") or ex.get("question") or ""
                solution = ex.get("solution") or ex.get("answer") or ""
                grams |= ngrams(tokenize(problem + " " + solution), n)

        else:
            raise ValueError(f"unknown benchmark: {name}")

        print(f"[decon] {name}: total grams so far = {len(grams):,}")

    if skipped:
        print(f"[decon] WARNING: skipped {skipped} (dataset unavailable). "
              f"Decontamination still applied for the rest.")
    return grams


def record_text(rec: dict) -> str:
    if "text" in rec and isinstance(rec["text"], str):
        return rec["text"]
    msgs = rec.get("messages") or []
    return "\n".join(str(m.get("content", "")) for m in msgs if isinstance(m, dict))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--benchmarks",
        nargs="+",
        default=["gsm8k", "humaneval", "math"],
        choices=["gsm8k", "humaneval", "math"],
    )
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--report_every", type=int, default=5000)
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"[decon] building n-gram set for: {args.benchmarks} (n={args.n})")
    bench = collect_benchmark_ngrams(args.benchmarks, args.n)
    print(f"[decon] benchmark n-grams: {len(bench):,}")
    if not bench:
        raise SystemExit(
            "[decon] FATAL: 0 benchmark n-grams collected — none of the requested "
            "benchmarks could be loaded. Re-run with a smaller --benchmarks list "
            "(e.g. --benchmarks gsm8k humaneval) or copy the dataset locally."
        )

    kept = dropped = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            toks = tokenize(record_text(rec))
            grams = ngrams(toks, args.n)
            if grams & bench:
                dropped += 1
            else:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            if (i + 1) % args.report_every == 0:
                print(
                    f"[decon] read={i + 1:,} kept={kept:,} dropped={dropped:,} "
                    f"({dropped / (i + 1):.2%})"
                )

    print(f"[decon] FINAL  kept={kept:,}  dropped={dropped:,}  → {args.output}")


if __name__ == "__main__":
    main()
