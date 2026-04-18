"""
prepare_base_mix.py — build the 20k curated base mix for P2 (70B SFT, Day 1).

Pipeline:
  1. Pull Tulu-3-SFT-mix from HF.
  2. Length-filter (64..max_tokens, English-only by simple heuristic).
  3. Write a "filtered" JSONL of {"messages": [...]} records.
  4. Run MinHash dedup (data/dedup.py) → "_dedup.jsonl".
  5. Run benchmark decontamination (data/decontaminate.py) → "_clean.jsonl".
  6. Sample/truncate to --target (default 20000) → final base_mix_20k.jsonl.

Each step writes intermediate files so reruns can skip completed work.

Usage:
  python data/prepare_base_mix.py --target 20000
  # or, fully custom paths:
  python data/prepare_base_mix.py \
      --workdir /scratch/finetune/data \
      --source allenai/tulu-3-sft-mixture \
      --target 20000
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

THIS_DIR = Path(__file__).resolve().parent

# --- token-cheap length proxy: ~0.75 tokens per word for English ---
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def approx_token_len(text: str) -> int:
    return int(len(_WORD_RE.findall(text)) / 0.75)


def looks_english(text: str, sample_chars: int = 500) -> bool:
    """Cheap heuristic: >=85% of letters in the first N chars are ASCII."""
    sample = text[:sample_chars]
    letters = [c for c in sample if c.isalpha()]
    if len(letters) < 20:
        return False
    ascii_ratio = sum(1 for c in letters if ord(c) < 128) / len(letters)
    return ascii_ratio >= 0.85


def normalize_record(rec: dict) -> dict | None:
    """Coerce any Tulu-style record to {"messages": [...]} only."""
    msgs = rec.get("messages")
    if not msgs:
        return None
    norm = []
    for m in msgs:
        if not isinstance(m, dict):
            return None
        role = m.get("role")
        content = m.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            return None
        if not isinstance(content, str) or not content.strip():
            return None
        norm.append({"role": role, "content": content})
    if not any(m["role"] == "assistant" for m in norm):
        return None
    return {"messages": norm}


def stream_filter(
    source: str,
    out_path: Path,
    *,
    min_tokens: int,
    max_tokens: int,
    english_only: bool,
    max_records: int | None = None,
) -> int:
    from datasets import load_dataset

    ds = load_dataset(source, split="train", streaming=True)
    n_in = n_out = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in ds:
            n_in += 1
            norm = normalize_record(rec)
            if not norm:
                continue
            joined = "\n".join(m["content"] for m in norm["messages"])
            tlen = approx_token_len(joined)
            if not (min_tokens <= tlen <= max_tokens):
                continue
            if english_only and not looks_english(joined):
                continue
            f.write(json.dumps(norm, ensure_ascii=False) + "\n")
            n_out += 1
            if max_records and n_out >= max_records:
                break
            if n_in % 10000 == 0:
                print(f"[filter] in={n_in:,} kept={n_out:,}")
    print(f"[filter] FINAL in={n_in:,} kept={n_out:,} → {out_path}")
    return n_out


def reservoir_sample(in_path: Path, out_path: Path, *, k: int, seed: int) -> int:
    """k-sized reservoir sample over a JSONL file."""
    import random

    rng = random.Random(seed)
    sample: list[str] = []
    with in_path.open() as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            if i < k:
                sample.append(line)
            else:
                j = rng.randint(0, i)
                if j < k:
                    sample[j] = line
    rng.shuffle(sample)
    with out_path.open("w") as f:
        for line in sample:
            f.write(line + "\n")
    return len(sample)


def run_module(module_path: Path, args: Iterable[str]) -> None:
    cmd = [sys.executable, str(module_path), *args]
    print(f"[step] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/scratch/finetune/data")
    ap.add_argument("--source", default="allenai/tulu-3-sft-mixture")
    ap.add_argument("--target", type=int, default=20_000)
    ap.add_argument("--min_tokens", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--prefilter_cap", type=int, default=200_000,
                    help="Stop streaming after this many *kept* records (saves time)")
    ap.add_argument("--dedup_threshold", type=float, default=0.8)
    ap.add_argument("--decon_n", type=int, default=13)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_dedup", action="store_true")
    ap.add_argument("--skip_decon", action="store_true")
    args = ap.parse_args()

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    raw = work / "tulu3_filtered.jsonl"
    deduped = work / "tulu3_dedup.jsonl"
    cleaned = work / "tulu3_clean.jsonl"
    final = work / f"base_mix_{args.target // 1000}k.jsonl"

    if not raw.exists():
        print(f"[1/4] streaming + length-filter from {args.source}")
        stream_filter(
            args.source, raw,
            min_tokens=args.min_tokens, max_tokens=args.max_tokens,
            english_only=True, max_records=args.prefilter_cap,
        )
    else:
        print(f"[1/4] reusing existing {raw}")

    if args.skip_dedup:
        print("[2/4] dedup skipped")
        deduped = raw
    elif not deduped.exists():
        print("[2/4] MinHash dedup")
        run_module(THIS_DIR / "dedup.py", [
            "--input", str(raw),
            "--output", str(deduped),
            "--threshold", str(args.dedup_threshold),
        ])
    else:
        print(f"[2/4] reusing existing {deduped}")

    if args.skip_decon:
        print("[3/4] decontamination skipped")
        cleaned = deduped
    elif not cleaned.exists():
        print("[3/4] benchmark decontamination")
        run_module(THIS_DIR / "decontaminate.py", [
            "--input", str(deduped),
            "--output", str(cleaned),
            "--n", str(args.decon_n),
        ])
    else:
        print(f"[3/4] reusing existing {cleaned}")

    print(f"[4/4] reservoir-sampling {args.target:,} → {final}")
    n = reservoir_sample(cleaned, final, k=args.target, seed=args.seed)
    print(f"[done] base mix: {n:,} records at {final}")


if __name__ == "__main__":
    main()
