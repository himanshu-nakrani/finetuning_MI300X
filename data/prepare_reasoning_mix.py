"""
prepare_reasoning_mix.py — build a 10k reasoning mix for P3 from public sources.

Replaces the self-generated synth path (03_synth_gen.py) that vLLM/Unsloth
couldn't deliver within budget. Uses high-quality, non-gated datasets:

  - nvidia/OpenMathInstruct-2       : ~6k sampled  (math reasoning + solutions)
  - m-a-p/CodeFeedback-Filtered-Instruction : ~4k sampled  (code problems)
  - (optional) existing synth_reasoning_5k.raw.jsonl — whatever completions
    HF-generate managed to produce; tagged source="p2_self_generated" so the
    portfolio story is honest.

Pipeline mirrors prepare_base_mix.py:
  1. Normalize each source to {"messages": [...], "source": "..."}.
  2. MinHash dedup (Jaccard 0.9).
  3. Decontaminate vs GSM8K-test, HumanEval-test, MATH (if available).
  4. Reservoir sample to --target.

Usage:
  python data/prepare_reasoning_mix.py --target 10000
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def approx_tokens(text: str) -> int:
    return int(len(_WORD_RE.findall(text)) / 0.75)


def load_openmathinstruct(n: int, seed: int) -> list[dict]:
    """Math reasoning — Nvidia's 14M-sample SFT set, we take a diverse slice."""
    from datasets import load_dataset
    print(f"[src] streaming nvidia/OpenMathInstruct-2 (target {n})")
    try:
        ds = load_dataset("nvidia/OpenMathInstruct-2",
                          split="train", streaming=True)
    except Exception as e:
        print(f"[src] nvidia/OpenMathInstruct-2 failed ({type(e).__name__}); "
              f"falling back to meta-math/MetaMathQA")
        ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)

    out, seen_questions = [], set()
    for ex in ds:
        # Schema is slightly different between sources; try common keys.
        q = ex.get("problem") or ex.get("question") or ex.get("query") or ""
        a = (ex.get("generated_solution") or ex.get("solution")
             or ex.get("response") or ex.get("answer") or "")
        q = q.strip(); a = a.strip()
        if not q or not a:
            continue
        # Dedup at question level (cheap first pass; MinHash will catch rest).
        key = q[:200]
        if key in seen_questions:
            continue
        seen_questions.add(key)
        toks = approx_tokens(q + a)
        if not (64 <= toks <= 2500):
            continue
        out.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "source": "openmathinstruct2",
            "domain": "math",
        })
        if len(out) >= n * 3:   # oversample so dedup/decon have headroom
            break
    print(f"[src] openmathinstruct: collected {len(out)}")
    import random
    random.Random(seed).shuffle(out)
    return out[: n * 2]      # cap at 2x target so we don't dominate dedup


def load_codefeedback(n: int, seed: int) -> list[dict]:
    """Code problems with step-by-step reasoning."""
    from datasets import load_dataset
    print(f"[src] streaming m-a-p/CodeFeedback-Filtered-Instruction (target {n})")
    candidates = []
    for path in ("m-a-p/CodeFeedback-Filtered-Instruction",
                 "m-a-p/Code-Feedback"):
        try:
            ds = load_dataset(path, split="train", streaming=True)
            for ex in ds:
                # Common schemas in this dataset family
                if "query" in ex and "answer" in ex:
                    q, a = ex["query"], ex["answer"]
                elif "instruction" in ex and "output" in ex:
                    q, a = ex["instruction"], ex["output"]
                elif "messages" in ex and ex["messages"]:
                    msgs = ex["messages"]
                    q = next((m["content"] for m in msgs
                              if m.get("role") == "user"), "")
                    a = next((m["content"] for m in msgs
                              if m.get("role") == "assistant"), "")
                else:
                    continue
                q, a = (q or "").strip(), (a or "").strip()
                if not q or not a:
                    continue
                toks = approx_tokens(q + a)
                if not (64 <= toks <= 3000):
                    continue
                candidates.append({
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                    "source": "codefeedback",
                    "domain": "code",
                })
                if len(candidates) >= n * 3:
                    break
            print(f"[src] codefeedback: collected {len(candidates)} from {path}")
            break
        except Exception as e:
            print(f"[src] {path} miss: {e}")
    import random
    random.Random(seed + 1).shuffle(candidates)
    return candidates[: n * 2]


def load_p2_self_generated(raw_path: Path) -> list[dict]:
    """Whatever the dead synth_gen run wrote before we killed it. Keep for
    portfolio honesty ('we generated some ourselves'). Each record:
    {prompt, response, domain, ...}
    """
    if not raw_path.exists():
        print(f"[src] {raw_path} not present, skipping self-generated")
        return []
    out = []
    with raw_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            q = (r.get("prompt") or "").strip()
            a = (r.get("response") or "").strip()
            if not q or not a or len(a) < 40:
                continue
            out.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "source": "p2_self_generated",
                "domain": r.get("domain", "reasoning"),
            })
    print(f"[src] self-generated: {len(out)} usable records from {raw_path}")
    return out


def reservoir_sample(records: list[dict], k: int, seed: int) -> list[dict]:
    import random
    if len(records) <= k:
        return records
    rng = random.Random(seed + 99)
    sample = list(records[:k])
    for i, r in enumerate(records[k:], start=k):
        j = rng.randint(0, i)
        if j < k:
            sample[j] = r
    rng.shuffle(sample)
    return sample


def run_module(script: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script), *args]
    print(f"[step] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/scratch/finetune/data")
    ap.add_argument("--target", type=int, default=10000)
    ap.add_argument("--n_math", type=int, default=6000)
    ap.add_argument("--n_code", type=int, default=4000)
    ap.add_argument("--self_gen_raw",
                    default="/scratch/finetune/data/synth_reasoning_5k.raw.jsonl")
    ap.add_argument("--dedup_threshold", type=float, default=0.9)
    ap.add_argument("--decon_n", type=int, default=13)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_dedup", action="store_true")
    ap.add_argument("--skip_decon", action="store_true")
    args = ap.parse_args()

    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    raw_combined = work / "reasoning_raw.jsonl"
    deduped = work / "reasoning_dedup.jsonl"
    cleaned = work / "reasoning_clean.jsonl"
    final = work / f"reasoning_mix_{args.target // 1000}k.jsonl"

    # [1/4] collect from all sources
    if not raw_combined.exists():
        print(f"[1/4] collecting from public sources + self-generated")
        all_records = []
        all_records += load_openmathinstruct(args.n_math, args.seed)
        all_records += load_codefeedback(args.n_code, args.seed)
        all_records += load_p2_self_generated(Path(args.self_gen_raw))
        import random
        random.Random(args.seed).shuffle(all_records)
        with raw_combined.open("w") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[1/4] wrote {len(all_records):,} raw records → {raw_combined}")
    else:
        print(f"[1/4] reusing {raw_combined}")

    # [2/4] dedup (reuse existing data/dedup.py)
    if args.skip_dedup:
        deduped = raw_combined
    elif not deduped.exists():
        print("[2/4] MinHash dedup")
        run_module(THIS_DIR / "dedup.py", [
            "--input", str(raw_combined),
            "--output", str(deduped),
            "--threshold", str(args.dedup_threshold),
        ])
    else:
        print(f"[2/4] reusing {deduped}")

    # [3/4] decontaminate (reuse existing data/decontaminate.py)
    if args.skip_decon:
        cleaned = deduped
    elif not cleaned.exists():
        print("[3/4] benchmark decontamination")
        run_module(THIS_DIR / "decontaminate.py", [
            "--input", str(deduped),
            "--output", str(cleaned),
            "--benchmarks", "gsm8k", "humaneval", "math",
            "--n", str(args.decon_n),
        ])
    else:
        print(f"[3/4] reusing {cleaned}")

    # [4/4] reservoir sample
    print(f"[4/4] reservoir-sampling {args.target} → {final}")
    rows = []
    with cleaned.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    sampled = reservoir_sample(rows, args.target, args.seed)
    with final.open("w") as f:
        for r in sampled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    from collections import Counter
    by_source = Counter(r.get("source", "?") for r in sampled)
    by_domain = Counter(r.get("domain", "?") for r in sampled)
    print(f"[done] {len(sampled)} records at {final}")
    print(f"[stats] by source: {dict(by_source)}")
    print(f"[stats] by domain: {dict(by_domain)}")


if __name__ == "__main__":
    main()
