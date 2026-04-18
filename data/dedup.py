"""
dedup.py — MinHash near-duplicate removal for instruction/chat data.

Operates on a JSONL of {"messages": [...]} (or "text"). Drops any record whose
MinHash Jaccard similarity to a kept record is >= --threshold (default 0.8).

Streaming-friendly: holds a MinHashLSH index in memory but never the full text.
On 200k rows it peaks well under 8 GB.

Usage:
  python data/dedup.py \
      --input  /scratch/finetune/data/raw_tulu3_filtered.jsonl \
      --output /scratch/finetune/data/raw_tulu3_dedup.jsonl \
      --threshold 0.8 \
      --num_perm 128
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasketch import MinHash, MinHashLSH


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def record_text(rec: dict) -> str:
    """Return a single string to fingerprint: the user+assistant content."""
    if "text" in rec and isinstance(rec["text"], str):
        return rec["text"]
    msgs = rec.get("messages") or []
    parts = []
    for m in msgs:
        if isinstance(m, dict) and "content" in m:
            parts.append(str(m["content"]))
    return "\n".join(parts)


def shingles(text: str, k: int = 5) -> set[str]:
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def make_minhash(text: str, num_perm: int, k: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for sh in shingles(text, k):
        mh.update(sh.encode("utf-8"))
    return mh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--num_perm", type=int, default=128)
    ap.add_argument("--shingle_k", type=int, default=5)
    ap.add_argument("--report_every", type=int, default=5000)
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    kept = 0
    dropped = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record_text(rec)
            if not text:
                continue
            mh = make_minhash(text, args.num_perm, args.shingle_k)
            if lsh.query(mh):
                dropped += 1
            else:
                key = f"r{i}"
                lsh.insert(key, mh)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            if (i + 1) % args.report_every == 0:
                print(
                    f"[dedup] read={i + 1:,} kept={kept:,} dropped={dropped:,} "
                    f"({dropped / (i + 1):.1%})"
                )

    print(f"[dedup] FINAL  kept={kept:,}  dropped={dropped:,}  → {args.output}")


if __name__ == "__main__":
    main()
