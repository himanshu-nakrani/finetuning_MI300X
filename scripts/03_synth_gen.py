"""
03_synth_gen.py — Day 2: synthetic reasoning/code data generation via vLLM.

Pipeline:
  1. Collect seed prompts (GSM8K-train, MBPP, general instructions)
  2. Add evol-instruct mutations to widen distribution
  3. Generate N completions/prompt via vLLM serving the merged P2 model
  4. Quality filter (length, schema, repetition)
  5. MinHash dedup against itself + against the P2 base mix
  6. Write JSONL `{prompt, response, domain, ...}` and push to HF

Two ways to talk to vLLM:
  - "online": vllm serve (already running) → OpenAI-compatible HTTP client
  - "offline": from vllm import LLM (in-process, no HTTP)

We default to **offline** because it's simpler (one process, no port mgmt, no
network), uses the same VRAM, and avoids a second HF model load. Switch to
online with --mode online if you want to test the OpenAI client path.

Usage (offline, recommended):
  python scripts/03_synth_gen.py \
      --merged_model /scratch/finetune/models/p2_merged \
      --output /scratch/finetune/data/synth_reasoning_15k.jsonl \
      --target 15000

Usage (online — vllm serve must already be running on :8000):
  python scripts/03_synth_gen.py --mode online \
      --base_url http://127.0.0.1:8000/v1 \
      --model_name p2_merged ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))
from _common import push_to_hub, set_global_seed, setup_env_dirs  # noqa: E402

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["offline", "online", "hf", "unsloth"],
                    default="offline",
                    help="offline = in-process vllm.LLM; online = HTTP to a running "
                         "vllm server; hf = plain transformers .generate (no vllm "
                         "needed); unsloth = Unsloth FastLanguageModel batched generate "
                         "(no vllm needed, ~2-3x faster than hf)")
    # offline-mode args
    ap.add_argument("--merged_model", default="/scratch/finetune/models/p2_merged",
                    help="Path to merged P2 weights (offline mode)")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem_util", type=float, default=0.92)
    # online-mode args
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model_name", default="p2_merged",
                    help="Name vLLM is serving the model under (online mode)")

    ap.add_argument("--output", required=True,
                    help="Output JSONL of synthetic data")
    ap.add_argument("--target", type=int, default=15000,
                    help="Final dataset size after dedup/filter")
    ap.add_argument("--n_per_prompt", type=int, default=2,
                    help="Completions to generate per seed prompt")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=64,
                    help="Prompts per vLLM request batch")

    ap.add_argument("--n_math", type=int, default=4000)
    ap.add_argument("--n_code", type=int, default=2500)
    ap.add_argument("--n_general", type=int, default=1500)
    ap.add_argument("--mutation_rate", type=float, default=0.5)

    ap.add_argument("--dedup_threshold", type=float, default=0.85)
    ap.add_argument("--also_dedup_against",
                    default="/scratch/finetune/data/base_mix_20k.jsonl",
                    help="Cross-dedup against this JSONL (P2's base mix). Pass empty to skip.")

    ap.add_argument("--hub_dataset_id", default=None,
                    help="Optional HF dataset repo_id; pushes the JSONL when done")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# -----------------------------------------------------------------------------
# Quality filters
# -----------------------------------------------------------------------------
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def looks_garbage(text: str) -> str | None:
    """Return a reason string if `text` should be rejected, else None."""
    t = text.strip()
    if not t:
        return "empty"
    if len(t) < 30:
        return "too_short"
    if len(t) > 8000:
        return "too_long"

    words = _WORD_RE.findall(t)
    if len(words) < 10:
        return "too_few_words"
    # Repetition heuristic: if any 5-gram appears > 6 times, it's looping.
    if len(words) >= 30:
        from collections import Counter
        grams = [tuple(words[i:i + 5]) for i in range(len(words) - 4)]
        if grams:
            top, count = Counter(grams).most_common(1)[0]
            if count >= 6:
                return f"repetition (5-gram x{count})"
    # Unicode garbage: huge ratio of non-printable / replacement chars
    nonprint = sum(1 for c in t if ord(c) < 32 and c not in "\n\t\r")
    if nonprint / len(t) > 0.02:
        return "nonprint_ratio"
    return None


def render_chat_prompt(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True,
    )


# -----------------------------------------------------------------------------
# Generation backends
# -----------------------------------------------------------------------------
def gen_offline(prompts: list[dict], args) -> list[dict]:
    """In-process vLLM generation. Returns list of records."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"[gen] vLLM (offline) loading {args.merged_model}")
    llm = LLM(
        model=args.merged_model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="bfloat16",
        enforce_eager=False,
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(args.merged_model)
    sp = SamplingParams(
        n=args.n_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    rendered = [render_chat_prompt(tok, s["prompt"]) for s in prompts]
    print(f"[gen] generating {len(rendered)} prompts × {args.n_per_prompt} completions...")
    t0 = time.time()
    outputs = llm.generate(rendered, sp)
    dt = (time.time() - t0) / 60.0
    print(f"[gen] done in {dt:.1f} min "
          f"({len(rendered) * args.n_per_prompt / max(dt * 60, 1):.1f} compl/s)")

    records = []
    for seed, out in zip(prompts, outputs):
        for completion in out.outputs:
            records.append({
                "prompt": seed["prompt"],
                "response": completion.text.strip(),
                "domain": seed["domain"],
                "ground_truth": seed.get("ground_truth"),
                "is_evol": seed.get("is_evol", False),
                "stop_reason": completion.finish_reason,
            })
    return records


def gen_unsloth(prompts: list[dict], args) -> list[dict]:
    """Unsloth FastLanguageModel batched generation — no vLLM needed.

    Writes raw completions to <output>.raw.jsonl as each batch finishes so a
    kill/crash doesn't lose work. The raw file is loaded back on resume.
    """
    import torch
    from unsloth import FastLanguageModel

    # Resume support — skip any seeds whose prompt already has a completion written.
    raw_path = Path(args.output).with_suffix(".raw.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    records: list[dict] = []
    if raw_path.exists():
        with raw_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    seen.add(r["prompt"])
                    records.append(r)
                except Exception:
                    continue
        print(f"[gen-unsloth] RESUMING: {len(seen):,} prompts already done "
              f"from {raw_path}")

    todo = [s for s in prompts if s["prompt"] not in seen]
    if not todo:
        print("[gen-unsloth] all prompts already generated; skipping load")
        return records

    print(f"[gen-unsloth] loading {args.merged_model}")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.merged_model,
        max_seq_length=args.max_model_len,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    rendered = [render_chat_prompt(tok, s["prompt"]) for s in todo]
    n_prompts = len(rendered)
    print(f"[gen-unsloth] generating {n_prompts} prompts × {args.n_per_prompt} "
          f"(batch_size={args.batch_size})  [skipped {len(seen)} already-done]")

    t0 = time.time()
    # Append mode so resume writes keep prior progress
    raw_f = raw_path.open("a", buffering=1)  # line-buffered
    try:
        for i in range(0, n_prompts, args.batch_size):
            batch_prompts = rendered[i : i + args.batch_size]
            batch_seeds = todo[i : i + args.batch_size]

            enc = tok(batch_prompts, return_tensors="pt", padding=True,
                      truncation=True,
                      max_length=args.max_model_len - args.max_new_tokens)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            input_lens = enc["input_ids"].shape[1]

            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.n_per_prompt,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                    repetition_penalty=1.05,
                    use_cache=True,
                )
            out = out[:, input_lens:]
            decoded = tok.batch_decode(out, skip_special_tokens=True)

            for s_idx, seed in enumerate(batch_seeds):
                for k in range(args.n_per_prompt):
                    completion = decoded[s_idx * args.n_per_prompt + k]
                    rec = {
                        "prompt": seed["prompt"],
                        "response": completion.strip(),
                        "domain": seed["domain"],
                        "ground_truth": seed.get("ground_truth"),
                        "is_evol": seed.get("is_evol", False),
                        "stop_reason": "length_or_eos",
                    }
                    records.append(rec)
                    raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            done = min(i + args.batch_size, n_prompts)
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1)
            eta_min = (n_prompts - done) / max(rate, 1e-6) / 60
            print(f"[gen-unsloth] {done}/{n_prompts} prompts ({rate:.2f}/s, "
                  f"ETA {eta_min:.0f} min, {len(records)} completions total)",
                  flush=True)
    finally:
        raw_f.close()

    print(f"[gen-unsloth] done in {(time.time() - t0) / 60:.1f} min, "
          f"{len(records)} completions total (incl. resumed)")
    return records


def gen_hf(prompts: list[dict], args) -> list[dict]:
    """HuggingFace `.generate` fallback — no vLLM required.

    Slower than vLLM (~3-5x) because no continuous batching, but works on any
    ROCm install without dev toolchain. Uses left-padding + manual batching.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[gen-hf] loading {args.merged_model} (bf16, device_map=auto)")
    tok = AutoTokenizer.from_pretrained(args.merged_model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.merged_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Resume support (shared with gen_unsloth)
    raw_path = Path(args.output).with_suffix(".raw.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    records: list[dict] = []
    if raw_path.exists():
        with raw_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    seen.add(r["prompt"])
                    records.append(r)
                except Exception:
                    continue
        print(f"[gen-hf] RESUMING: {len(seen):,} already done from {raw_path}")
    todo = [s for s in prompts if s["prompt"] not in seen]
    if not todo:
        print("[gen-hf] all prompts already generated")
        return records

    rendered = [render_chat_prompt(tok, s["prompt"]) for s in todo]
    n_prompts = len(rendered)
    print(f"[gen-hf] generating {n_prompts} prompts × {args.n_per_prompt} completions "
          f"(batch_size={args.batch_size})  [skipped {len(seen)}]")

    t0 = time.time()
    raw_f = raw_path.open("a", buffering=1)
    for i in range(0, n_prompts, args.batch_size):
        batch_prompts = rendered[i : i + args.batch_size]
        batch_seeds = todo[i : i + args.batch_size]

        enc = tok(batch_prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_model_len - args.max_new_tokens)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        input_lens = enc["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.n_per_prompt,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                repetition_penalty=1.05,
            )
        # out shape: (batch * n, seq)
        out = out[:, input_lens:]
        decoded = tok.batch_decode(out, skip_special_tokens=True)

        for s_idx, seed in enumerate(batch_seeds):
            for k in range(args.n_per_prompt):
                completion = decoded[s_idx * args.n_per_prompt + k]
                rec = {
                    "prompt": seed["prompt"],
                    "response": completion.strip(),
                    "domain": seed["domain"],
                    "ground_truth": seed.get("ground_truth"),
                    "is_evol": seed.get("is_evol", False),
                    "stop_reason": "length_or_eos",
                }
                records.append(rec)
                raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        done = min(i + args.batch_size, n_prompts)
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1)
        eta_min = (n_prompts - done) / max(rate, 1e-6) / 60
        print(f"[gen-hf] {done}/{n_prompts} prompts ({rate:.2f}/s, "
              f"ETA {eta_min:.0f} min, {len(records)} completions so far)",
              flush=True)
    raw_f.close()

    print(f"[gen-hf] done in {(time.time() - t0) / 60:.1f} min, "
          f"{len(records)} completions total (incl. resumed)")
    return records


def gen_online(prompts: list[dict], args) -> list[dict]:
    """OpenAI-compatible HTTP client to a running `vllm serve`."""
    import concurrent.futures as cf
    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    print(f"[gen] vLLM (online) {args.base_url} model={args.model_name}")

    def one(seed: dict) -> list[dict]:
        try:
            r = client.chat.completions.create(
                model=args.model_name,
                messages=[{"role": "user", "content": seed["prompt"]}],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                n=args.n_per_prompt,
            )
            out = []
            for choice in r.choices:
                out.append({
                    "prompt": seed["prompt"],
                    "response": (choice.message.content or "").strip(),
                    "domain": seed["domain"],
                    "ground_truth": seed.get("ground_truth"),
                    "is_evol": seed.get("is_evol", False),
                    "stop_reason": choice.finish_reason,
                })
            return out
        except Exception as e:
            print(f"[gen] req failed: {e}")
            return []

    records = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=32) as ex:
        for i, batch in enumerate(ex.map(one, prompts)):
            records.extend(batch)
            if (i + 1) % 200 == 0:
                rate = (i + 1) / max(time.time() - t0, 1)
                print(f"[gen] {i + 1}/{len(prompts)} prompts done ({rate:.1f}/s)")
    print(f"[gen] done in {(time.time() - t0) / 60:.1f} min, {len(records)} completions")
    return records


# -----------------------------------------------------------------------------
# Filter + dedup (in-memory MinHash)
# -----------------------------------------------------------------------------
def filter_records(records: list[dict]) -> list[dict]:
    rejects: dict[str, int] = {}
    out = []
    for r in records:
        why = looks_garbage(r["response"])
        if why:
            rejects[why] = rejects.get(why, 0) + 1
            continue
        # Light-touch math correctness boost: if we have a ground truth and the
        # last "boxed"/numeric token in the response matches, mark verified.
        gt = (r.get("ground_truth") or "").strip()
        if gt and r["domain"] in ("math",):
            tail_nums = re.findall(r"-?\d+\.?\d*", r["response"][-200:])
            if tail_nums and tail_nums[-1].rstrip(".0") == gt.rstrip(".0"):
                r["verified"] = True
        out.append(r)
    print(f"[filter] kept {len(out)}/{len(records)}; rejects: {rejects}")
    return out


def shingles(text: str, k: int = 5) -> set[str]:
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def minhash(text: str, num_perm: int = 128) -> "MinHash":
    from datasketch import MinHash
    mh = MinHash(num_perm=num_perm)
    for sh in shingles(text):
        mh.update(sh.encode("utf-8"))
    return mh


def cross_dedup_with_base_mix(
    records: list[dict], base_mix_jsonl: str, threshold: float, num_perm: int = 128
) -> list[dict]:
    """Drop records whose prompt is too similar to any prompt in the base mix
    (prevents the synthetic set from re-teaching what P2 already saw)."""
    from datasketch import MinHashLSH

    if not base_mix_jsonl or not Path(base_mix_jsonl).exists():
        print("[xdedup] no base-mix file, skipping cross-dedup")
        return records

    print(f"[xdedup] indexing base mix prompts from {base_mix_jsonl}")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    with open(base_mix_jsonl) as f:
        for i, line in enumerate(f):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = rec.get("messages") or []
            user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            if not user:
                continue
            lsh.insert(f"b{i}", minhash(user, num_perm))

    out, dropped = [], 0
    for r in records:
        if lsh.query(minhash(r["prompt"], num_perm)):
            dropped += 1
        else:
            out.append(r)
    print(f"[xdedup] dropped {dropped}/{len(records)} (overlap with base mix)")
    return out


def self_dedup(records: list[dict], threshold: float, num_perm: int = 128) -> list[dict]:
    """Drop near-duplicate (prompt, response) pairs."""
    from datasketch import MinHashLSH

    print(f"[selfdedup] threshold={threshold}, n={len(records)}")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    out, dropped = [], 0
    for i, r in enumerate(records):
        text = r["prompt"] + "\n" + r["response"]
        mh = minhash(text, num_perm)
        if lsh.query(mh):
            dropped += 1
            continue
        lsh.insert(f"r{i}", mh)
        out.append(r)
    print(f"[selfdedup] kept {len(out)}, dropped {dropped}")
    return out


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    setup_env_dirs()
    set_global_seed(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Seeds + evol-mutation
    sys.path.insert(0, str(REPO_ROOT))
    from data.synth_seeds.seed_prompts import collect_seeds, mutate_prompts
    seeds = collect_seeds(
        n_math=args.n_math, n_code=args.n_code, n_general=args.n_general, seed=args.seed
    )
    seeds = mutate_prompts(seeds, mutation_rate=args.mutation_rate, seed=args.seed)

    # 2. Generate
    if args.mode == "offline":
        records = gen_offline(seeds, args)
    elif args.mode == "online":
        records = gen_online(seeds, args)
    elif args.mode == "hf":
        records = gen_hf(seeds, args)
    elif args.mode == "unsloth":
        records = gen_unsloth(seeds, args)
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    # 3. Filter
    records = filter_records(records)

    # 4. Cross-dedup against base mix
    records = cross_dedup_with_base_mix(
        records, args.also_dedup_against, args.dedup_threshold
    )

    # 5. Self-dedup
    records = self_dedup(records, args.dedup_threshold)

    # 6. Trim to target
    if len(records) > args.target:
        import random as _r
        _r.Random(args.seed).shuffle(records)
        records = records[: args.target]
        print(f"[done] trimmed to target={args.target}")
    elif len(records) < args.target:
        print(f"[warn] only {len(records)} records survived filtering "
              f"(target was {args.target}). Proceeding anyway.")

    # 7. Write
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(records)} records → {out_path}")

    # Quick stats by domain
    from collections import Counter
    print("[stats] by domain:", dict(Counter(r["domain"] for r in records)))
    print("[stats] verified math:", sum(1 for r in records if r.get("verified")))

    # 8. Push to HF as a dataset
    if args.hub_dataset_id:
        print(f"[push] uploading {out_path} → {args.hub_dataset_id} (dataset)")
        # Easiest path: stage a tiny folder and upload it
        stage = out_path.parent / f"_hf_stage_{out_path.stem}"
        stage.mkdir(exist_ok=True)
        target = stage / out_path.name
        if target.exists():
            target.unlink()
        os.link(out_path, target) if hasattr(os, "link") else target.write_bytes(
            out_path.read_bytes()
        )
        # Drop a minimal README
        (stage / "README.md").write_text(
            f"# Synthetic reasoning/code SFT data (P3)\n\n"
            f"Generated by `scripts/03_synth_gen.py` from seeds: GSM8K-train, "
            f"MBPP, general instructions; with evol-instruct mutations.\n\n"
            f"- Source model: P2 merged (`scripts/_merge_adapter.py` of the P2 adapter)\n"
            f"- Records: {len(records)}\n"
            f"- Sampling: temp={args.temperature}, top_p={args.top_p}, "
            f"n={args.n_per_prompt}/prompt\n"
            f"- Dedup: MinHash Jaccard {args.dedup_threshold}, cross-deduped "
            f"vs the P2 base mix\n"
        )
        push_to_hub(
            local_dir=stage, repo_id=args.hub_dataset_id, repo_type="dataset",
            commit_message=f"Synthetic SFT data ({len(records)} records)",
        )
        print(f"[push] OK → https://huggingface.co/datasets/{args.hub_dataset_id}")


if __name__ == "__main__":
    main()
