"""
05_pref_gen.py — generate preference pairs for DPO (Project A, ~2.5 GPU-hr).

Uses HF `generate` (no vLLM). For each prompt, samples 2 completions at
temperature 0.9 from the P3 model. Ranks them as follows:

  - Math (GSM8K / verifiable): chosen = the one whose numeric answer matches
    the ground truth; rejected = the other. If both match or neither matches,
    fall back to LENGTH heuristic (longer = chosen) — this is weak but cheap
    and only used when we have no better signal.
  - Code (HumanEval-style): chosen = the one that compiles and passes the
    visible tests; rejected = the other. If both/neither pass: skip the pair.
  - General (non-verifiable): ranked by LENGTH + presence of reasoning markers
    (numbers, code blocks, enumerated steps). Weak, but it only seeds ~20% of
    the pairs and keeps DPO from being pure math.

Writes {"prompt", "chosen", "rejected", "domain", "ranking_reason"} JSONL.
Streaming-write with resume support.

Usage:
  python scripts/05_pref_gen.py \
      --merged_model /scratch/finetune/models/p3_merged \
      --output       /scratch/finetune/data/pref_pairs_3k.jsonl \
      --target       3000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))
from _common import push_to_hub, set_global_seed, setup_env_dirs  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_model", required=True,
                    help="Path to merged P3 weights OR base model + --adapter")
    ap.add_argument("--adapter", default=None,
                    help="Optional PEFT adapter path (if merged_model is base)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--target", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_model_len", type=int, default=2048)

    ap.add_argument("--n_math", type=int, default=2000)
    ap.add_argument("--n_code", type=int, default=600)
    ap.add_argument("--n_general", type=int, default=400)
    ap.add_argument("--hub_dataset_id", default=None)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# -----------------------------------------------------------------------------
# Seed prompts — UNSEEN vs P3's training data (fresh GSM8K-test slice,
# fresh MBPP slice, and general prompts that don't appear in reasoning_mix).
# -----------------------------------------------------------------------------
def collect_pref_seeds(n_math: int, n_code: int, n_general: int, seed: int):
    import random as _r
    from datasets import load_dataset

    seeds = []

    # Math: GSM8K-test (1319 problems). Ground-truth is the numeric answer.
    print(f"[seeds] gsm8k test (math) target={n_math}")
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        idx = _r.Random(seed).sample(range(len(ds)), min(n_math, len(ds)))
        for i in idx:
            ex = ds[i]
            gt = ex["answer"].split("####")[-1].strip().replace(",", "")
            seeds.append({
                "prompt": ex["question"].strip(),
                "domain": "math",
                "ground_truth": gt,
            })
    except Exception as e:
        print(f"[seeds] gsm8k failed: {e}")

    # Code: MBPP validation + HumanEval. Both have test cases we can run.
    print(f"[seeds] code target={n_code}")
    code_seeds: list[dict] = []
    for (path, split) in [
        ("google-research-datasets/mbpp", "validation"),
        ("openai_humaneval", "test"),
    ]:
        try:
            ds = load_dataset(path, split=split)
            for ex in ds:
                if "text" in ex:
                    problem = ex["text"]; tests = ex.get("test_list", [])
                    canonical = ex.get("code", "")
                elif "prompt" in ex:
                    problem = ex["prompt"]; tests = [ex.get("test", "")]
                    canonical = ex.get("canonical_solution", "")
                else:
                    continue
                code_seeds.append({
                    "prompt": (
                        f"{problem}\n\nWrite Python code. It must pass these "
                        f"tests:\n" + "\n".join(tests)
                    ),
                    "domain": "code",
                    "tests": tests,
                    "canonical_solution": canonical,
                })
        except Exception as e:
            print(f"[seeds] code {path}/{split} failed: {e}")
    _r.Random(seed + 1).shuffle(code_seeds)
    seeds += code_seeds[:n_code]

    # General: OpenHermes-2.5 short instructions (diversity seed)
    print(f"[seeds] general target={n_general}")
    general = []
    try:
        ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
        for ex in ds:
            msgs = ex.get("conversations") or []
            user = next((m.get("value", "") for m in msgs
                         if m.get("from") in ("human", "user")), "")
            if user and 50 <= len(user) <= 600:
                general.append({
                    "prompt": user.strip(),
                    "domain": "general",
                })
            if len(general) >= n_general * 4:
                break
    except Exception as e:
        print(f"[seeds] general failed: {e}")
    _r.Random(seed + 2).shuffle(general)
    seeds += general[:n_general]

    _r.Random(seed + 99).shuffle(seeds)
    print(f"[seeds] total = {len(seeds)}")
    return seeds


# -----------------------------------------------------------------------------
# Ranking
# -----------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_last_number(text: str) -> str | None:
    m = list(_NUM_RE.finditer(text[-500:]))
    if not m:
        return None
    return m[-1].group(0).replace(",", "")


def rank_math(a: str, b: str, ground_truth: str):
    """Returns ('chosen_idx', reason) where chosen_idx is 0 or 1."""
    gt = ground_truth.rstrip(".0")
    na = (extract_last_number(a) or "").rstrip(".0")
    nb = (extract_last_number(b) or "").rstrip(".0")
    a_ok = na == gt
    b_ok = nb == gt
    if a_ok and not b_ok:
        return 0, f"math_correct:{gt}"
    if b_ok and not a_ok:
        return 1, f"math_correct:{gt}"
    if a_ok and b_ok:
        return (0 if len(a) <= len(b) else 1), "math_both_correct_shorter_wins"
    return None, "math_neither_correct"


def rank_code(a: str, b: str, tests: list[str]):
    """Runs each response's code against the tests (tight timeout).
    Returns (chosen_idx, reason) or (None, reason)."""
    import tempfile, subprocess

    def extract_code(resp: str) -> str:
        m = re.search(r"```(?:python)?\n(.*?)```", resp, re.DOTALL)
        return m.group(1) if m else resp

    def run(code: str) -> bool:
        try:
            src = code + "\n" + "\n".join(tests)
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
                tf.write(src); tf.flush()
                r = subprocess.run(
                    [sys.executable, tf.name],
                    capture_output=True, timeout=8,
                )
            return r.returncode == 0
        except Exception:
            return False

    a_ok = run(extract_code(a))
    b_ok = run(extract_code(b))
    if a_ok and not b_ok:
        return 0, "code_a_passes"
    if b_ok and not a_ok:
        return 1, "code_b_passes"
    return None, "code_both_or_neither"


_MARKERS = (r"\b\d+\.", r"```", r"\n-\s", r"\n\*\s", r"\nStep")


def reasoning_score(s: str) -> int:
    score = min(len(s), 2000) // 20
    for pat in _MARKERS:
        score += 5 * len(re.findall(pat, s))
    return score


def rank_general(a: str, b: str):
    sa, sb = reasoning_score(a), reasoning_score(b)
    if sa == sb:
        return (0 if len(a) > len(b) else 1), "general_length_tiebreak"
    return (0 if sa > sb else 1), f"general_reasoning_score:{sa}_{sb}"


# -----------------------------------------------------------------------------
# Generation + pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    setup_env_dirs()
    set_global_seed(args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Raw completions file — lets us resume generation without redoing pairs.
    raw_path = out_path.with_suffix(".raw.jsonl")

    seeds = collect_pref_seeds(args.n_math, args.n_code, args.n_general, args.seed)

    # Resume
    done_prompts: set[str] = set()
    raw_records: list[dict] = []
    if raw_path.exists():
        with raw_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                done_prompts.add(r["prompt"])
                raw_records.append(r)
        print(f"[resume] {len(done_prompts)} prompts already generated")

    todo = [s for s in seeds if s["prompt"] not in done_prompts]
    if todo:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[gen] loading {args.merged_model}")
        tok = AutoTokenizer.from_pretrained(args.merged_model, padding_side="left")
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            args.merged_model, torch_dtype=torch.bfloat16,
            device_map="auto", low_cpu_mem_usage=True,
        )
        if args.adapter:
            from peft import PeftModel
            print(f"[gen] loading adapter {args.adapter}")
            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

        raw_f = raw_path.open("a", buffering=1)
        t0 = time.time()
        try:
            for i in range(0, len(todo), args.batch_size):
                batch = todo[i : i + args.batch_size]
                prompts = [tok.apply_chat_template(
                    [{"role": "user", "content": s["prompt"]}],
                    tokenize=False, add_generation_prompt=True)
                    for s in batch]
                enc = tok(prompts, return_tensors="pt", padding=True,
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
                        num_return_sequences=2,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                        repetition_penalty=1.05,
                    )
                out = out[:, input_lens:]
                decoded = tok.batch_decode(out, skip_special_tokens=True)
                for si, s in enumerate(batch):
                    a = decoded[si * 2].strip()
                    b = decoded[si * 2 + 1].strip()
                    rec = {**s, "response_a": a, "response_b": b}
                    raw_records.append(rec)
                    raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done = min(i + args.batch_size, len(todo))
                rate = done / max(time.time() - t0, 1)
                eta = (len(todo) - done) / max(rate, 1e-6) / 60
                print(f"[gen] {done}/{len(todo)} ({rate:.2f}/s, ETA {eta:.0f} min)",
                      flush=True)
        finally:
            raw_f.close()
    else:
        print("[gen] all prompts already generated; proceeding to ranking")

    # Rank
    pairs: list[dict] = []
    stats = {"kept": 0, "skipped": 0, "reasons": {}}
    for r in raw_records:
        a, b = r["response_a"], r["response_b"]
        if not a or not b or a.strip() == b.strip():
            stats["skipped"] += 1
            stats["reasons"]["empty_or_identical"] = \
                stats["reasons"].get("empty_or_identical", 0) + 1
            continue
        if r["domain"] == "math" and r.get("ground_truth"):
            idx, reason = rank_math(a, b, r["ground_truth"])
        elif r["domain"] == "code" and r.get("tests"):
            idx, reason = rank_code(a, b, r["tests"])
        else:
            idx, reason = rank_general(a, b)
        stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
        if idx is None:
            stats["skipped"] += 1
            continue
        chosen, rejected = (a, b) if idx == 0 else (b, a)
        pairs.append({
            "prompt": r["prompt"],
            "chosen": chosen,
            "rejected": rejected,
            "domain": r["domain"],
            "ranking_reason": reason,
        })
        stats["kept"] += 1

    print(f"[rank] kept {stats['kept']}, skipped {stats['skipped']}")
    print(f"[rank] reasons: {stats['reasons']}")

    if len(pairs) > args.target:
        import random as _r
        _r.Random(args.seed).shuffle(pairs)
        pairs = pairs[: args.target]
        print(f"[done] trimmed to {args.target}")

    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(pairs)} pairs → {out_path}")

    if args.hub_dataset_id:
        stage = out_path.parent / f"_hf_stage_{out_path.stem}"
        stage.mkdir(exist_ok=True)
        (stage / out_path.name).write_bytes(out_path.read_bytes())
        (stage / "README.md").write_text(
            f"# Preference pairs for DPO (P4)\n\n"
            f"{len(pairs)} pairs, chosen by verifiable-correctness where possible:\n"
            f"- math: exact-match vs GSM8K ground truth\n"
            f"- code: unit tests pass/fail (MBPP + HumanEval)\n"
            f"- general: reasoning-score heuristic (weaker signal)\n\n"
            f"Source: Qwen2.5-72B after P3 reasoning-SFT continuation.\n"
        )
        push_to_hub(local_dir=stage, repo_id=args.hub_dataset_id,
                    repo_type="dataset",
                    commit_message=f"DPO preference pairs ({len(pairs)})")
        print(f"[push] OK → https://huggingface.co/datasets/{args.hub_dataset_id}")


if __name__ == "__main__":
    main()
