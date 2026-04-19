"""
seed_prompts.py — sources of seed prompts for synthetic data generation.

Returns lists of {"prompt": str, "domain": str, "ground_truth": str | None}.
Caller (scripts/03_synth_gen.py) feeds them through the model in batches and
applies evol-instruct mutations.

Three seed sources, mixed in a target ratio (default 50% math, 35% code, 15%
general/evol). All sources cap at the requested N and shuffle deterministically.
"""
from __future__ import annotations

import random
from typing import Iterable


# Pool of evol-instruct mutation templates. Used by mutate_prompt() to create
# additional variants from a base prompt — increases diversity without needing
# more seed datasets. Adapted from WizardLM / Evol-Instruct.
EVOL_TEMPLATES = [
    "Rewrite the following problem to make it more challenging by adding "
    "an additional constraint or step. Keep it solvable and unambiguous.\n\n"
    "Original problem:\n{prompt}\n\nRewritten problem:",

    "Take the problem below and increase its difficulty by introducing a "
    "real-world scenario with named entities and concrete numbers. Preserve "
    "the underlying skill being tested.\n\n"
    "Original:\n{prompt}\n\nNew problem:",

    "Generalize the following problem so it is no longer about specific "
    "numbers but about an arbitrary input of the same type. State your "
    "answer in terms of variables.\n\n"
    "Original:\n{prompt}\n\nGeneralized version:",

    "Convert the following problem into a multi-step reasoning question "
    "that requires combining at least two distinct techniques to solve.\n\n"
    "Original:\n{prompt}\n\nMulti-step version:",
]


def _gsm8k_seeds(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    out = []
    for i in idx:
        ex = ds[i]
        # GSM8K answer format: "...\n#### 42"
        ans = ex["answer"].split("####")[-1].strip()
        out.append({
            "prompt": ex["question"].strip(),
            "domain": "math",
            "ground_truth": ans,
        })
    return out


def _mbpp_seeds(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    # MBPP train: 374 problems. Use train + a slice of validation if we need more.
    candidates = []
    for split in ("train", "validation", "prompt"):
        try:
            ds = load_dataset("google-research-datasets/mbpp", split=split)
            for ex in ds:
                candidates.append({
                    "prompt": (
                        f"Write a Python function to solve the following problem.\n"
                        f"Problem: {ex['text'].strip()}\n"
                        f"Your solution must pass these test cases:\n"
                        + "\n".join(ex.get("test_list", []))
                    ),
                    "domain": "code",
                    "ground_truth": ex.get("code", "").strip() or None,
                })
        except Exception as e:
            print(f"[seeds] mbpp split={split} miss: {e}")
    rng = random.Random(seed + 1)
    rng.shuffle(candidates)
    return candidates[:n]


def _general_seeds(n: int, seed: int) -> list[dict]:
    """Pull diverse general-instruction seeds from a non-gated dataset."""
    from datasets import load_dataset
    candidates: list[dict] = []
    # Try a couple of mirrors; first that loads wins.
    for spec in [
        ("teknium/OpenHermes-2.5", None),
        ("HuggingFaceH4/no_robots", None),
        ("databricks/databricks-dolly-15k", None),
    ]:
        path, conf = spec
        try:
            ds = (load_dataset(path, conf, split="train", streaming=True)
                  if conf else load_dataset(path, split="train", streaming=True))
            for i, ex in enumerate(ds):
                # Best-effort field extraction across schemas
                if "conversations" in ex and ex["conversations"]:
                    msgs = ex["conversations"]
                    user = next((m.get("value", "") for m in msgs
                                 if m.get("from") in ("human", "user")), "")
                elif "prompt" in ex:
                    user = ex["prompt"]
                elif "instruction" in ex:
                    user = ex["instruction"]
                else:
                    continue
                if user and 30 <= len(user) <= 1500:
                    candidates.append({
                        "prompt": user.strip(),
                        "domain": "general",
                        "ground_truth": None,
                    })
                if len(candidates) >= n * 4:
                    break
            print(f"[seeds] general loaded {len(candidates)} from {path}")
            break
        except Exception as e:
            print(f"[seeds] {path} miss: {e}")
            continue
    rng = random.Random(seed + 2)
    rng.shuffle(candidates)
    return candidates[:n]


def collect_seeds(
    *,
    n_math: int = 4000,
    n_code: int = 2500,
    n_general: int = 1500,
    seed: int = 42,
) -> list[dict]:
    """Return a deterministically-shuffled list of seed prompts.

    Counts roughly mirror the 50/30/20 ratio that Tulu-3 uses for its
    reasoning-mix SFT, scaled so the post-mutation total lands near 15k.
    """
    print(f"[seeds] collecting math={n_math} code={n_code} general={n_general}")
    seeds = []
    seeds += _gsm8k_seeds(n_math, seed)
    seeds += _mbpp_seeds(n_code, seed)
    seeds += _general_seeds(n_general, seed)
    rng = random.Random(seed + 99)
    rng.shuffle(seeds)
    print(f"[seeds] total = {len(seeds)}")
    return seeds


def mutate_prompts(
    seeds: Iterable[dict],
    *,
    mutation_rate: float = 0.5,
    seed: int = 42,
) -> list[dict]:
    """Wrap a fraction of seeds in an evol-instruct mutation template.

    Returns a list combining originals (always kept) and mutated copies.
    Mutated copies set ground_truth=None because the rewrite changes the answer.
    """
    rng = random.Random(seed + 7)
    out = list(seeds)
    n_mut = int(len(out) * mutation_rate)
    base_for_mut = rng.sample(out, n_mut)
    for s in base_for_mut:
        tmpl = rng.choice(EVOL_TEMPLATES)
        out.append({
            "prompt": tmpl.format(prompt=s["prompt"]),
            "domain": s["domain"] + "_evol",
            "ground_truth": None,           # evolved problem has a new answer
            "is_evol": True,
        })
    rng.shuffle(out)
    print(f"[seeds] after mutation: {len(out)} ({n_mut} evol-rewrites added)")
    return out
