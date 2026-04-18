"""
02_70b_sft.py — P2: Llama-3.1-70B LoRA SFT on the curated 20k base mix.

Day 1 headline: ~6 GPU-hr. Reads a pre-built JSONL of {"messages": [...]} from
data/prepare_base_mix.py. LoRA on bf16 (not QLoRA) — MI300X has the VRAM and
LoRA on bf16 trains noticeably better than QLoRA at this scale.

Usage:
  # 1. Build the base mix (one-time, ~20-40 min CPU+net):
  python data/prepare_base_mix.py --target 20000

  # 2. Train:
  python scripts/02_70b_sft.py \
      --config configs/p2_70b_sft.yaml \
      --hub_repo_id <user>/llama-3.1-70b-instruct-mix-sft
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _common import (  # noqa: E402
    load_yaml,
    make_wallclock_callback,
    push_to_hub,
    set_global_seed,
    setup_env_dirs,
    write_model_card,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p2_70b_sft.yaml")
    ap.add_argument("--hub_repo_id", default=None)
    ap.add_argument("--no_push", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the latest checkpoint in output_dir")
    ap.add_argument("--max_steps", type=int, default=None)
    return ap.parse_args()


def load_jsonl_dataset(path: str, tokenizer, max_seq_length: int, seed: int):
    """Load {"messages": [...]} JSONL → HF dataset with rendered "text"."""
    from datasets import Dataset

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first with:\n"
            f"  python data/prepare_base_mix.py --target 20000"
        )

    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = rec.get("messages")
            if not msgs:
                continue
            rows.append({"messages": msgs})

    print(f"[p2] loaded {len(rows):,} JSONL rows from {path}")

    def render(ex):
        return {
            "text": tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    ds = Dataset.from_list(rows).shuffle(seed=seed)
    ds = ds.map(render, remove_columns=ds.column_names, num_proc=4)

    # Drop rows that exceed max_seq_length after rendering (saves trainer pad time)
    def short_enough(ex):
        return len(tokenizer(ex["text"], add_special_tokens=False)["input_ids"]) <= max_seq_length

    before = len(ds)
    ds = ds.filter(short_enough, num_proc=4)
    print(f"[p2] post length-filter: {len(ds):,}/{before:,}")
    return ds


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = setup_env_dirs()
    set_global_seed(cfg.get("seed", 42))

    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[p2] output dir: {out_dir}")
    print(f"[p2] scratch:    {paths['scratch']}")

    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    print(f"[p2] loading {cfg['model']['name']} (this is 70B, ~5-10 min)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=cfg["model"].get("dtype"),
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        target_modules=cfg["lora"]["target_modules"],
        use_gradient_checkpointing=cfg["lora"]["use_gradient_checkpointing"],
        random_state=cfg.get("seed", 42),
    )

    print("[p2] building dataset")
    train_ds = load_jsonl_dataset(
        cfg["data"]["jsonl_path"],
        tokenizer,
        cfg["model"]["max_seq_length"],
        cfg["data"].get("shuffle_seed", 42),
    )

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg["train"]["num_train_epochs"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        learning_rate=cfg["train"]["learning_rate"],
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        weight_decay=cfg["train"]["weight_decay"],
        optim=cfg["train"]["optim"],
        bf16=cfg["train"]["bf16"],
        logging_steps=cfg["train"]["logging_steps"],
        save_strategy=cfg["train"]["save_strategy"],
        save_steps=cfg["train"]["save_steps"],
        save_total_limit=cfg["train"]["save_total_limit"],
        report_to=cfg["train"]["report_to"],
        run_name=cfg["run_name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dataset_text_field="text",
        seed=cfg.get("seed", 42),
        max_steps=args.max_steps if args.max_steps else -1,
    )

    cb = make_wallclock_callback(cfg["train"].get("max_runtime_minutes"))
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=sft_cfg,
        callbacks=[cb] if cb else [],
    )

    print(f"[p2] training... (wall-clock cap: "
          f"{cfg['train'].get('max_runtime_minutes', 'none')} min)")
    result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    train_min = (time.time() - t0) / 60.0
    print(f"[p2] training done in {train_min:.1f} min "
          f"({train_min / 60:.2f} hr), loss={result.training_loss:.4f}")

    # Save adapter (always)
    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[p2] adapter saved → {adapter_dir}")

    # Optional merged save (skipped by default — 70B merged is ~140 GB on disk)
    merged_dir = None
    if cfg["hub"].get("push_merged"):
        merged_dir = out_dir / "merged_bf16"
        print(f"[p2] saving merged bf16 (large!) → {merged_dir}")
        trainer.model.save_pretrained_merged(
            str(merged_dir), tokenizer, save_method="merged_16bit"
        )

    summary = (
        f"LoRA fine-tune (rank={cfg['lora']['r']}, alpha={cfg['lora']['alpha']}, "
        f"target=`{cfg['lora']['target_modules']}`) of `{cfg['model']['name']}` on a "
        f"20k Tulu-3 base mix (deduped, decontaminated against GSM8K/HumanEval/MATH). "
        f"{cfg['train']['num_train_epochs']} epochs, effective batch size "
        f"{cfg['train']['per_device_train_batch_size'] * cfg['train']['gradient_accumulation_steps']}, "
        f"final loss {result.training_loss:.4f}, {train_min / 60:.2f} hr on a single MI300X."
    )
    write_model_card(
        adapter_dir,
        title="Llama-3.1-70B-Instruct Base-Mix SFT (P2, MI300X bootcamp)",
        base_model=cfg["model"]["name"],
        summary=summary,
        extras={
            "LoRA rank": cfg["lora"]["r"],
            "Target modules": cfg["lora"]["target_modules"],
            "Epochs": cfg["train"]["num_train_epochs"],
            "Effective batch size": (
                cfg["train"]["per_device_train_batch_size"]
                * cfg["train"]["gradient_accumulation_steps"]
            ),
            "Learning rate": cfg["train"]["learning_rate"],
            "Final train loss": f"{result.training_loss:.4f}",
            "Wall-clock": f"{train_min / 60:.2f} hr",
        },
    )

    repo_id = args.hub_repo_id or cfg["hub"].get("repo_id")
    if cfg["hub"].get("push") and not args.no_push:
        if not repo_id:
            print("[p2] hub.push=true but no repo_id given — skipping push.")
        else:
            print(f"[p2] pushing adapter → {repo_id}")
            push_to_hub(
                local_dir=adapter_dir,
                repo_id=repo_id,
                private=cfg["hub"].get("private", False),
                commit_message="P2: Llama-3.1-70B base-mix LoRA SFT adapter",
            )
            if merged_dir:
                merged_repo = f"{repo_id}-merged"
                print(f"[p2] pushing merged bf16 → {merged_repo}")
                push_to_hub(
                    local_dir=merged_dir,
                    repo_id=merged_repo,
                    private=cfg["hub"].get("private", False),
                    commit_message="P2: Llama-3.1-70B base-mix merged bf16",
                )

    (out_dir / "RUN_OK").write_text(
        f"loss={result.training_loss:.4f}\nminutes={train_min:.2f}\n"
    )
    print("[p2] done. Day 1 complete. Proceed to data gen / P3.")


if __name__ == "__main__":
    main()
