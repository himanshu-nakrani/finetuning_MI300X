"""
ab_run.py — generic runner for the Project-B ablation (Llama-3.1-8B on the
reasoning mix). Reads a config with `mode: lora` or `mode: full_ft` and
trains accordingly. Single script so all three ablations are an apples-to-
apples comparison (same data, same prompt template, same eval path).

Usage:
  python scripts/ab_run.py --config configs/ab_rank16.yaml \
      --hub_repo_id <user>/llama-3.1-8b-reasoning-qlora-r16

  python scripts/ab_run.py --config configs/ab_rank64.yaml \
      --hub_repo_id <user>/llama-3.1-8b-reasoning-qlora-r64

  python scripts/ab_run.py --config configs/ab_fullft.yaml \
      --hub_repo_id <user>/llama-3.1-8b-reasoning-fullft
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
    ap.add_argument("--config", required=True)
    ap.add_argument("--hub_repo_id", default=None)
    ap.add_argument("--no_push", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None)
    return ap.parse_args()


def load_jsonl_dataset(path: str, tokenizer, max_seq_length: int, seed: int):
    from datasets import Dataset

    rows = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            msgs = rec.get("messages")
            if msgs:
                rows.append({"messages": msgs})
            elif rec.get("prompt") and rec.get("response"):
                rows.append({"messages": [
                    {"role": "user", "content": rec["prompt"]},
                    {"role": "assistant", "content": rec["response"]},
                ]})
    print(f"[ab] loaded {len(rows):,} rows from {path}")

    def render(ex):
        return {"text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}

    ds = Dataset.from_list(rows).shuffle(seed=seed)
    ds = ds.map(render, remove_columns=ds.column_names, num_proc=4)
    def short(ex):
        return len(tokenizer(ex["text"],
                             add_special_tokens=False)["input_ids"]
                   ) <= max_seq_length
    before = len(ds)
    ds = ds.filter(short, num_proc=4)
    print(f"[ab] post length-filter: {len(ds):,}/{before:,}")
    return ds


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = setup_env_dirs()
    set_global_seed(cfg.get("seed", 42))

    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = cfg.get("mode", "lora")
    print(f"[ab] mode={mode} output={out_dir}")

    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    print(f"[ab] loading {cfg['model']['name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=cfg["model"].get("dtype"),
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    if mode == "lora":
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
    elif mode == "full_ft":
        # Full fine-tune: do NOT call get_peft_model. Unsloth still gives us
        # its fused attention kernels. All params trainable.
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(f"unknown mode: {mode}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[ab] trainable: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.3f}%)")

    train_ds = load_jsonl_dataset(
        cfg["data"]["jsonl_path"], tokenizer,
        cfg["model"]["max_seq_length"], cfg["data"].get("shuffle_seed", 42),
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
        model=model, tokenizer=tokenizer, train_dataset=train_ds,
        args=sft_cfg, callbacks=[cb] if cb else [],
    )

    print(f"[ab] training... (wallclock cap "
          f"{cfg['train'].get('max_runtime_minutes')} min)")
    result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    train_min = (time.time() - t0) / 60.0
    print(f"[ab] done in {train_min:.1f} min, loss={result.training_loss:.4f}")

    if mode == "lora":
        artifact = out_dir / "adapter"
        trainer.model.save_pretrained(str(artifact))
    else:
        artifact = out_dir / "merged"
        trainer.model.save_pretrained(
            str(artifact), safe_serialization=True, max_shard_size="5GB"
        )
    tokenizer.save_pretrained(str(artifact))
    print(f"[ab] artifact saved → {artifact}")

    summary = (
        f"{mode.upper()} fine-tune of `{cfg['model']['name']}` on a 10k "
        f"reasoning mix (OpenMathInstruct-2 + CodeFeedback, deduped + "
        f"decontaminated). Part of Project B ablation "
        f"(rank-16 vs rank-64 vs full-FT). Final loss "
        f"{result.training_loss:.4f}, {train_min:.1f} min on MI300X."
    )
    write_model_card(
        artifact,
        title=f"Llama-3.1-8B Reasoning {mode.upper()} (Project B ablation)",
        base_model=cfg["model"]["name"],
        summary=summary,
        extras={
            "Mode": mode,
            "LoRA rank": cfg.get("lora", {}).get("r", "—"),
            "Epochs": cfg["train"]["num_train_epochs"],
            "Effective batch size": (
                cfg["train"]["per_device_train_batch_size"]
                * cfg["train"]["gradient_accumulation_steps"]
            ),
            "Learning rate": cfg["train"]["learning_rate"],
            "Final train loss": f"{result.training_loss:.4f}",
            "Wall-clock": f"{train_min:.1f} min",
        },
    )

    repo_id = args.hub_repo_id or cfg["hub"].get("repo_id")
    if cfg["hub"].get("push") and not args.no_push and repo_id:
        print(f"[ab] pushing → {repo_id}")
        push_to_hub(local_dir=artifact, repo_id=repo_id,
                    private=cfg["hub"].get("private", False),
                    commit_message=f"Project B ablation: {cfg['run_name']}")

    (out_dir / "RUN_OK").write_text(
        f"loss={result.training_loss:.4f}\nminutes={train_min:.2f}\n"
        f"mode={mode}\n"
    )
    print("[ab] done.")


if __name__ == "__main__":
    main()
