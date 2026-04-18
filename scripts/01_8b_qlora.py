"""
01_8b_qlora.py — P1: Llama-3.1-8B QLoRA warm-up on Alpaca-cleaned.

Day 1, ~1.5 GPU-hr. End-to-end smoke of the real pipeline (Unsloth + TRL SFT)
on a small model before committing 6 hr to the 70B run.

Usage:
  python scripts/01_8b_qlora.py \
      --config configs/p1_8b_qlora.yaml \
      --hub_repo_id <user>/llama-3.1-8b-alpaca-qlora
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
    ap.add_argument("--config", default="configs/p1_8b_qlora.yaml")
    ap.add_argument("--hub_repo_id", default=None,
                    help="Override config; e.g. <user>/llama-3.1-8b-alpaca-qlora")
    ap.add_argument("--no_push", action="store_true",
                    help="Skip HF Hub push even if config says push=true")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="Override num_train_epochs (debug)")
    return ap.parse_args()


def build_dataset(tokenizer, cfg):
    from datasets import load_dataset

    ds = load_dataset(cfg["data"]["dataset"], split=cfg["data"]["split"])
    ds = ds.shuffle(seed=cfg["data"].get("shuffle_seed", 42))

    def fmt(ex):
        instr = ex["instruction"].strip()
        inp = (ex.get("input") or "").strip()
        out = ex["output"].strip()
        user = instr if not inp else f"{instr}\n\n{inp}"
        msgs = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": out},
        ]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

    return ds.map(fmt, remove_columns=ds.column_names, num_proc=4)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = setup_env_dirs()
    set_global_seed(cfg.get("seed", 42))

    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[p1] output dir: {out_dir}")
    print(f"[p1] scratch:    {paths['scratch']}")

    from unsloth import FastLanguageModel  # import early so it can patch
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    print(f"[p1] loading {cfg['model']['name']} (4bit={cfg['model']['load_in_4bit']})")
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

    print("[p1] building dataset")
    train_ds = build_dataset(tokenizer, cfg)
    print(f"[p1] train examples: {len(train_ds):,}")

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

    print("[p1] training...")
    result = trainer.train()
    train_min = (time.time() - t0) / 60.0
    print(f"[p1] training done in {train_min:.1f} min, loss={result.training_loss:.4f}")

    # Save adapter
    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[p1] adapter saved → {adapter_dir}")

    # Vibe check
    print("[p1] vibe check:")
    FastLanguageModel.for_inference(trainer.model)
    samples = [
        "Explain why the sky is blue in one short paragraph.",
        "Write a Python one-liner that returns the second-largest element of a list.",
        "Summarize the plot of Hamlet in two sentences.",
    ][: cfg["eval"].get("vibe_check_prompts", 3)]
    vibe_path = out_dir / "vibe_check.jsonl"
    with vibe_path.open("w") as f:
        for q in samples:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
            out = trainer.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            print(f"  Q: {q}\n  A: {ans.strip()[:200]}\n")
            f.write(json.dumps({"q": q, "a": ans.strip()}) + "\n")

    # Optionally also save merged fp16 (cheap for 8B)
    merged_dir = None
    if cfg["hub"].get("push_merged"):
        merged_dir = out_dir / "merged"
        print(f"[p1] saving merged fp16 → {merged_dir}")
        trainer.model.save_pretrained_merged(
            str(merged_dir), tokenizer, save_method="merged_16bit"
        )

    # Model card
    summary = (
        f"QLoRA fine-tune (rank={cfg['lora']['r']}, alpha={cfg['lora']['alpha']}) of "
        f"`{cfg['model']['name']}` on `{cfg['data']['dataset']}` — "
        f"{cfg['train']['num_train_epochs']} epochs, "
        f"final loss {result.training_loss:.4f}, {train_min:.1f} min on MI300X."
    )
    write_model_card(
        adapter_dir,
        title="Llama-3.1-8B Alpaca QLoRA (P1, MI300X bootcamp)",
        base_model=cfg["model"]["name"],
        summary=summary,
        extras={
            "LoRA rank": cfg["lora"]["r"],
            "Epochs": cfg["train"]["num_train_epochs"],
            "Effective batch size": (
                cfg["train"]["per_device_train_batch_size"]
                * cfg["train"]["gradient_accumulation_steps"]
            ),
            "Learning rate": cfg["train"]["learning_rate"],
            "Final train loss": f"{result.training_loss:.4f}",
        },
    )
    if merged_dir:
        write_model_card(
            merged_dir,
            title="Llama-3.1-8B Alpaca QLoRA — merged fp16 (P1)",
            base_model=cfg["model"]["name"],
            summary=summary,
        )

    # Push
    repo_id = args.hub_repo_id or cfg["hub"].get("repo_id")
    if cfg["hub"].get("push") and not args.no_push:
        if not repo_id:
            print("[p1] hub.push=true but no repo_id given — skipping push.")
        else:
            print(f"[p1] pushing adapter → {repo_id}")
            push_to_hub(
                local_dir=adapter_dir,
                repo_id=repo_id,
                private=cfg["hub"].get("private", False),
                commit_message="P1: Llama-3.1-8B Alpaca QLoRA adapter",
            )
            if merged_dir:
                merged_repo = f"{repo_id}-merged"
                print(f"[p1] pushing merged fp16 → {merged_repo}")
                push_to_hub(
                    local_dir=merged_dir,
                    repo_id=merged_repo,
                    private=cfg["hub"].get("private", False),
                    commit_message="P1: Llama-3.1-8B Alpaca merged fp16",
                )

    (out_dir / "RUN_OK").write_text(
        f"loss={result.training_loss:.4f}\nminutes={train_min:.2f}\n"
    )
    print("[p1] done.")


if __name__ == "__main__":
    main()
