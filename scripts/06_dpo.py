"""
06_dpo.py — P4: DPO on 3k preference pairs (Project A headline, ~4.0 GPU-hr).

Continues the P3 LoRA with TRL's DPOTrainer. Critical config choices:
  - LR 5e-7 (not 2e-4!) — DPO destabilizes fast at higher LRs.
  - beta 0.1 — TRL's default; smaller = further from reference, larger = closer.
  - 1 epoch — DPO benchmarks saturate within 1 epoch; 2+ often hurts.
  - bf16 throughout; no mixed-precision tricks needed on MI300X.

Usage:
  python scripts/06_dpo.py \
      --config configs/p4_dpo.yaml \
      --hub_repo_id <user>/qwen2.5-72b-reasoning-dpo
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
    ap.add_argument("--config", default="configs/p4_dpo.yaml")
    ap.add_argument("--hub_repo_id", default=None)
    ap.add_argument("--no_push", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None)
    return ap.parse_args()


def load_pref_dataset(path: str, seed: int):
    from datasets import Dataset
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if all(r.get(k) for k in ("prompt", "chosen", "rejected")):
                rows.append({"prompt": r["prompt"],
                             "chosen": r["chosen"],
                             "rejected": r["rejected"]})
    print(f"[p4] loaded {len(rows):,} preference pairs from {path}")
    return Dataset.from_list(rows).shuffle(seed=seed)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = setup_env_dirs()
    set_global_seed(cfg.get("seed", 42))

    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[p4] output dir: {out_dir}")

    # Unsloth FIRST (patches attention etc. before trl imports)
    from unsloth import FastLanguageModel
    from trl import DPOConfig, DPOTrainer
    from peft import PeftModel

    t0 = time.time()
    print(f"[p4] loading {cfg['model']['name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=cfg["model"].get("dtype"),
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    resume_adapter = cfg["model"].get("resume_adapter")
    if resume_adapter:
        print(f"[p4] CONTINUING from P3 adapter at {resume_adapter}")
        model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
    else:
        print("[p4] no resume_adapter — initializing FRESH LoRA "
              "(breaks P2→P3→P4 lineage; likely unintended)")
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

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[p4] trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.3f}%)")

    train_ds = load_pref_dataset(cfg["data"]["jsonl_path"],
                                 cfg["data"]["shuffle_seed"])

    dpo_cfg = DPOConfig(
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
        seed=cfg.get("seed", 42),
        beta=cfg["dpo"]["beta"],
        loss_type=cfg["dpo"]["loss_type"],
        max_length=cfg["model"]["max_seq_length"],
        max_prompt_length=cfg["model"]["max_prompt_length"],
        max_steps=args.max_steps if args.max_steps else -1,
    )

    cb = make_wallclock_callback(cfg["train"].get("max_runtime_minutes"))
    # TRL's DPOTrainer uses the model itself as the reference policy when
    # ref_model=None *and* the model has PEFT adapters — it disables the
    # adapters to compute reference log-probs. That's exactly what we want.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_cfg,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        callbacks=[cb] if cb else [],
    )

    print(f"[p4] training DPO... (beta={cfg['dpo']['beta']}, "
          f"wallclock cap {cfg['train'].get('max_runtime_minutes')} min)")
    result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    train_min = (time.time() - t0) / 60.0
    print(f"[p4] done in {train_min:.1f} min ({train_min / 60:.2f} hr), "
          f"loss={result.training_loss:.4f}")

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[p4] adapter saved → {adapter_dir}")

    summary = (
        f"DPO continuation (beta={cfg['dpo']['beta']}, LR={cfg['train']['learning_rate']}) "
        f"of the P3 reasoning-SFT adapter on a 3k preference-pair dataset. "
        f"Preferences scored by verifiable correctness (math exact-match, "
        f"code unit-tests) with a length-based fallback for general prompts. "
        f"1 epoch, final loss {result.training_loss:.4f}, {train_min / 60:.2f} hr "
        f"on MI300X."
    )
    write_model_card(
        adapter_dir,
        title="Qwen2.5-72B Reasoning DPO (P4, MI300X bootcamp)",
        base_model=cfg["model"]["name"],
        summary=summary,
        extras={
            "Stage": "P4 (P2 → P3 → P4 lineage)",
            "Beta": cfg["dpo"]["beta"],
            "Learning rate": cfg["train"]["learning_rate"],
            "Final train loss": f"{result.training_loss:.4f}",
            "Wall-clock": f"{train_min / 60:.2f} hr",
        },
    )

    repo_id = args.hub_repo_id or cfg["hub"].get("repo_id")
    if cfg["hub"].get("push") and not args.no_push and repo_id:
        print(f"[p4] pushing adapter → {repo_id}")
        push_to_hub(local_dir=adapter_dir, repo_id=repo_id,
                    private=cfg["hub"].get("private", False),
                    commit_message="P4: Reasoning DPO adapter")

    (out_dir / "RUN_OK").write_text(
        f"loss={result.training_loss:.4f}\nminutes={train_min:.2f}\n"
    )
    print("[p4] done. Headline model ready for eval sweep.")


if __name__ == "__main__":
    main()
