"""
04_reasoning_sft.py — P3: continue P2's LoRA on synthetic reasoning/code data.

Day 2, ~6.5 GPU-hr. Reads the JSONL produced by 03_synth_gen.py
(format: one record per line, fields {prompt, response, domain, ...}).

Critically: this does NOT init a new LoRA. It loads the P2 adapter and keeps
training it. Subsequent DPO (P4) and GRPO (P5) stages do the same, so the
final headline model is the result of one continuous LoRA lineage rather
than four independent adapters glued together.

Usage:
  python scripts/04_reasoning_sft.py \
      --config configs/p3_reasoning_sft.yaml \
      --hub_repo_id <user>/qwen2.5-72b-reasoning-booster-sft
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
    ap.add_argument("--config", default="configs/p3_reasoning_sft.yaml")
    ap.add_argument("--hub_repo_id", default=None)
    ap.add_argument("--no_push", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the latest trainer checkpoint in output_dir")
    ap.add_argument("--max_steps", type=int, default=None)
    return ap.parse_args()


def load_synth_jsonl(path: str, tokenizer, max_seq_length: int, seed: int):
    """Load reasoning JSONL in either supported format.

    Accepted row shapes:
      1) {"prompt": "...", "response": "..."}  (old synth-gen format)
      2) {"messages": [{"role": "...", "content": "..."}, ...]}  (current reasoning-mix format)
    """
    from datasets import Dataset

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first with:\n"
            f"  python scripts/03_synth_gen.py "
            f"--merged_model /scratch/finetune/models/p2_merged "
            f"--output {path}"
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
            # Newer pipeline format: pre-built chat messages
            if isinstance(rec.get("messages"), list) and rec["messages"]:
                msgs = []
                for m in rec["messages"]:
                    if not isinstance(m, dict):
                        continue
                    role = m.get("role")
                    content = (m.get("content") or "").strip()
                    if role in {"system", "user", "assistant"} and content:
                        msgs.append({"role": role, "content": content})
                # Must include at least one user and assistant turn for SFT
                has_user = any(m["role"] == "user" for m in msgs)
                has_asst = any(m["role"] == "assistant" for m in msgs)
                if has_user and has_asst:
                    rows.append({"messages": msgs, "domain": rec.get("domain", "")})
                continue

            # Legacy synth format: prompt/response pair
            prompt = (rec.get("prompt") or "").strip()
            response = (rec.get("response") or "").strip()
            if prompt and response:
                rows.append({
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                    "domain": rec.get("domain", ""),
                })

    print(f"[p3] loaded {len(rows):,} synth rows from {path}")

    if not rows:
        raise ValueError(
            f"[p3] loaded 0 rows from {path}. "
            f"Expected either 'messages' rows or 'prompt/response' rows."
        )

    def render(ex):
        return {"text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}

    ds = Dataset.from_list(rows).shuffle(seed=seed)
    ds = ds.map(render, remove_columns=ds.column_names, num_proc=4)

    def short_enough(ex):
        return len(tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
                   ) <= max_seq_length

    before = len(ds)
    ds = ds.filter(short_enough, num_proc=4)
    print(f"[p3] post length-filter: {len(ds):,}/{before:,}")
    return ds


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = setup_env_dirs()
    set_global_seed(cfg.get("seed", 42))

    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[p3] output dir: {out_dir}")
    print(f"[p3] scratch:    {paths['scratch']}")

    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    print(f"[p3] loading {cfg['model']['name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=cfg["model"].get("dtype"),
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    resume_adapter = cfg["model"].get("resume_adapter")
    if resume_adapter:
        # Continue training the P2 adapter. Load it as a PEFT adapter and
        # mark it trainable.
        from peft import PeftModel

        print(f"[p3] CONTINUING training of adapter at {resume_adapter}")
        model = PeftModel.from_pretrained(
            model, resume_adapter, is_trainable=True
        )
        # Sanity: report trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[p3] trainable params: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.3f}%)")
    else:
        print("[p3] no resume_adapter — initializing FRESH LoRA "
              "(this breaks the P2→P3→P4→P5 lineage; intentional?)")
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

    print("[p3] building dataset")
    train_ds = load_synth_jsonl(
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
        model=model, tokenizer=tokenizer, train_dataset=train_ds,
        args=sft_cfg, callbacks=[cb] if cb else [],
    )

    print(f"[p3] training... (wall-clock cap: "
          f"{cfg['train'].get('max_runtime_minutes', 'none')} min)")
    result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    train_min = (time.time() - t0) / 60.0
    print(f"[p3] done in {train_min:.1f} min ({train_min / 60:.2f} hr), "
          f"loss={result.training_loss:.4f}")

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[p3] adapter saved → {adapter_dir}")

    summary = (
        f"Continued LoRA (rank {cfg['lora']['r']}) of `{cfg['model']['name']}` "
        f"starting from the P2 base-mix SFT adapter, fine-tuned on a 15k "
        f"synthetic reasoning/code mix generated by `03_synth_gen.py` "
        f"(GSM8K-train + MBPP + general seeds, evol-instruct mutations, "
        f"MinHash-deduped vs the P2 base mix). "
        f"{cfg['train']['num_train_epochs']} epochs, "
        f"final loss {result.training_loss:.4f}, "
        f"{train_min / 60:.2f} hr on a single MI300X."
    )
    write_model_card(
        adapter_dir,
        title="Qwen2.5-72B Reasoning-Booster SFT (P3, MI300X bootcamp)",
        base_model=cfg["model"]["name"],
        summary=summary,
        extras={
            "Stage": "P3 (continuation of P2 SFT)",
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
            print("[p3] hub.push=true but no repo_id given — skipping push.")
        else:
            print(f"[p3] pushing adapter → {repo_id}")
            push_to_hub(
                local_dir=adapter_dir, repo_id=repo_id,
                private=cfg["hub"].get("private", False),
                commit_message="P3: Reasoning-Booster SFT adapter",
            )

    (out_dir / "RUN_OK").write_text(
        f"loss={result.training_loss:.4f}\nminutes={train_min:.2f}\n"
    )
    print("[p3] done. Day 2 complete. Proceed to preference-pair gen / P4.")


if __name__ == "__main__":
    main()
