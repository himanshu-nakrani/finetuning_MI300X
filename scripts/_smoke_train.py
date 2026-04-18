"""
_smoke_train.py — 100-step QLoRA smoke train on Llama-3.1-8B.
Purpose: catch ROCm/driver/HF/disk issues BEFORE committing real GPU hours.
Invoked by scripts/00_preflight.sh. Not a real training run.

Tiny: 500 Alpaca samples, rank=8, bs=2, max_steps=100. Should take <30 min on MI300X.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--n_samples", type=int, default=500)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Import lazily so the preflight shell can surface package-level errors first.
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    print(f"[smoke] loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=1024,
        dtype=None,           # auto (bf16 on MI300X)
        load_in_4bit=True,    # QLoRA
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    print("[smoke] loading dataset: yahma/alpaca-cleaned (tiny slice)")
    ds = load_dataset("yahma/alpaca-cleaned", split=f"train[:{args.n_samples}]")

    def fmt(ex):
        instr = ex["instruction"]
        inp = ex.get("input", "")
        out = ex["output"]
        user = instr if not inp else f"{instr}\n\n{inp}"
        msgs = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": out},
        ]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

    ds = ds.map(fmt, remove_columns=ds.column_names)

    cfg = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        report_to="none",
        max_seq_length=1024,
        dataset_text_field="text",
        warmup_steps=5,
        optim="adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=cfg,
    )

    print(f"[smoke] training for {args.max_steps} steps...")
    result = trainer.train()

    dt = time.time() - t0
    print(f"[smoke] done in {dt/60:.1f} min")
    print(f"[smoke] final loss: {result.training_loss:.4f}")

    # Quick generation sanity
    FastLanguageModel.for_inference(model)
    msgs = [{"role": "user", "content": "Say hello in one short sentence."}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    print("[smoke] sample generation:")
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    # Write a tiny summary
    (Path(args.output_dir) / "SMOKE_OK").write_text(
        f"loss={result.training_loss:.4f}\nmin={dt/60:.1f}\nsteps={args.max_steps}\n"
    )


if __name__ == "__main__":
    main()
