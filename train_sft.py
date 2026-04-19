"""
train_sft.py - LoRA SFT training on synthetic reasoning data
Run on Digital Ocean MI300X droplet
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
import torch

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.3-70B-Instruct",
    max_seq_length=2048,
    dtype=None,  # auto bf16 on ROCm
    load_in_4bit=False,  # full 16-bit LoRA with 192 GB
)

print("Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

print("Loading dataset...")
# Load dataset - adjust path if needed
try:
    dataset = load_dataset("json", data_files="data/final/train.jsonl", split="train")
except:
    # Fallback: try to find the data file
    import glob
    data_files = glob.glob("data/final/*.jsonl")
    if data_files:
        dataset = load_dataset("json", data_files=data_files[0], split="train")
    else:
        raise FileNotFoundError("No data files found in data/final/")

dataset = standardize_sharegpt(dataset)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

print(f"Dataset size: {len(dataset)}")

print("Configuring trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=128,  # effective batch 128
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_steps=100,
        optim="adamw_8bit",
        output_dir="/scratch/reasoning_booster/outputs_sft",
        bf16=True,
        report_to="wandb",
        run_name="reasoning-booster-sft",
    ),
)

trainer = train_on_responses_only(trainer)
FastLanguageModel.for_training(model)

print("Starting training (this will take 10-12 hours)...")
trainer.train()

print("Saving models...")
model.save_pretrained_merged("reasoning_booster_sft_merged", tokenizer, save_method="merged_16bit")
model.save_pretrained("reasoning_booster_lora")

print("SFT training complete!")
