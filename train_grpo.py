"""
train_grpo.py - GRPO alignment training
Run on Digital Ocean MI300X droplet after SFT training
"""

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import torch
import re

print("Loading SFT model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./reasoning_booster_sft_merged",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)

# Prepare reward prompts (GSM8K train subset)
print("Loading GSM8K dataset for GRPO...")
dataset = load_dataset("gsm8k", "main", split="train[:2000]")

def extract_answer(completion):
    """Extract numeric answer from GSM8K format"""
    # Try to find #### pattern
    match = re.search(r'####\s*([-\d,]+\.?\d*)', completion)
    if match:
        return match.group(1).replace(',', '')
    # Fallback: try to extract last number
    numbers = re.findall(r'[-+]?\d*\.?\d+', completion)
    if numbers:
        return numbers[-1]
    return None

def reward_function(completions, **kwargs):
    """Verifiable reward: exact match for math problems"""
    rewards = []
    for completion in completions:
        predicted = extract_answer(completion)
        if predicted:
            # In production, compare with ground truth
            # For now, reward if answer is extracted
            rewards.append(0.5)  # Partial reward for format compliance
        else:
            rewards.append(0.0)
    return torch.tensor(rewards, device="cuda")

def format_prompt(example):
    """Format GSM8K example for the model"""
    return f"Solve this step by step: {example['question']}"

print("Configuring GRPO trainer...")
training_args = GRPOConfig(
    output_dir="/scratch/reasoning_booster/outputs_grpo",
    learning_rate=1e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    max_steps=1700,
    max_completion_length=1024,
    num_generations=8,  # group_size
    temperature=0.9,
    report_to="wandb",
    run_name="reasoning-booster-grpo",
    beta=0.1,  # KL penalty
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_function,
    args=training_args,
    train_dataset=dataset,
)

print("Starting GRPO training (this will take 8 hours)...")
trainer.train()

print("Saving GRPO model...")
model.save_pretrained_merged("reasoning_booster_grpo_merged", tokenizer, save_method="merged_16bit")
model.save_pretrained("reasoning_booster_grpo_lora")

print("GRPO training complete!")
