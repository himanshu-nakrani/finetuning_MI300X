"""
03_generate_synth_data_unsloth.py - Generate synthetic reasoning data using Unsloth
Alternative to vLLM for ROCm systems where vLLM installation is problematic
"""

from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import json
import torch
from tqdm import tqdm
import random

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.3-70B-Instruct",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,  # Use 4-bit to fit in memory with generation
)

print("Model loaded. Starting generation...")

# Generation prompt
cot_prompt = """Create a complex logical reasoning problem with a detailed Chain-of-Thought solution.
The problem should cover one of these topics: math, logic puzzles, code reasoning, or multi-step inference.
Each problem must include:
- Clear problem statement
- Step-by-step reasoning
- Final answer

Return ONLY valid JSON with this format:
{
  "question": "...",
  "reasoning": "...",
  "answer": "..."
}

Problem:"""

def generate_sample():
    """Generate one synthetic reasoning sample"""
    messages = [{"role": "user", "content": cot_prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON from generated text
    try:
        # Find JSON in the response
        start = generated.find("{")
        end = generated.rfind("}") + 1
        if start != -1 and end > start:
            json_str = generated[start:end]
            data = json.loads(json_str)
            return data
    except:
        pass
    
    return None

# Generate samples
num_samples = 100  # Start with 100, can increase
samples = []

print(f"Generating {num_samples} synthetic reasoning samples...")
for i in tqdm(range(num_samples)):
    sample = generate_sample()
    if sample:
        samples.append(sample)
    
    # Save progress every 10 samples
    if (i + 1) % 10 == 0:
        with open("data/synthetic_reasoning.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"\nSaved {len(samples)} samples to data/synthetic_reasoning.jsonl")

# Final save
with open("data/synthetic_reasoning.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")

print(f"\nGeneration complete! Generated {len(samples)} valid samples.")
print(f"Saved to data/synthetic_reasoning.jsonl")
