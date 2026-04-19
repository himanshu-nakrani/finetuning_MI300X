"""
demo_app.py - Gradio demo for Reasoning Booster model
Can be run locally or deployed to Hugging Face Spaces
"""

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

# Load model (adjust path for HF Hub)
MODEL_PATH = "yourusername/mi300x-reasoning-booster"  # Change this

print(f"Loading model from {MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def solve_reasoning(problem):
    """Solve a reasoning problem with Chain-of-Thought"""
    prompt = f"Solve this step by step: {problem}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def extract_answer(response):
    """Extract the final answer from the response"""
    match = re.search(r'####\s*([-\d,]+\.?\d*)', response)
    if match:
        return match.group(1)
    return "Answer not found in expected format"

# Create Gradio interface
with gr.Blocks(title="MI300X Reasoning Booster") as demo:
    gr.Markdown("# MI300X Reasoning Booster - Llama 3.3 70B")
    gr.Markdown("""
    Fine-tuned on AMD MI300X (192 GB VRAM) using:
    - Synthetic Chain-of-Thought data (5k examples)
    - LoRA SFT (16-bit, rank=64)
    - GRPO alignment with verifiable rewards
    """)
    
    with gr.Row():
        with gr.Column():
            problem_input = gr.Textbox(
                lines=5,
                placeholder="Enter a reasoning problem (e.g., math, logic puzzle, code reasoning)...",
                label="Problem"
            )
            submit_btn = gr.Button("Solve", variant="primary")
        
        with gr.Column():
            solution_output = gr.Textbox(
                lines=10,
                label="Solution",
                placeholder="Solution will appear here..."
            )
            answer_output = gr.Textbox(
                label="Extracted Answer",
                placeholder="Final answer will appear here..."
            )
    
    submit_btn.click(
        solve_reasoning,
        inputs=problem_input,
        outputs=solution_output
    )
    
    # Also update answer when solution changes
    solution_output.change(
        extract_answer,
        inputs=solution_output,
        outputs=answer_output
    )
    
    gr.Examples(
        examples=[
            ["If a baker has 24 cupcakes and sells 3/4 of them, how many are left?"],
            ["A train travels at 60 mph for 2.5 hours. How far does it travel?"],
            ["What is the sum of the first 100 positive integers?"],
        ],
        inputs=problem_input
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
