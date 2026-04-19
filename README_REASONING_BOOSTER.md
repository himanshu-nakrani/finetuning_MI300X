# MI300X Reasoning Booster

Fine-tuned Llama 3.3 70B on single AMD MI300X (192 GB VRAM) using synthetic Chain-of-Thought data → LoRA SFT → GRPO alignment.

## Results

| Model | GSM8K | HumanEval |
|-------|-------|-----------|
| Llama 3.3 70B Instruct | XX.X% | XX.X% |
| Reasoning Booster (Ours) | XX.X% | XX.X% |
| **Improvement** | **+X.X%** | **+X.X%** |

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("yourusername/mi300x-reasoning-booster")
tokenizer = AutoTokenizer.from_pretrained("yourusername/mi300x-reasoning-booster")

prompt = "Solve this step by step: What is 15% of 240?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

## Training Pipeline

### 1. Environment Setup
```bash
bash 01_reasoning_booster_setup.sh
```

### 2. Start vLLM Server (Terminal 1)
```bash
bash 02_start_vllm.sh
```

### 3. Generate Synthetic Data (Terminal 2)
```bash
bash 03_generate_synth_data.sh
```

### 4. Train LoRA SFT
```bash
python train_sft.py
```

### 5. Train GRPO
```bash
python train_grpo.py
```

### 6. Evaluate and Push to Hugging Face
```bash
bash 05_eval_and_push.sh
```

## Training Configuration

**SFT (LoRA):**
- Model: Llama 3.3 70B Instruct
- Rank: 64, Alpha: 64
- Batch size: 1 (effective: 128 with grad_accum)
- Epochs: 1
- Learning rate: 1e-4
- Precision: bf16
- Data: 5k synthetic reasoning pairs

**GRPO:**
- Base: SFT model
- Group size: 8
- Steps: 1700
- Learning rate: 1e-6
- KL beta: 0.1
- Reward: Verifiable math matching

## Hardware

- **GPU:** AMD MI300X (192 GB VRAM)
- **CPU:** 20 vCPU
- **RAM:** 240 GB
- **Storage:** 5 TB NVMe scratch
- **Training time:** ~28 hours

## Dataset

Synthetic reasoning dataset available at: `yourusername/reasoning-booster-synthetic-data`

- 5,000 Chain-of-Thought reasoning pairs
- Generated using vLLM + self-instruct
- Curated with quality threshold 7.0
- Topics: math, logic, code reasoning

## Demo

Try the Gradio demo: [Link to HF Space or local instructions]

```bash
pip install gradio transformers torch
python demo_app.py
```

## Resume/LinkedIn

**Resume bullet:**
"Fine-tuned Llama 3.3 70B on single AMD MI300X (192 GB VRAM) using Unsloth + synthetic Chain-of-Thought data + GRPO alignment. Built end-to-end pipeline achieving [X%] gain on GSM8K and [Y%] on HumanEval over baseline."

**LinkedIn post:**
"I just spent 28 hours on an AMD MI300X and shipped a fully fine-tuned Llama 3.3 70B reasoning model. The pipeline: synthetic data generation → LoRA SFT → GRPO alignment. Results: [X%] improvement on GSM8K, [Y%] on HumanEval. Full repo + configs + benchmarks below. #AI #FineTuning #MI300X"

## Monitoring

```bash
# Monitor VRAM
watch -n 2 rocm-smi --showmeminfo vram

# Monitor training logs
tail -f /scratch/reasoning_booster/outputs_*/trainer.log

# Check disk usage
df -h /scratch
```

## License

Model weights follow the Llama 3.3 license.
