# MI300X Reasoning Booster - Deployment Instructions

## Prerequisites

- Access to Digital Ocean MI300X droplet (192 GB VRAM)
- SSH access to the droplet
- Hugging Face account with write access
- (Optional) Weights & Biases account for training logs

## Step 1: SSH into the Droplet

```bash
# SSH into your Digital Ocean MI300X droplet
ssh root@<your-droplet-ip>

# Or if you have SSH keys configured
ssh <your-username>@<your-droplet-ip>
```

## Step 2: Transfer Files to Droplet

From your local machine:

```bash
# Copy all scripts to the droplet
scp -r /Users/himanshu/Git/finetuning_MI300X/* root@<your-droplet-ip>:/scratch/reasoning_booster/

# Or use rsync for better performance with large files
rsync -avz /Users/himanshu/Git/finetuning_MI300X/ root@<your-droplet-ip>:/scratch/reasoning_booster/
```

## Step 3: Run Setup Script

On the droplet:

```bash
cd /scratch/reasoning_booster
chmod +x *.sh
bash install_mi300x.sh
```

This will:
- Create virtual environment at `.venv`
- Install PyTorch for ROCm 6.2 (verified working configuration)
- Install Unsloth (AMD native)
- Install TRL, PEFT, and other tools
- Set BNB_ROCM_ARCH environment variable
- Verify installation with ROCm-specific checks

**Note:** The script uses the proven `requirements-mi300x.txt` configuration that specifically prevents CUDA package conflicts.

## Step 4: Authenticate

```bash
# Hugging Face
huggingface-cli login
# Enter your token when prompted

# Weights & Biases (optional)
wandb login
```

## Step 5: Install vLLM (ROCm version)

The PyPI vLLM is CUDA-only. Install the ROCm version:

```bash
cd /scratch/reasoning_booster
source .venv/bin/activate
pip uninstall -y vllm || true
pip install -r requirements-mi300x-vllm.txt
```

Verify torch wasn't replaced:
```bash
python -c "import torch; assert '+rocm' in torch.__version__, torch.__version__; print('OK')"
```

## Step 6: Start vLLM Server (Terminal 1)

```bash
cd /scratch/reasoning_booster
source .venv/bin/activate
bash 02_start_vllm.sh
```

Keep this terminal open. The server will run on port 8001.

## Step 7: Generate Synthetic Data (Terminal 2)

Open a new SSH session or use `tmux`/`screen`:

```bash
cd /scratch/reasoning_booster
source .venv/bin/activate
bash 03_generate_synth_data.sh
```

This will take 4-5 hours. You can monitor progress in the logs.

## Step 7: Train LoRA SFT

After data generation completes:

```bash
cd /scratch/reasoning_booster
source .venv/bin/activate
pyth oornft2/finstupyMI300X
```

This will take 10-12 hours. The model will be saved to:
- `reasoning_booster_sft_merged/` (merged 16-bit)
- `reasoning_booster_lora/` (LoRA adapter)

## Step 10: Train GRPO

After SFT training completes:

```bash
cd /scratch/reasoning_booster
source .venv/bin/activate
python train_grpo.py

```

This will take 8 hours. The model will be saved to:
- `reasoning_booster_grpo_merged/` (merged 16-bit)
- `reasoning_booster_grpo_lora/` (LoRA adapter)

## Step 11: Evaluate and Push

After GRPO training completes:

```bash
cd /scratch/reasoning_booster
source .venv/bin/activate
bash5ooelft2/finatupush.MI300X
```

This will:
- Run GSM8K evaluation
- Run HumanEval evaluation
- Compare with baseline
- Push models and dataset to Hugging Face

## Step 12: Create Gradio Demo (Optional)

On the droplet or locally:

```bash
# Update the MODEL_PATH in demo_app.py with your HF username
# Then run:
pip install gradio
python demo_app.py
```

Access at: `http://<droplet-ip>:7860`

## Step 10: Update GitHub

From your local machine:

```bash
cd /Users/himanshu/Git/finetuning_MI300X
git add .
git commit -m "Add MI300X Reasoning Booster pipeline"
git push
```

## Using tmux for Long-Running Sessions

For long-running tasks, use tmux to keep sessions alive:

```bash
# Install tmux if not present
sudo apt-get install tmux  # or yum install tmux

# Create a new session
tmux new -s training

# Detach from session (Ctrl+b, then d)

# Reattach to session
tmux attach -t training

# List sessions
tmux ls
```

## Monitoring

```bash
# Monitor VRAM usage
watch -n 2 rocm-smi --showmeminfo vram

# Monitor disk usage
df -h /scratch

# Check running processes
ps aux | grep python

# View training logs
tail -f /scratch/reasoning_booster/outputs_*/trainer.log
```

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size` in training scripts
- Reduce `max_seq_length`
- Ensure gradient checkpointing is enabled (already on)

### vLLM Server Not Responding
- Check if port 8001 is in use: `netstat -tlnp | grep 8001`
- Restart the server
- Check ROCm installation: `amd-smi`

### Hugging Face Push Fails
- Verify you're logged in: `huggingface-cli whoami`
- Check your token has write access
- Ensure model name doesn't already exist (or use `repo_id` parameter)

### Synthetic Data Generation Fails
- Ensure vLLM server is running on port 8001
- Check `config.yaml` paths are correct
- Verify `synthetic-data-kit` is installed correctly

## Estimated Timeline

- Setup: 1 hour
- Synthetic data: 5-7 hours
- SFT training: 10-12 hours
- GRPO training: 8 hours
- Eval & push: 2 hours
- **Total: ~28 hours**

## Cost Considerations

Digital Ocean MI300X droplets are billed hourly. At approximately $X/hour, the total cost will be ~$28X for this project.

## Next Steps After Completion

1. Review benchmark results
2. Update README with actual metrics
3. Create Hugging Face model card with details
4. Deploy Gradio demo to HF Spaces
5. Write LinkedIn post with results
6. Update resume with project details
