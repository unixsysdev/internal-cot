# Coconut Implementation for Qwen3-0.6B

This directory contains an adapted implementation of Coconut (Chain of Continuous Thought) for the Qwen3-0.6B model. The implementation follows the paper "Training Large Language Models to Reason in a Continuous Latent Space" (arXiv:2412.06769).

## Overview

Coconut enables LLMs to reason in a continuous latent space instead of explicitly generating reasoning steps as text tokens. The key innovation is:

1. **Continuous Thoughts**: Instead of decoding hidden states to tokens, the model directly feeds hidden states back as input embeddings
2. **Multi-Stage Training**: Progressive curriculum that replaces language reasoning steps with continuous thoughts
3. **BFS-like Reasoning**: The continuous representation can encode multiple alternative next steps, enabling breadth-first search behavior

## Files

### Core Implementation

- `coconut_qwen.py`: Core CoconutQwen model wrapper that implements the continuous thought mechanism
- `dataset_qwen.py`: Dataset processing utilities adapted for Qwen3 tokenizer
- `run_qwen.py`: Training script supporting both single-GPU and distributed training
- `utils.py`: Utility functions (Config class, seed setting)

### Configuration Files

- `args/qwen_cot.yaml`: Stage 0 - Chain-of-Thought baseline training
- `args/qwen_coconut.yaml`: Stages 1+ - Coconut multi-stage training
- `args/qwen_coconut_eval.yaml`: Evaluation configuration
- `args/qwen_prosqa_coconut.yaml`: ProsQA dataset configuration

### Testing & Demo

- `test_qwen_coconut.py`: Test script to verify the implementation
- `demo_train.py`: Quick training and inference demo

## Quick Demo

To see Coconut in action with minimal setup:

```bash
# Setup environment
./setup_env.sh
source coconut_env/bin/activate

# Run the demo (2-5 min on CPU, ~30s on GPU)
python demo_train.py
```

The demo will:
1. Load Qwen3-0.6B and add Coconut special tokens
2. Show inference before training
3. Train for 10 steps on arithmetic problems
4. Show inference after training with loss comparison

Example output:
```
Device: cpu
Training steps: 10
Latent tokens: 2

>>> Before Training:
Q: What is 3 + 4?
A: <thinking output>

Training...
Step 5/10, Loss: 1.6142
Step 10/10, Loss: 2.0395

Initial loss: 8.2619
Final loss: 2.0395
Loss change: -6.2223

>>> After Training:
Q: What is 3 + 4?
A: <generated output>
```

## Environment Setup

### Quick Setup (Recommended)

Use the provided setup script to create a virtual environment with all dependencies:

```bash
# Make the script executable (if not already)
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh

# Activate the environment
source coconut_env/bin/activate

# Verify installation by running tests
python test_qwen_coconut.py
```

### Manual Setup

If you prefer to set up the environment manually:

```bash
# Create virtual environment
python3 -m venv coconut_env

# Activate the environment
source coconut_env/bin/activate  # On Linux/macOS
# OR
coconut_env\Scripts\activate     # On Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Requirements

The `requirements.txt` includes:
- `torch>=2.0.0` - PyTorch deep learning framework
- `numpy>=1.24.0` - Numerical computing
- `transformers>=4.40.0` - Hugging Face Transformers
- `wandb>=0.15.0` - Weights & Biases logging
- `datasets>=2.14.0` - Hugging Face Datasets
- `tqdm>=4.65.0` - Progress bars
- `pyyaml>=6.0` - YAML configuration parsing
- `accelerate>=0.25.0` - Hugging Face Accelerate for distributed training

### Verifying Installation

After setting up the environment, run the test script to verify everything works:

```bash
# Activate environment if not already active
source coconut_env/bin/activate

# Run tests
python test_qwen_coconut.py
```

Expected output:
```
Running Coconut Qwen3-0.6B Tests
============================================================
Test 1: Model Loading
...
Test 2: Coconut Wrapper
...
Test 3: Generation
...
Test 4: Dataset Processing
...
Test 5: Collator
...
============================================================
ALL TESTS PASSED!
============================================================
```

## Usage

### Step 1: Prepare Data

The data should be in JSON format with the following structure:
```json
[
  {
    "question": "John has 5 apples...",
    "steps": ["Step 1: ...", "Step 2: ..."],
    "answer": "10"
  },
  ...
]
```

For GSM8k, use the preprocessing scripts in `preprocessing/`:
```bash
bash preprocessing/gsm_icot.bash
```

### Step 2: Stage 0 - Train CoT Baseline

Train a standard chain-of-thought model that will serve as initialization:

```bash
# Activate environment
source coconut_env/bin/activate

# Single GPU
python run_qwen.py args/qwen_cot.yaml --single-gpu

# Multi-GPU (distributed)
torchrun --nnodes 1 --nproc_per_node 4 run_qwen.py args/qwen_cot.yaml
```

### Step 3: Stages 1-3 - Train Coconut

After Stage 0 completes, update `load_model_path` in `args/qwen_coconut.yaml` to point to the best CoT checkpoint, then:

```bash
# Single GPU
python run_qwen.py args/qwen_coconut.yaml --single-gpu

# Multi-GPU (distributed)
torchrun --nnodes 1 --nproc_per_node 4 run_qwen.py args/qwen_coconut.yaml
```

### Step 4: Evaluate

```bash
python run_qwen.py args/qwen_coconut_eval.yaml --single-gpu
```

### Deactivating the Environment

When you're done working:
```bash
deactivate
```

## Key Configuration Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `c_thought` | Number of continuous thoughts per reasoning step | 1-2 |
| `epochs_per_stage` | Epochs per training stage | 3-5 |
| `max_latent_stage` | Maximum number of latent stages | 3 (GSM8k), 6 (ProsQA) |
| `pad_latent_to_max` | Pad latent count to max_latent_stage | True |
| `reset_optimizer` | Reset optimizer when switching stages | True |

## How It Works

### Training Procedure

1. **Stage 0**: Train standard CoT model
2. **Stage 1**: Replace 1st reasoning step with `c_thought` continuous thoughts
3. **Stage 2**: Replace 1st and 2nd reasoning steps with `2 * c_thought` continuous thoughts
4. **Stage k**: Replace first k reasoning steps with `k * c_thought` continuous thoughts

### Continuous Thought Mechanism

During the forward pass:
1. Identify positions of `<|latent|>` tokens
2. For each latent position, replace its embedding with the hidden state from the previous position
3. This allows information to flow through "thinking" without explicit token generation

### Special Tokens

- `<|start-latent|>`: Marks the beginning of latent reasoning
- `<|latent|>`: Placeholder for continuous thought
- `<|end-latent|>`: Marks the end of latent reasoning

## Expected Results

Based on the paper (Table 1):

| Dataset | No-CoT | CoT | Coconut |
|---------|--------|-----|---------|
| GSM8k | 16.5% | 42.9% | 34.1% |
| ProntoQA | 93.8% | 98.8% | 99.8% |
| ProsQA | 76.7% | 77.5% | 97.0% |

Note: These results are from GPT-2. Performance may vary with Qwen3-0.6B.

## Differences from Original Implementation

The Qwen3 adaptation includes:

1. **Model-agnostic embedding access**: Uses `get_input_embeddings()` instead of model-specific paths
2. **Single-GPU support**: Added `--single-gpu` flag for easier development
3. **FSDP auto-wrap policy**: Configured for Qwen2DecoderLayer
4. **Improved error handling**: Better handling of missing checkpoints and configurations

## Troubleshooting

### Environment Issues

**Python not found:**
```bash
# Check Python version (3.8+ required)
python3 --version

# If using pyenv or conda, ensure correct Python is active
which python3
```

**Package conflicts:**
```bash
# Create a fresh environment
rm -rf coconut_env
./setup_env.sh
```

### CUDA Out of Memory

- Reduce `batch_size_training`
- Increase `gradient_accumulation_steps`
- Enable `bf16: True` for mixed precision

### Poor Performance

- Ensure Stage 0 CoT model is well-trained first
- Try different `c_thought` values
- Adjust learning rate

### Model Download Issues

```bash
# If Hugging Face model download fails, try:
export HF_HUB_ENABLE_HF_TRANSFER=0
python test_qwen_coconut.py
```

## Citation

```bibtex
@article{hao2024coconut,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```

## License

This implementation is adapted from the official Coconut repository by Meta Platforms, Inc.
