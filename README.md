# Coconut Implementation for Qwen3-0.6B

This is an implementation of Coconut (Chain of Continuous Thought) for the Qwen3-0.6B model, based on the paper "Training Large Language Models to Reason in a Continuous Latent Space" (arXiv:2412.06769).

## Overview

Coconut enables LLMs to reason in a continuous latent space instead of explicitly generating reasoning steps as text tokens. Key features:

1. **Continuous Thoughts**: Hidden states are fed back as input embeddings instead of decoding to tokens
2. **Multi-Stage Training**: Progressive curriculum that replaces language reasoning with continuous thoughts
3. **Adaptive Latent Count**: Confidence-based dynamic latent token count (experimental)

## Files

### Core Implementation

- `coconut_qwen.py`: Core CoconutQwen model wrapper
- `coconut_adaptive.py`: Adaptive Coconut with confidence-based dynamic latent count
- `dataset_qwen.py`: Dataset processing utilities
- `run_qwen.py`: Training script (single-GPU and distributed)
- `utils.py`: Utility functions

### Data

- `data/reasoning_dataset.py`: Built-in reasoning problems (arithmetic, logic, word problems)

### Configuration

- `args/qwen_cot.yaml`: Stage 0 - Chain-of-Thought baseline
- `args/qwen_coconut.yaml`: Stages 1+ - Coconut training
- `args/qwen_coconut_eval.yaml`: Evaluation configuration
- `args/qwen_prosqa_coconut.yaml`: ProsQA dataset configuration

### Testing & Demo

- `tests/test_coconut_basic.py`: Basic implementation tests
- `tests/test_adaptive_coconut.py`: Adaptive latent mechanism tests
- `demo_train.py`: Interactive training demo

## Quick Start

```bash
# Setup environment
./setup_env.sh
source coconut_env/bin/activate

# Run the demo (~15-20 min on CPU, ~3-5 min on GPU)
python demo_train.py
```

The demo shows adaptive latent reasoning:
1. **Stage 0**: Train with full Chain-of-Thought
2. **Stages 1-2**: Progressive latent replacement + confidence training
3. **Inference**: Dynamic latent count based on confidence

Example output:
```
STAGE 0: Chain-of-Thought Training
  Epoch 1: loss=0.7951, conf_loss=0.0000

STAGE 1: Latent Reasoning + Confidence Training
  Epoch 1: loss=0.6056, conf_loss=0.5630
  Epoch 3: loss=0.1310, conf_loss=0.0151

ADAPTIVE INFERENCE DEMO
[Complexity 1] What is 5 + 3?
  Latent 1: confidence = 0.016
  Latent 2: confidence = 0.019
  ...
  → Used 6 latent tokens
```

## Environment Setup

### Quick Setup

```bash
chmod +x setup_env.sh
./setup_env.sh
source coconut_env/bin/activate
```

### Manual Setup

```bash
python3 -m venv coconut_env
source coconut_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify Installation

```bash
# Run basic tests
python tests/test_coconut_basic.py

# Run adaptive tests (quick)
python tests/test_adaptive_coconut.py --quick

# Run full test suite (~10-15 min)
python tests/test_adaptive_coconut.py
```

## Training

### Data Format

JSON format with questions, reasoning steps, and answers:
```json
[
  {
    "question": "John has 5 apples...",
    "steps": ["Step 1: ...", "Step 2: ..."],
    "answer": "10"
  }
]
```

Or use the built-in dataset:
```python
from data.reasoning_dataset import get_train_test_split
train, test = get_train_test_split()
```

### Stage 0 - Train CoT Baseline

```bash
python run_qwen.py args/qwen_cot.yaml --single-gpu
```

### Stages 1+ - Train Coconut

Update `load_model_path` in `args/qwen_coconut.yaml`, then:

```bash
python run_qwen.py args/qwen_coconut.yaml --single-gpu
```

### Evaluate

```bash
python run_qwen.py args/qwen_coconut_eval.yaml --single-gpu
```

## How It Works

### Continuous Thought Mechanism

1. Identify `<|latent|>` token positions
2. Replace each latent embedding with the hidden state from the previous position
3. Information flows through "thinking" without generating text

### Special Tokens

- `<|start-latent|>`: Beginning of latent reasoning
- `<|latent|>`: Continuous thought placeholder
- `<|end-latent|>`: End of latent reasoning

### Training Stages

- **Stage 0**: Train standard CoT model
- **Stage k**: Replace first k reasoning steps with `k * c_thought` latent tokens

## Adaptive Latent Reasoning

The adaptive extension adds confidence-based dynamic latent count:

```python
from coconut_adaptive import AdaptiveCoconutQwen

model = AdaptiveCoconutQwen(
    base_model,
    latent_token_id=latent_id,
    start_latent_id=start_id,
    end_latent_id=end_id,
    eos_token_id=eos_id,
    max_latent_tokens=8,
    confidence_threshold=0.7,
)

# Adaptive generation
outputs, num_latent_used = model.generate_adaptive(
    input_ids, attention_mask,
    min_latent=1, max_latent=6,
    verbose=True,
)
```

The confidence head predicts "readiness to answer" after each latent token, allowing the model to use more thinking for hard problems and less for easy ones.

## Configuration Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `c_thought` | Latent tokens per reasoning step | 1-2 |
| `epochs_per_stage` | Epochs per training stage | 3-5 |
| `max_latent_stage` | Maximum latent stages | 3-6 |

## Troubleshooting

**Environment issues:**
```bash
rm -rf coconut_env
./setup_env.sh
```

**CUDA out of memory:**
- Reduce `batch_size_training`
- Enable `bf16: True`

**Model download issues:**
```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
python tests/test_coconut_basic.py
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

Adapted from the official Coconut repository by Meta Platforms, Inc.
