#!/usr/bin/env python3
"""
Coconut Curriculum Training Demo for Qwen3-0.6B.

This script demonstrates the ACTUAL Coconut training approach:
1. Stage 0: Train on Chain-of-Thought (explicit reasoning steps)
2. Stage 1: Replace first reasoning step with latent tokens
3. Stage 2: Replace more reasoning steps with latent tokens

The key insight: the model learns to compress CoT reasoning into
continuous hidden states because it was FIRST trained with explicit
reasoning, then progressively had the text replaced with latent tokens.

Run with: python demo_train.py
Expected runtime: ~10-15 minutes on CPU, ~2-3 minutes on GPU
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random

from coconut_qwen import CoconutQwen
from dataset_qwen import MyCollator


# Simple arithmetic problems with step-by-step reasoning
TRAINING_DATA = [
    {
        "question": "What is 3 + 5?",
        "steps": ["First, I need to add 3 and 5.", "3 plus 5 equals 8."],
        "answer": "8"
    },
    {
        "question": "What is 7 - 2?",
        "steps": ["First, I need to subtract 2 from 7.", "7 minus 2 equals 5."],
        "answer": "5"
    },
    {
        "question": "What is 4 + 6?",
        "steps": ["First, I need to add 4 and 6.", "4 plus 6 equals 10."],
        "answer": "10"
    },
    {
        "question": "What is 9 - 3?",
        "steps": ["First, I need to subtract 3 from 9.", "9 minus 3 equals 6."],
        "answer": "6"
    },
    {
        "question": "What is 2 + 8?",
        "steps": ["First, I need to add 2 and 8.", "2 plus 8 equals 10."],
        "answer": "10"
    },
    {
        "question": "What is 6 - 4?",
        "steps": ["First, I need to subtract 4 from 6.", "6 minus 4 equals 2."],
        "answer": "2"
    },
    {
        "question": "What is 5 + 5?",
        "steps": ["First, I need to add 5 and 5.", "5 plus 5 equals 10."],
        "answer": "10"
    },
    {
        "question": "What is 8 - 1?",
        "steps": ["First, I need to subtract 1 from 8.", "8 minus 1 equals 7."],
        "answer": "7"
    },
]

TEST_PROBLEMS = [
    {"question": "What is 3 + 4?", "answer": "7"},
    {"question": "What is 9 - 5?", "answer": "4"},
    {"question": "What is 6 + 2?", "answer": "8"},
]


def setup_model_and_tokenizer(device="cpu"):
    """Load Qwen3-0.6B and set up for Coconut training."""
    print("Loading Qwen3-0.6B model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float32,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens for Coconut
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    model.resize_token_embeddings(len(tokenizer))

    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")

    coconut_model = CoconutQwen(
        model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    coconut_model = coconut_model.to(device)

    print(f"Model loaded on {device}")
    print(f"Special tokens: start={start_id}, end={end_id}, latent={latent_id}")

    return coconut_model, tokenizer, (start_id, end_id, latent_id)


def create_stage_batch(
    samples, tokenizer, start_id, end_id, latent_id,
    stage, c_thought=2
):
    """
    Create training batch for a specific curriculum stage.

    Stage 0: Full CoT - "Q: ... Step1 Step2 Answer: X"
    Stage 1: Replace step 1 - "Q: ... <latent><latent> Step2 Answer: X"
    Stage 2: Replace steps 1&2 - "Q: ... <latent><latent><latent><latent> Answer: X"
    """
    batch_data = []

    for sample in samples:
        question = sample["question"]
        steps = sample["steps"]
        answer = sample["answer"]

        # Encode question
        q_tokens = tokenizer.encode(
            f"Q: {question}\nLet me think step by step.\n",
            add_special_tokens=True
        )

        # Encode answer
        a_tokens = tokenizer.encode(f"Answer: {answer}", add_special_tokens=False)

        if stage == 0:
            # Full Chain-of-Thought: all steps as text
            step_text = " ".join(steps) + " "
            step_tokens = tokenizer.encode(step_text, add_special_tokens=False)

            input_ids = q_tokens + step_tokens + a_tokens
            # Train on steps AND answer
            labels = [-100] * len(q_tokens) + step_tokens + a_tokens

        else:
            # Replace first `stage` steps with latent tokens
            num_latent = min(stage, len(steps)) * c_thought
            remaining_steps = steps[stage:] if stage < len(steps) else []

            # Build sequence: Q + <start> + latents + <end> + remaining_steps + answer
            latent_section = [start_id] + [latent_id] * num_latent + [end_id]

            if remaining_steps:
                remaining_text = " ".join(remaining_steps) + " "
                remaining_tokens = tokenizer.encode(remaining_text, add_special_tokens=False)
            else:
                remaining_tokens = []

            input_ids = q_tokens + latent_section + remaining_tokens + a_tokens

            # Labels: -100 for question and latent section, train on remaining + answer
            labels = (
                [-100] * (len(q_tokens) + len(latent_section)) +
                (remaining_tokens if remaining_tokens else []) +
                a_tokens
            )

        batch_data.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "position_ids": list(range(len(input_ids))),
        })

    return batch_data


def train_epoch(model, data, tokenizer, start_id, end_id, latent_id,
                optimizer, collator, device, stage, c_thought=2):
    """Train for one epoch at a given stage."""
    model.train()
    total_loss = 0

    # Shuffle data
    shuffled = data.copy()
    random.shuffle(shuffled)

    # Create batches of size 2
    batch_size = 2
    for i in range(0, len(shuffled), batch_size):
        batch_samples = shuffled[i:i+batch_size]
        batch_data = create_stage_batch(
            batch_samples, tokenizer, start_id, end_id, latent_id,
            stage=stage, c_thought=c_thought
        )
        batch = collator(batch_data)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        position_ids = batch["position_ids"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / (len(data) / batch_size)


def inference_demo(model, tokenizer, start_id, end_id, latent_id, device,
                   stage, c_thought=2):
    """Run inference and show results."""
    model.eval()

    correct = 0
    results = []

    for problem in TEST_PROBLEMS:
        question = problem["question"]
        expected = problem["answer"]

        # Build input based on stage
        q_tokens = tokenizer.encode(
            f"Q: {question}\nLet me think step by step.\n",
            add_special_tokens=True
        )

        if stage == 0:
            # No latent tokens for stage 0
            input_tokens = q_tokens
        else:
            # Add latent tokens
            num_latent = stage * c_thought
            input_tokens = q_tokens + [start_id] + [latent_id] * num_latent + [end_id]

        input_ids = torch.tensor([input_tokens], device=device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask,
                max_new_tokens=30,
            )

        generated_tokens = outputs[0][len(input_tokens):].tolist()
        generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Check if answer is correct (simple check)
        is_correct = expected in generated
        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "expected": expected,
            "generated": generated.strip()[:50],  # Truncate for display
            "correct": is_correct
        })

    return results, correct / len(TEST_PROBLEMS)


def print_stage_header(stage, total_stages):
    """Print a nice header for each stage."""
    print("\n" + "=" * 60)
    if stage == 0:
        print(f"STAGE 0: Chain-of-Thought Training")
        print("Training with FULL explicit reasoning steps")
        print("Format: Q: ... Step1 Step2 Answer: X")
    else:
        print(f"STAGE {stage}: Latent Reasoning Training")
        print(f"Replacing first {stage} step(s) with latent tokens")
        if stage == 1:
            print("Format: Q: ... <latent><latent> Step2 Answer: X")
        else:
            print("Format: Q: ... <latent>... Answer: X")
    print("=" * 60)


def main():
    """Run the curriculum training demo."""
    print("=" * 60)
    print("Coconut Curriculum Training Demo")
    print("=" * 60)
    print()
    print("This demo shows how Coconut actually works:")
    print("1. First train with explicit Chain-of-Thought")
    print("2. Progressively replace reasoning steps with latent tokens")
    print("3. Model learns to compress reasoning into hidden states")
    print()

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs_per_stage = 3
    c_thought = 2  # Latent tokens per reasoning step
    learning_rate = 5e-5
    max_stage = 2  # We have 2 reasoning steps

    print(f"Device: {device}")
    print(f"Epochs per stage: {epochs_per_stage}")
    print(f"Latent tokens per step: {c_thought}")
    print(f"Total stages: {max_stage + 1} (0 to {max_stage})")
    print()

    # Setup
    model, tokenizer, (start_id, end_id, latent_id) = setup_model_and_tokenizer(device)
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Show example of each stage format
    print("\n" + "-" * 60)
    print("TRAINING FORMAT EXAMPLES:")
    print("-" * 60)
    sample = TRAINING_DATA[0]
    print(f"\nOriginal problem:")
    print(f"  Q: {sample['question']}")
    print(f"  Steps: {sample['steps']}")
    print(f"  Answer: {sample['answer']}")
    print(f"\nStage 0 (Full CoT):")
    print(f"  Q: {sample['question']}")
    print(f"  {sample['steps'][0]} {sample['steps'][1]} Answer: {sample['answer']}")
    print(f"\nStage 1 (Replace step 1):")
    print(f"  Q: {sample['question']}")
    print(f"  <|start-latent|><|latent|><|latent|><|end-latent|> {sample['steps'][1]} Answer: {sample['answer']}")
    print(f"\nStage 2 (Replace steps 1&2):")
    print(f"  Q: {sample['question']}")
    print(f"  <|start-latent|><|latent|><|latent|><|latent|><|latent|><|end-latent|> Answer: {sample['answer']}")
    print("-" * 60)

    # Track metrics across stages
    all_losses = []
    all_accuracies = []

    # Curriculum training loop
    for stage in range(max_stage + 1):
        print_stage_header(stage, max_stage)

        stage_losses = []

        # Train for multiple epochs at this stage
        for epoch in range(epochs_per_stage):
            loss = train_epoch(
                model, TRAINING_DATA, tokenizer, start_id, end_id, latent_id,
                optimizer, collator, device, stage=stage, c_thought=c_thought
            )
            stage_losses.append(loss)
            print(f"  Epoch {epoch + 1}/{epochs_per_stage}, Loss: {loss:.4f}")

        all_losses.append(stage_losses)

        # Evaluate at end of stage
        print(f"\n  Inference test (stage {stage}):")
        results, accuracy = inference_demo(
            model, tokenizer, start_id, end_id, latent_id, device,
            stage=stage, c_thought=c_thought
        )
        all_accuracies.append(accuracy)

        for r in results:
            status = "✓" if r["correct"] else "✗"
            print(f"    {status} Q: {r['question']}")
            print(f"       Expected: {r['expected']}, Got: {r['generated']}")

        print(f"\n  Stage {stage} accuracy: {accuracy*100:.0f}%")

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)

    print("\nLoss progression by stage:")
    for stage, losses in enumerate(all_losses):
        print(f"  Stage {stage}: {losses[0]:.4f} -> {losses[-1]:.4f}")

    print("\nAccuracy by stage:")
    for stage, acc in enumerate(all_accuracies):
        mode = "CoT" if stage == 0 else f"{stage*c_thought} latent tokens"
        print(f"  Stage {stage} ({mode}): {acc*100:.0f}%")

    print("\n" + "-" * 60)
    print("KEY INSIGHT:")
    print("-" * 60)
    print("The model first learns to reason with explicit text (Stage 0),")
    print("then learns to compress that reasoning into latent tokens")
    print("(Stages 1-2). The hidden states carry the reasoning forward")
    print("without generating intermediate text.")
    print()
    print("For production training, use run_qwen.py with larger datasets")
    print("and more epochs per stage.")
    print("=" * 60)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()
