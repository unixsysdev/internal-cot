#!/usr/bin/env python3
"""
Quick training and inference demo for Coconut Qwen3-0.6B.

This script demonstrates:
1. Loading the model and adding special tokens
2. Training on a small arithmetic dataset for a few steps
3. Running inference to show the model's predictions

Run with: python demo_train.py
Expected runtime: ~2-5 minutes on CPU, ~30 seconds on GPU
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random

from coconut_qwen import CoconutQwen
from dataset_qwen import create_sample_dataset, MyCollator


def setup_model_and_tokenizer(device="cpu"):
    """Load Qwen3-0.6B and set up for Coconut training."""
    print("Loading Qwen3-0.6B model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float32,  # Use float32 for CPU compatibility
    )

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens for Coconut
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    model.resize_token_embeddings(len(tokenizer))

    # Get special token IDs
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")

    # Create Coconut wrapper
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


def create_training_batch(tokenizer, start_id, end_id, latent_id, num_latent=2, batch_size=4):
    """Create a training batch with latent tokens for arithmetic problems."""
    samples = create_sample_dataset(num_samples=batch_size)

    batch_data = []
    for sample in samples:
        question = sample["question"]
        answer = sample["answer"]

        # Format: Question <|start-latent|> <|latent|>... <|end-latent|> Answer
        question_tokens = tokenizer.encode(question, add_special_tokens=False)
        answer_tokens = tokenizer.encode(" " + answer, add_special_tokens=False)

        # Build input sequence
        input_ids = (
            [tokenizer.bos_token_id] if tokenizer.bos_token_id else []
        ) + question_tokens + [start_id] + [latent_id] * num_latent + [end_id] + answer_tokens

        # Labels: -100 for everything except the answer
        labels = [-100] * (len(input_ids) - len(answer_tokens)) + answer_tokens

        batch_data.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "position_ids": list(range(len(input_ids))),
        })

    return batch_data


def train_step(model, batch, optimizer, device):
    """Perform a single training step."""
    model.train()

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

    return loss.item()


def inference_demo(model, tokenizer, start_id, end_id, latent_id, device, num_latent=2):
    """Run inference on a few test problems."""
    model.eval()

    test_problems = [
        "What is 3 + 4?",
        "What is 7 - 2?",
        "What is 5 + 5?",
    ]

    print("\n" + "=" * 50)
    print("Inference Demo")
    print("=" * 50)

    for problem in test_problems:
        # Create input with latent tokens
        question_tokens = tokenizer.encode(problem, add_special_tokens=True)
        input_tokens = question_tokens + [start_id] + [latent_id] * num_latent + [end_id]

        input_ids = torch.tensor([input_tokens], device=device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask,
                max_new_tokens=10,
            )

        # Decode only the generated tokens (after end_latent)
        generated_tokens = outputs[0][len(input_tokens):].tolist()
        generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"Q: {problem}")
        print(f"A: {generated}")
        print()


def main():
    """Run the training and inference demo."""
    print("=" * 60)
    print("Coconut Qwen3-0.6B Training & Inference Demo")
    print("=" * 60)
    print()

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_steps = 10
    batch_size = 2
    num_latent = 2
    learning_rate = 1e-5

    print(f"Device: {device}")
    print(f"Training steps: {num_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Latent tokens: {num_latent}")
    print()

    # Setup
    model, tokenizer, (start_id, end_id, latent_id) = setup_model_and_tokenizer(device)

    # Create collator
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Initial inference (before training)
    print("\n>>> Before Training:")
    inference_demo(model, tokenizer, start_id, end_id, latent_id, device, num_latent)

    # Training loop
    print("=" * 50)
    print("Training...")
    print("=" * 50)

    losses = []
    for step in tqdm(range(num_steps), desc="Training"):
        # Create a fresh batch each step
        batch_data = create_training_batch(
            tokenizer, start_id, end_id, latent_id,
            num_latent=num_latent, batch_size=batch_size
        )
        batch = collator(batch_data)

        loss = train_step(model, batch, optimizer, device)
        losses.append(loss)

        if (step + 1) % 5 == 0:
            print(f"Step {step + 1}/{num_steps}, Loss: {loss:.4f}")

    # Training summary
    print(f"\nTraining complete!")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss change: {losses[-1] - losses[0]:.4f}")

    # Inference after training
    print("\n>>> After Training:")
    inference_demo(model, tokenizer, start_id, end_id, latent_id, device, num_latent)

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print()
    print("Note: With only 10 training steps, significant improvement")
    print("is not expected. For real training, use run_qwen.py with")
    print("the full curriculum training approach.")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    main()
