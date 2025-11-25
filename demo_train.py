#!/usr/bin/env python3
"""
Adaptive Coconut Training Demo for Qwen3-0.6B.

This script demonstrates:
1. Curriculum training (CoT -> Latent reasoning)
2. Adaptive latent count based on confidence
3. Different latent usage for easy vs hard problems

The key insight: the model learns WHEN to stop thinking by training
a confidence head alongside the main model.

Run with: python demo_train.py
Expected runtime: ~15-20 minutes on CPU, ~3-5 minutes on GPU
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

from coconut_adaptive import AdaptiveCoconutQwen
from dataset_qwen import MyCollator
from data.reasoning_dataset import get_all_problems, get_train_test_split


def setup_model(device="cpu"):
    """Load Qwen3-0.6B with adaptive Coconut wrapper."""
    print("Loading Qwen3-0.6B with Adaptive Coconut...")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float32,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    base_model.resize_token_embeddings(len(tokenizer))

    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")

    # Create ADAPTIVE model (not regular CoconutQwen)
    model = AdaptiveCoconutQwen(
        base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        max_latent_tokens=6,
        confidence_threshold=0.6,
    )

    model = model.to(device)
    print(f"Model loaded on {device}")
    print(f"Special tokens: start={start_id}, end={end_id}, latent={latent_id}")

    return model, tokenizer, (start_id, end_id, latent_id)


def create_stage_batch(problems, tokenizer, start_id, end_id, latent_id, stage, c_thought=2):
    """Create training batch for curriculum stage with complexity-aware latent count."""
    batch_data = []

    for problem in problems:
        question = problem["question"]
        steps = problem["steps"]
        answer = problem["answer"]
        complexity = problem.get("complexity", 2)

        q_tokens = tokenizer.encode(
            f"Q: {question}\nThink step by step.\n",
            add_special_tokens=True
        )
        a_tokens = tokenizer.encode(f"Answer: {answer}", add_special_tokens=False)

        if stage == 0:
            # Full CoT
            step_text = " ".join(steps) + " "
            step_tokens = tokenizer.encode(step_text, add_special_tokens=False)
            input_ids = q_tokens + step_tokens + a_tokens
            labels = [-100] * len(q_tokens) + step_tokens + a_tokens
        else:
            # Latent tokens based on complexity (more complex = more latent)
            num_latent = min(stage * c_thought, complexity * c_thought)
            remaining_steps = steps[min(stage, len(steps)):]

            latent_section = [start_id] + [latent_id] * num_latent + [end_id]

            if remaining_steps:
                remaining_text = " ".join(remaining_steps) + " "
                remaining_tokens = tokenizer.encode(remaining_text, add_special_tokens=False)
            else:
                remaining_tokens = []

            input_ids = q_tokens + latent_section + remaining_tokens + a_tokens
            labels = (
                [-100] * (len(q_tokens) + len(latent_section)) +
                remaining_tokens + a_tokens
            )

        batch_data.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "position_ids": list(range(len(input_ids))),
        })

    return batch_data


def train_epoch(model, problems, tokenizer, start_id, end_id, latent_id,
                optimizer, collator, device, stage, c_thought=2):
    """Train one epoch with combined loss (main + confidence)."""
    model.train()
    total_loss = 0
    total_conf_loss = 0

    random.shuffle(problems)
    batch_size = 2

    for i in range(0, len(problems), batch_size):
        batch_problems = problems[i:i+batch_size]
        batch_data = create_stage_batch(
            batch_problems, tokenizer, start_id, end_id, latent_id,
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

        # Combined loss: main + confidence
        loss = outputs.loss + 0.1 * outputs.confidence_loss
        loss.backward()
        optimizer.step()

        total_loss += outputs.loss.item()
        total_conf_loss += outputs.confidence_loss.item()

    n_batches = max(1, len(problems) / batch_size)
    return total_loss / n_batches, total_conf_loss / n_batches


def adaptive_inference(model, tokenizer, start_id, end_id, latent_id, device, test_problems):
    """Run adaptive inference showing dynamic latent count."""
    model.eval()

    print("\n" + "=" * 60)
    print("ADAPTIVE INFERENCE DEMO")
    print("=" * 60)
    print("The model dynamically chooses how many latent tokens to use")
    print("based on its confidence. More thinking for hard problems!")
    print("-" * 60)

    results_by_complexity = {}

    for problem in test_problems:
        question = problem["question"]
        expected = problem["answer"]
        complexity = problem.get("complexity", 2)

        # Build input WITHOUT latent tokens (adaptive will add them)
        q_tokens = tokenizer.encode(
            f"Q: {question}\nThink step by step.\n",
            add_special_tokens=True
        )
        input_ids = torch.tensor([q_tokens], device=device)
        attention_mask = torch.ones_like(input_ids)

        # Adaptive generation with verbose output
        print(f"\n[Complexity {complexity}] {question}")
        with torch.no_grad():
            outputs, num_latent = model.generate_adaptive(
                input_ids,
                attention_mask,
                max_new_tokens=25,
                min_latent=1,
                max_latent=6,
                verbose=True,
            )

        # Decode only generated part
        generated_tokens = outputs[0][len(q_tokens) + num_latent + 2:].tolist()
        generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        is_correct = expected.lower() in generated.lower()
        status = "✓" if is_correct else "✗"

        print(f"  {status} Expected: {expected}, Got: {generated.strip()[:30]}")
        print(f"  → Used {num_latent} latent tokens")

        # Track by complexity
        if complexity not in results_by_complexity:
            results_by_complexity[complexity] = []
        results_by_complexity[complexity].append({
            "latents": num_latent,
            "correct": is_correct
        })

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS BY COMPLEXITY")
    print("=" * 60)

    for c in sorted(results_by_complexity.keys()):
        items = results_by_complexity[c]
        avg_latent = sum(r["latents"] for r in items) / len(items)
        accuracy = sum(1 for r in items if r["correct"]) / len(items)
        print(f"  Complexity {c}: avg {avg_latent:.1f} latents, {accuracy*100:.0f}% accuracy")

    return results_by_complexity


def main():
    """Run the adaptive Coconut demo."""
    print("=" * 60)
    print("Adaptive Coconut Training Demo")
    print("=" * 60)
    print()
    print("This demo shows ADAPTIVE latent reasoning:")
    print("1. Train with curriculum (CoT -> Latent)")
    print("2. Confidence head learns when to stop thinking")
    print("3. Simple problems = fewer latents, hard = more")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs_per_stage = 3
    c_thought = 2
    max_stage = 2

    print(f"Device: {device}")
    print(f"Epochs per stage: {epochs_per_stage}")
    print()

    # Setup
    model, tokenizer, (start_id, end_id, latent_id) = setup_model(device)
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Get diverse problems
    train_problems, test_problems = get_train_test_split(test_ratio=0.3)
    print(f"Training: {len(train_problems)} problems")
    print(f"Testing: {len(test_problems)} problems")

    # Show problem diversity
    print("\n" + "-" * 60)
    print("SAMPLE PROBLEMS BY COMPLEXITY:")
    print("-" * 60)
    for c in [1, 2, 3]:
        samples = [p for p in train_problems if p.get("complexity", 2) == c][:1]
        for s in samples:
            print(f"  [{c}] {s['question'][:50]}...")
    print("-" * 60)

    # Curriculum training
    for stage in range(max_stage + 1):
        print(f"\n{'=' * 40}")
        if stage == 0:
            print("STAGE 0: Chain-of-Thought Training")
        else:
            print(f"STAGE {stage}: Latent Reasoning + Confidence Training")
        print(f"{'=' * 40}")

        for epoch in range(epochs_per_stage):
            loss, conf_loss = train_epoch(
                model, train_problems, tokenizer, start_id, end_id, latent_id,
                optimizer, collator, device, stage=stage, c_thought=c_thought
            )
            print(f"  Epoch {epoch + 1}: loss={loss:.4f}, conf_loss={conf_loss:.4f}")

    # Adaptive inference demo
    results = adaptive_inference(
        model, tokenizer, start_id, end_id, latent_id, device, test_problems
    )

    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("The confidence head learns to predict 'readiness to answer'.")
    print("After training, the model should use:")
    print("  - FEWER latent tokens for simple problems (early exit)")
    print("  - MORE latent tokens for complex problems (keep thinking)")
    print()
    print("With more training data and epochs, this differentiation")
    print("becomes more pronounced.")
    print("=" * 60)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()
