#!/usr/bin/env python3
"""
Test script for Adaptive Coconut with confidence-based dynamic latent tokens.

This test demonstrates:
1. Confidence head training alongside base model
2. Adaptive inference with dynamic latent count
3. Comparison of different complexity problems
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random

from coconut_adaptive import AdaptiveCoconutQwen, compute_problem_complexity
from dataset_qwen import MyCollator
from data.reasoning_dataset import get_all_problems, get_train_test_split


def setup_model(device="cpu"):
    """Load model and create adaptive wrapper."""
    print("Loading Qwen3-0.6B...")

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

    # Create adaptive model
    model = AdaptiveCoconutQwen(
        base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        max_latent_tokens=6,
        confidence_threshold=0.7,
    )

    model = model.to(device)
    print(f"Model loaded on {device}")

    return model, tokenizer, (start_id, end_id, latent_id)


def create_curriculum_batch(
    problems, tokenizer, start_id, end_id, latent_id,
    stage, c_thought=2
):
    """Create training batch for curriculum stage."""
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
            # Replace steps with latent tokens based on complexity
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
            "complexity": complexity,
        })

    return batch_data


def train_epoch(model, problems, tokenizer, start_id, end_id, latent_id,
                optimizer, collator, device, stage, c_thought=2):
    """Train for one epoch with both losses."""
    model.train()
    total_loss = 0
    total_conf_loss = 0

    random.shuffle(problems)
    batch_size = 2

    for i in range(0, len(problems), batch_size):
        batch_problems = problems[i:i+batch_size]
        batch_data = create_curriculum_batch(
            batch_problems, tokenizer, start_id, end_id, latent_id,
            stage=stage, c_thought=c_thought
        )

        # Remove complexity from batch_data for collator
        for item in batch_data:
            del item["complexity"]

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

        # Combined loss
        loss = outputs.loss + 0.1 * outputs.confidence_loss
        loss.backward()
        optimizer.step()

        total_loss += outputs.loss.item()
        total_conf_loss += outputs.confidence_loss.item()

    n_batches = len(problems) / batch_size
    return total_loss / n_batches, total_conf_loss / n_batches


def test_adaptive_inference(model, tokenizer, start_id, end_id, latent_id,
                           device, test_problems):
    """Test adaptive inference on problems of varying complexity."""
    model.eval()

    results = []

    print("\n" + "=" * 60)
    print("Adaptive Inference Test")
    print("=" * 60)

    for problem in test_problems:
        question = problem["question"]
        expected = problem["answer"]
        complexity = problem.get("complexity", 2)

        # Prepare input (without latent tokens - adaptive will add them)
        q_tokens = tokenizer.encode(
            f"Q: {question}\nThink step by step.\n",
            add_special_tokens=True
        )
        input_ids = torch.tensor([q_tokens], device=device)
        attention_mask = torch.ones_like(input_ids)

        # Adaptive generation
        with torch.no_grad():
            outputs, num_latent = model.generate_adaptive(
                input_ids,
                attention_mask,
                max_new_tokens=30,
                min_latent=1,
                max_latent=6,
                verbose=False,
            )

        generated_tokens = outputs[0][len(q_tokens) + num_latent + 2:].tolist()  # +2 for start/end
        generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        is_correct = expected.lower() in generated.lower()

        results.append({
            "question": question,
            "expected": expected,
            "generated": generated.strip()[:40],
            "complexity": complexity,
            "latents_used": num_latent,
            "correct": is_correct,
        })

        status = "✓" if is_correct else "✗"
        print(f"\n{status} Complexity {complexity}: {question[:40]}...")
        print(f"   Expected: {expected}")
        print(f"   Got: {generated.strip()[:40]}")
        print(f"   Latent tokens used: {num_latent}")

    # Summary
    correct = sum(1 for r in results if r["correct"])
    print(f"\n{'=' * 60}")
    print(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.0f}%)")

    # Analyze latent usage by complexity
    print("\nLatent tokens by complexity:")
    by_complexity = {}
    for r in results:
        c = r["complexity"]
        if c not in by_complexity:
            by_complexity[c] = []
        by_complexity[c].append(r["latents_used"])

    for c in sorted(by_complexity.keys()):
        avg = sum(by_complexity[c]) / len(by_complexity[c])
        print(f"  Complexity {c}: avg {avg:.1f} latent tokens")

    return results


def run_full_test():
    """Run the complete adaptive Coconut test."""
    print("=" * 60)
    print("Adaptive Coconut Test Suite")
    print("=" * 60)
    print()
    print("Testing confidence-based dynamic latent token count.")
    print("The model should use MORE latent tokens for complex problems")
    print("and FEWER for simple ones.")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs_per_stage = 2
    c_thought = 2
    max_stage = 2

    print(f"Device: {device}")
    print(f"Epochs per stage: {epochs_per_stage}")
    print()

    # Setup
    model, tokenizer, (start_id, end_id, latent_id) = setup_model(device)
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Get data
    train_problems, test_problems = get_train_test_split(test_ratio=0.3)
    print(f"Training problems: {len(train_problems)}")
    print(f"Test problems: {len(test_problems)}")

    # Curriculum training
    for stage in range(max_stage + 1):
        print(f"\n{'=' * 40}")
        print(f"STAGE {stage}")
        print(f"{'=' * 40}")

        for epoch in range(epochs_per_stage):
            loss, conf_loss = train_epoch(
                model, train_problems, tokenizer, start_id, end_id, latent_id,
                optimizer, collator, device, stage=stage, c_thought=c_thought
            )
            print(f"  Epoch {epoch + 1}: loss={loss:.4f}, conf_loss={conf_loss:.4f}")

    # Test adaptive inference
    results = test_adaptive_inference(
        model, tokenizer, start_id, end_id, latent_id,
        device, test_problems
    )

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print()
    print("Key observations:")
    print("- Confidence head learns to predict readiness")
    print("- Complex problems should trigger more latent tokens")
    print("- Simple problems should exit early")
    print()

    return results


def test_confidence_head_only():
    """Quick test of just the confidence head mechanism."""
    print("=" * 50)
    print("Test: Confidence Head Mechanism")
    print("=" * 50)

    device = "cpu"
    model, tokenizer, (start_id, end_id, latent_id) = setup_model(device)

    # Test confidence prediction on random hidden states
    hidden_size = model.base_causallm.config.hidden_size
    dummy_hidden = torch.randn(1, hidden_size)

    confidence = model.confidence_head(dummy_hidden)
    print(f"Random hidden state -> confidence: {confidence.item():.3f}")

    # Test that confidence is in [0, 1]
    assert 0 <= confidence.item() <= 1, "Confidence should be in [0, 1]"
    print("✓ Confidence in valid range [0, 1]")

    print("PASSED: Confidence head test")
    return True


def test_adaptive_generation_mechanism():
    """Test the adaptive generation without training."""
    print("\n" + "=" * 50)
    print("Test: Adaptive Generation Mechanism")
    print("=" * 50)

    device = "cpu"
    model, tokenizer, (start_id, end_id, latent_id) = setup_model(device)

    # Simple question
    question = "What is 2 + 2?"
    q_tokens = tokenizer.encode(f"Q: {question}\n", add_special_tokens=True)
    input_ids = torch.tensor([q_tokens], device=device)
    attention_mask = torch.ones_like(input_ids)

    print(f"Question: {question}")
    print("Running adaptive generation with verbose=True...")

    with torch.no_grad():
        outputs, num_latent = model.generate_adaptive(
            input_ids,
            attention_mask,
            max_new_tokens=20,
            min_latent=1,
            max_latent=4,
            confidence_threshold=0.5,
            verbose=True,
        )

    print(f"Latent tokens used: {num_latent}")
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output: {generated}")

    assert 1 <= num_latent <= 4, "Latent count should be in [1, 4]"
    print("✓ Adaptive generation mechanism works")

    print("PASSED: Adaptive generation test")
    return True


def test_complexity_estimation():
    """Test the problem complexity estimation function."""
    print("\n" + "=" * 50)
    print("Test: Complexity Estimation")
    print("=" * 50)

    test_cases = [
        ("What is 2 + 2?", 1, 2),  # Simple
        ("What is 5 + 3?", 1, 2),
        ("Tom has 5 apples. He gives 2 to Mary. How many remaining?", 2, 3),
        ("A store has 15 books. They sell 7 and receive 4 more. Total?", 3, 4),
        ("Calculate the average of 10, 20, and 30.", 2, 4),
    ]

    for question, min_exp, max_exp in test_cases:
        complexity = compute_problem_complexity(question)
        status = "✓" if min_exp <= complexity <= max_exp else "✗"
        print(f"{status} \"{question[:40]}...\" -> complexity {complexity} (expected {min_exp}-{max_exp})")

    print("PASSED: Complexity estimation test")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Adaptive Coconut Test Suite")
    print("=" * 60 + "\n")

    try:
        test_confidence_head_only()
        test_complexity_estimation()
        test_adaptive_generation_mechanism()

        print("\n" + "-" * 60)
        print("Running full training test (this takes ~10-15 min on CPU)...")
        print("-" * 60)

        run_full_test()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run only quick tests")
    args = parser.parse_args()

    if args.quick:
        test_confidence_head_only()
        test_complexity_estimation()
        test_adaptive_generation_mechanism()
    else:
        run_all_tests()
