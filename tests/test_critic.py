#!/usr/bin/env python3
"""
Test script for Latent Critic implementation.

Tests both value-based and contrastive critics for verifying
intermediate reasoning steps in continuous space.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut_critic import (
    LatentValueCritic,
    ContrastiveLatentCritic,
    CoconutWithCritic,
    CriticOutputs,
)


def test_value_critic():
    """Test the LatentValueCritic module."""
    print("=" * 50)
    print("Test: LatentValueCritic")
    print("=" * 50)

    hidden_size = 512
    critic = LatentValueCritic(hidden_size)

    # Test single hidden state
    h = torch.randn(1, hidden_size)
    value = critic(h)
    assert value.shape == (1, 1), f"Expected (1, 1), got {value.shape}"
    assert 0 <= value.item() <= 1, "Value should be in [0, 1]"
    print(f"✓ Single hidden state -> value: {value.item():.3f}")

    # Test sequence of hidden states
    seq_len = 5
    h_seq = torch.randn(2, seq_len, hidden_size)
    values = critic(h_seq)
    assert values.shape == (2, seq_len, 1), f"Expected (2, {seq_len}, 1), got {values.shape}"
    assert (values >= 0).all() and (values <= 1).all(), "Values should be in [0, 1]"
    print(f"✓ Sequence of hidden states -> shape: {values.shape}")

    print("PASSED: LatentValueCritic test\n")
    return True


def test_contrastive_critic():
    """Test the ContrastiveLatentCritic module."""
    print("=" * 50)
    print("Test: ContrastiveLatentCritic")
    print("=" * 50)

    hidden_size = 512
    projection_dim = 128
    memory_size = 50
    max_steps = 5

    critic = ContrastiveLatentCritic(
        hidden_size=hidden_size,
        projection_dim=projection_dim,
        memory_size=memory_size,
        max_steps=max_steps,
    )

    # Test with empty memory (should return 0.5)
    h = torch.randn(3, hidden_size)
    step_indices = torch.tensor([0, 1, 2])
    scores = critic(h, step_indices)
    assert scores.shape == (3,), f"Expected (3,), got {scores.shape}"
    assert all(s == 0.5 for s in scores.tolist()), "Empty memory should return 0.5"
    print(f"✓ Empty memory -> neutral scores: {scores.tolist()}")

    # Add some reference states
    for step in range(3):
        for _ in range(5):
            ref_h = torch.randn(hidden_size)
            critic.update_memory(ref_h, step, is_correct=True)

    print(f"✓ Added reference states to memory")
    print(f"  Memory filled: {critic.memory_filled[:3].tolist()}")

    # Test with populated memory
    h = torch.randn(3, hidden_size)
    scores = critic(h, step_indices)
    assert scores.shape == (3,), f"Expected (3,), got {scores.shape}"
    print(f"✓ With memory -> scores: {[f'{s:.3f}' for s in scores.tolist()]}")

    # Test that similar states get higher scores
    ref_state = torch.randn(hidden_size)
    critic.update_memory(ref_state, 0, is_correct=True)

    # Query with same state (should get high score)
    similar_score = critic(ref_state.unsqueeze(0), torch.tensor([0]))[0]

    # Query with random state
    random_state = torch.randn(1, hidden_size)
    random_score = critic(random_state, torch.tensor([0]))[0]

    print(f"✓ Similar state score: {similar_score:.3f}")
    print(f"  Random state score: {random_score:.3f}")

    # Test that incorrect trajectories don't update memory
    filled_before = critic.memory_filled[4].item()
    critic.update_memory(torch.randn(hidden_size), 4, is_correct=False)
    filled_after = critic.memory_filled[4].item()
    assert filled_before == filled_after, "Incorrect trajectories should not update memory"
    print("✓ Incorrect trajectories correctly ignored")

    print("PASSED: ContrastiveLatentCritic test\n")
    return True


def test_critic_integration_mock():
    """Test CoconutWithCritic with mock components."""
    print("=" * 50)
    print("Test: CoconutWithCritic Integration (Mock)")
    print("=" * 50)

    # Create a minimal mock model for testing
    class MockConfig:
        hidden_size = 256

    class MockModel:
        config = MockConfig()

        def get_input_embeddings(self):
            return torch.nn.Embedding(1000, 256)

        def parameters(self, recurse=True):
            return iter([torch.nn.Parameter(torch.randn(10))])

    # Test CriticOutputs namedtuple
    outputs = CriticOutputs(
        loss=torch.tensor(1.0),
        logits=torch.randn(2, 10, 1000),
        step_values=[torch.tensor([0.5, 0.6, 0.7])],
        trajectory_value=torch.tensor(0.6),
        critic_loss=torch.tensor(0.1),
    )

    assert abs(outputs.loss.item() - 1.0) < 1e-5
    assert abs(outputs.trajectory_value.item() - 0.6) < 1e-5
    print("✓ CriticOutputs namedtuple works correctly")

    print("PASSED: Integration mock test\n")
    return True


def test_critic_with_real_model():
    """Test CoconutWithCritic with actual Qwen model."""
    print("=" * 50)
    print("Test: CoconutWithCritic with Real Model")
    print("=" * 50)

    device = "cpu"
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

    # Test value critic
    print("\nTesting with value critic...")
    model = CoconutWithCritic(
        base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        critic_type="value",
    )
    model = model.to(device)

    # Create test input with latent tokens
    question = "What is 2 + 3?"
    q_tokens = tokenizer.encode(f"Q: {question}\n", add_special_tokens=True)
    a_tokens = tokenizer.encode("Answer: 5", add_special_tokens=False)

    input_ids = q_tokens + [start_id] + [latent_id] * 3 + [end_id] + a_tokens
    labels = [-100] * (len(q_tokens) + 5) + a_tokens  # 5 = start + 3 latent + end

    input_ids = torch.tensor([input_ids], device=device)
    labels = torch.tensor([labels], device=device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )

    print(f"✓ Forward pass successful")
    print(f"  Loss: {outputs.loss.item():.4f}")
    print(f"  Trajectory value: {outputs.trajectory_value.item():.4f}")
    print(f"  Critic loss: {outputs.critic_loss.item():.4f}")

    if outputs.step_values:
        print(f"  Step values: {[f'{v.mean().item():.3f}' for v in outputs.step_values]}")

    # Test with correctness labels
    is_correct = torch.tensor([True], device=device)
    outputs_with_labels = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        position_ids=position_ids,
        is_correct=is_correct,
    )
    print(f"✓ Forward with correctness labels")
    print(f"  Critic loss (with labels): {outputs_with_labels.critic_loss.item():.4f}")

    # Test contrastive critic
    print("\nTesting with contrastive critic...")
    model_contrastive = CoconutWithCritic(
        base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        critic_type="contrastive",
    )
    model_contrastive = model_contrastive.to(device)

    with torch.no_grad():
        outputs = model_contrastive(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            is_correct=is_correct,
        )

    print(f"✓ Contrastive critic forward pass")
    print(f"  Memory filled: {model_contrastive.critic.memory_filled[:4].tolist()}")

    print("\nPASSED: Real model integration test\n")
    return True


def run_quick_tests():
    """Run only unit tests (no model loading)."""
    print("\n" + "=" * 60)
    print("Running Quick Critic Tests")
    print("=" * 60 + "\n")

    test_value_critic()
    test_contrastive_critic()
    test_critic_integration_mock()

    print("=" * 60)
    print("ALL QUICK TESTS PASSED!")
    print("=" * 60)


def run_all_tests():
    """Run all tests including model loading."""
    print("\n" + "=" * 60)
    print("Running Full Critic Test Suite")
    print("=" * 60 + "\n")

    try:
        test_value_critic()
        test_contrastive_critic()
        test_critic_integration_mock()
        test_critic_with_real_model()

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
    parser.add_argument("--quick", action="store_true", help="Run only quick unit tests")
    args = parser.parse_args()

    if args.quick:
        run_quick_tests()
    else:
        run_all_tests()
