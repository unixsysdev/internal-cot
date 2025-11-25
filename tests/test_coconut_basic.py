#!/usr/bin/env python3
"""
Test script for Coconut with Qwen3-0.6B.
This script verifies that the implementation works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut_qwen import CoconutQwen
from dataset_qwen import (
    get_dataset,
    get_question_latent_dataset,
    get_cot_latent_dataset,
    MyCollator,
    create_sample_dataset,
)
from utils import Config


def test_model_loading():
    """Test that Qwen3-0.6B can be loaded correctly."""
    print("=" * 50)
    print("Test 1: Model Loading")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

    print(f"Model type: {type(model)}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model config: {model.config.hidden_size} hidden, {model.config.num_hidden_layers} layers")

    # Add special tokens
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    print(f"Special tokens added - latent: {latent_id}, start: {start_id}, end: {end_id}")

    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Test forward pass
    input_text = "Hello, this is a test."
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Hidden states count: {len(outputs.hidden_states)}")

    print("PASSED: Model loading test")
    return tokenizer, model, latent_id, start_id, end_id


def test_coconut_wrapper(tokenizer, base_model, latent_id, start_id, end_id):
    """Test the CoconutQwen wrapper."""
    print("\n" + "=" * 50)
    print("Test 2: Coconut Wrapper")
    print("=" * 50)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = CoconutQwen(
        base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    print(f"CoconutQwen created successfully")

    # Test forward pass without latent tokens
    input_text = "What is 2 + 2?\n"
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = input_ids.clone()
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, labels, position_ids)

    print(f"Forward pass without latent tokens:")
    print(f"  Loss: {outputs.loss.item():.4f}")
    print(f"  Logits shape: {outputs.logits.shape}")
    print(f"  Inputs embeds shape: {outputs.inputs_embeds.shape}")

    # Test forward pass with latent tokens
    tokens = input_ids.tolist()[0] + [start_id, latent_id, latent_id, end_id]
    input_ids_latent = torch.tensor([tokens])
    attention_mask_latent = torch.ones_like(input_ids_latent)
    labels_latent = torch.tensor([[-100] * (len(tokens) - 1) + [tokenizer.eos_token_id]])
    position_ids_latent = torch.arange(len(tokens)).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids_latent, attention_mask_latent, labels_latent, position_ids_latent)

    print(f"\nForward pass with 2 latent tokens:")
    print(f"  Loss: {outputs.loss.item():.4f}")
    print(f"  Logits shape: {outputs.logits.shape}")

    print("PASSED: Coconut wrapper test")
    return model


def test_generation(model, tokenizer, latent_id, start_id, end_id):
    """Test generation with Coconut."""
    print("\n" + "=" * 50)
    print("Test 3: Generation")
    print("=" * 50)

    # Create input with latent tokens
    question = "What is 5 + 3?\n"
    question_tokens = tokenizer.encode(question, add_special_tokens=True)
    tokens = question_tokens + [start_id, latent_id, latent_id, end_id]

    input_ids = torch.tensor([tokens])
    attention_mask = torch.ones_like(input_ids)

    print(f"Input: {tokenizer.decode(tokens)}")
    print(f"Number of latent tokens: 2")

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask,
            max_new_tokens=20,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")

    print("PASSED: Generation test")


def test_dataset_processing():
    """Test dataset creation and processing."""
    print("\n" + "=" * 50)
    print("Test 4: Dataset Processing")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # Create sample data
    samples = create_sample_dataset(num_samples=10)

    # Save to temp file
    temp_file = "/tmp/test_coconut_data.json"
    with open(temp_file, "w") as f:
        json.dump(samples, f)

    print(f"Created {len(samples)} sample problems")

    # Test dataset loading
    dataset = get_dataset(temp_file, tokenizer)
    print(f"Loaded dataset with {len(dataset)} samples")

    # Create config mock
    class MockConfig:
        c_thought = 2
        max_latent_stage = 3
        pad_latent_to_max = True
        uniform_prob = 0.0
        no_cot = False

    configs = MockConfig()

    # Test question latent dataset (for inference)
    dataset_gen = get_question_latent_dataset(
        scheduled_stage=2,
        base_dataset_valid=dataset,
        configs=configs,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
    )
    print(f"Generated inference dataset with {len(dataset_gen)} samples")

    sample = dataset_gen[0]
    print(f"Sample input_ids length: {len(sample['input_ids'])}")
    print(f"Number of latent tokens: {sample['input_ids'].count(latent_id)}")

    # Test CoT latent dataset (for training)
    dataset_train = get_cot_latent_dataset(
        scheduled_stage=2,
        base_dataset=dataset,
        configs=configs,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
    )
    print(f"Generated training dataset with {len(dataset_train)} samples")

    sample = dataset_train[0]
    print(f"Training sample input_ids length: {len(sample['input_ids'])}")
    print(f"Training sample labels length: {len(sample['labels'])}")
    num_ignored = sum(1 for l in sample['labels'] if l == -100)
    print(f"Number of ignored positions in labels: {num_ignored}")

    # Clean up
    os.remove(temp_file)

    print("PASSED: Dataset processing test")


def test_collator():
    """Test the custom collator."""
    print("\n" + "=" * 50)
    print("Test 5: Collator")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    # Create sample features with different latent positions
    features = [
        {
            "input_ids": [1, 2, 3, latent_id, latent_id, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1],
            "labels": [-100, -100, -100, -100, -100, 4, 5],
            "position_ids": [0, 1, 2, 3, 4, 5, 6],
        },
        {
            "input_ids": [1, latent_id, 2, 3],
            "attention_mask": [1, 1, 1, 1],
            "labels": [-100, -100, 2, 3],
            "position_ids": [0, 1, 2, 3],
        },
    ]

    batch = collator(features)

    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"Batch position_ids shape: {batch['position_ids'].shape}")

    # Verify latent tokens are aligned
    latent_positions_0 = (batch['input_ids'][0] == latent_id).nonzero().squeeze().tolist()
    latent_positions_1 = (batch['input_ids'][1] == latent_id).nonzero().squeeze().tolist()

    print(f"Latent positions in sample 0: {latent_positions_0}")
    print(f"Latent positions in sample 1: {latent_positions_1}")

    # Check alignment
    if isinstance(latent_positions_0, int):
        latent_positions_0 = [latent_positions_0]
    if isinstance(latent_positions_1, int):
        latent_positions_1 = [latent_positions_1]

    first_latent_0 = min(latent_positions_0) if latent_positions_0 else -1
    first_latent_1 = min(latent_positions_1) if latent_positions_1 else -1

    if first_latent_0 != -1 and first_latent_1 != -1:
        print(f"First latent position aligned: {first_latent_0 == first_latent_1}")

    print("PASSED: Collator test")


def run_all_tests():
    """Run all tests."""
    print("\nRunning Coconut Qwen3-0.6B Tests")
    print("=" * 60)

    try:
        # Test 1: Model loading
        tokenizer, model, latent_id, start_id, end_id = test_model_loading()

        # Test 2: Coconut wrapper
        coconut_model = test_coconut_wrapper(tokenizer, model, latent_id, start_id, end_id)

        # Test 3: Generation
        test_generation(coconut_model, tokenizer, latent_id, start_id, end_id)

        # Test 4: Dataset processing
        test_dataset_processing()

        # Test 5: Collator
        test_collator()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    run_all_tests()
