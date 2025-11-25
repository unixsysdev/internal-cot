# Dataset processing for Coconut with Qwen3
# Adapted from the original Coconut implementation by Meta Platforms, Inc.

import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


def get_dataset(path: str, tokenizer: PreTrainedTokenizerBase, max_size: int = 1000000000) -> Dataset:
    """
    Load and tokenize a dataset from JSON format.

    The JSON file should contain a list of samples, each with:
    - "question": The problem statement
    - "steps": List of reasoning steps
    - "answer": The final answer

    Args:
        path: Path to the JSON data file
        tokenizer: Tokenizer for the Qwen3 model
        max_size: Maximum number of samples to load

    Returns:
        HuggingFace Dataset with tokenized fields
    """

    def tokenize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
        # Tokenize question with a newline
        question_tokenized = tokenizer.encode(
            sample["question"] + "\n", add_special_tokens=True
        )

        # Tokenize each reasoning step with newlines
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]

        # Tokenize the answer with "### " prefix and EOS token
        answer_tokenized = tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]

        return {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
        }

    # Load data from JSON
    data = json.load(open(path))[:max_size]
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]

    # Create HuggingFace Dataset
    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})

    # Process with distributed support
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and dist.is_initialized():
        if dist.get_rank() == 0:
            processed_dataset = [
                dataset.map(
                    tokenize_sample, remove_columns=list(dataset.features), num_proc=32
                )
            ]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        dataset = dataset.map(
            tokenize_sample, remove_columns=list(dataset.features), num_proc=32
        )

    # Verify tokenization is correct
    d = data[0]
    complete = d["question"] + "\n" + "\n".join(d["steps"]) + "\n### " + d["answer"]
    complete_tokenized = tokenizer.encode(complete, add_special_tokens=True) + [
        tokenizer.eos_token_id
    ]
    assembled = (
        dataset[0]["question_tokenized"]
        + list(itertools.chain.from_iterable(dataset[0]["steps_tokenized"]))
        + dataset[0]["answer_tokenized"]
    )
    assert complete_tokenized == assembled, "Tokenization verification failed"

    return dataset


@dataclass
class MyCollator:
    """
    Custom data collator that aligns latent token positions across the batch.

    This alignment is crucial for maximizing KV-cache reuse efficiency.
    The collator pads sequences so that latent tokens align across batch elements.

    Example alignment:
        xxxxxxxxxx<latent><latent>xxxxx--
        -----xxxxx<latent>xxxxxxxx-------
        ---xxxxxxx<latent><latent>xxxxxxx

    ("x" is word token, "-" is pad token)
    """

    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(
        self, features: List[Dict[str, Any]], return_tensors: str = None
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features with latent token alignment.

        Args:
            features: List of feature dictionaries from the dataset
            return_tensors: Format for output tensors ("pt" for PyTorch)

        Returns:
            Batched and padded tensors ready for model input
        """
        assert self.tokenizer.padding_side == "right", "Padding must be on the right"

        # Find the earliest latent token position in each sequence
        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:
            # Align latent tokens by padding at the beginning
            latest_earliest_latent = max(earliest_latent)

            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0

                # Add position IDs with offset for padding
                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )

                # Pad input_ids at the beginning
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]

                # Pad labels if present
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "labels"
                    ]

                # Pad attention mask
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        # Separate features that need special handling
        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
            }
            for feature in features
        ]

        # Pad the non-label features using HuggingFace utility
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        # Handle labels separately
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None

        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )

        # Pad labels and position_ids manually
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)
            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        return batch


def get_question_latent_dataset(
    scheduled_stage: int,
    base_dataset_valid: Dataset,
    configs,
    start_id: int,
    latent_id: int,
    end_id: int,
    no_special_marker: bool = False,
) -> Dataset:
    """
    Create a dataset for inference/validation with latent token placeholders.

    This prepares inputs with the format:
    question + <|start-latent|> + [<|latent|>]*k + <|end-latent|>

    where k = min(max_latent_stage, scheduled_stage) * c_thought

    Args:
        scheduled_stage: Current training stage number
        base_dataset_valid: Base dataset with tokenized questions
        configs: Configuration object with training parameters
        start_id: Token ID for <|start-latent|>
        latent_id: Token ID for <|latent|>
        end_id: Token ID for <|end-latent|>
        no_special_marker: If True, don't add start/end markers

    Returns:
        Dataset with input_ids ready for generation
    """

    def process_dataset(sample: Dict[str, Any]) -> Dict[str, Any]:
        if configs.pad_latent_to_max:
            max_latent_stage = configs.max_latent_stage
        else:
            max_latent_stage = min(
                configs.max_latent_stage, len(sample["steps_tokenized"])
            )

        k = min(max_latent_stage, scheduled_stage)
        k *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )

        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }

    return base_dataset_valid.map(
        process_dataset, remove_columns=list(base_dataset_valid.features), num_proc=32
    )


def get_cot_latent_dataset(
    scheduled_stage: int,
    base_dataset: Dataset,
    configs,
    start_id: int,
    latent_id: int,
    end_id: int,
    no_special_marker: bool = False,
    shuffle: bool = False,
) -> Dataset:
    """
    Create a training dataset with chain-of-thought and latent tokens.

    This prepares training data with the format:
    question + <|start-latent|> + [<|latent|>]*n + <|end-latent|> + remaining_steps + answer

    The first n_skip_steps reasoning steps are replaced with latent tokens.
    Loss is computed only on the remaining steps and answer (labels=-100 for ignored parts).

    Args:
        scheduled_stage: Current training stage number
        base_dataset: Base dataset with tokenized questions, steps, and answers
        configs: Configuration object with training parameters
        start_id: Token ID for <|start-latent|>
        latent_id: Token ID for <|latent|>
        end_id: Token ID for <|end-latent|>
        no_special_marker: If True, don't add start/end markers
        shuffle: If True, shuffle the dataset after processing

    Returns:
        Dataset ready for training with input_ids and labels
    """
    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample: Dict[str, Any]) -> Dict[str, Any]:
        # With some probability, randomly sample a different stage for data augmentation
        if random.random() < configs.uniform_prob:
            scheduled_stage_to_train = random.choice(
                list(range(len(sample["steps_tokenized"]) + 1))
            )
        else:
            scheduled_stage_to_train = scheduled_stage

        if scheduled_stage_to_train > configs.max_latent_stage:
            # Beyond max stage: skip all steps, use max latent tokens
            n_skip_steps = 10000  # Skip all
            if configs.pad_latent_to_max:
                n_latent_tokens = configs.max_latent_stage
            else:
                n_latent_tokens = min(
                    len(sample["steps_tokenized"]), configs.max_latent_stage
                )
        else:
            n_skip_steps, n_latent_tokens = (
                scheduled_stage_to_train,
                scheduled_stage_to_train,
            )

        if configs.no_cot:
            # No chain-of-thought: skip all steps
            n_skip_steps = 100  # Skip all steps
            n_latent_tokens = 0

        n_latent_tokens *= configs.c_thought

        # Build the token sequence
        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + list(
                itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
            )
            + sample["answer_tokenized"]
        )

        # Build labels with -100 for positions we don't want loss on
        return {
            "input_ids": tokens,
            "labels": [-100]
            * (
                len(sample["question_tokenized"])
                + n_latent_tokens
                + n_additional_tokens
            )
            + tokens[
                n_latent_tokens
                + n_additional_tokens
                + len(sample["question_tokenized"]) :
            ],
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
        }

    # Process with distributed support
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and dist.is_initialized():
        if dist.get_rank() == 0:
            processed_dataset = base_dataset.map(
                process_dataset, remove_columns=list(base_dataset.features), num_proc=32
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        processed_dataset = base_dataset.map(
            process_dataset, remove_columns=list(base_dataset.features), num_proc=32
        )
        if shuffle:
            processed_dataset = processed_dataset.shuffle()
        dataset = processed_dataset

    return dataset


def create_sample_dataset(num_samples: int = 100) -> List[Dict[str, Any]]:
    """
    Create a simple sample dataset for testing.

    This generates synthetic reasoning problems with addition/subtraction.

    Args:
        num_samples: Number of samples to generate

    Returns:
        List of sample dictionaries with question, steps, and answer
    """
    samples = []
    for i in range(num_samples):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        c = random.randint(1, 50)

        question = f"John has {a} apples. He buys {b} more apples. Then he gives {c} apples to Mary. How many apples does John have now?"

        steps = [
            f"John starts with {a} apples.",
            f"After buying {b} more, John has {a} + {b} = {a + b} apples.",
            f"After giving {c} apples to Mary, John has {a + b} - {c} = {a + b - c} apples.",
        ]

        answer = str(a + b - c)

        samples.append({
            "question": question,
            "steps": steps,
            "answer": answer,
        })

    return samples
