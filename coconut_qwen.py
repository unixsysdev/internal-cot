# Coconut implementation for Qwen3-0.6B
# Adapted from the original Coconut implementation by Meta Platforms, Inc.
# This implementation supports Qwen3 models for continuous latent reasoning.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from typing import Optional, List, Tuple, Union

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8


def extract_kv_cache_slice(kv_cache, end_pos: int):
    """
    Extract a slice of KV cache up to end_pos.
    Handles both legacy tuple format and new DynamicCache format.

    Args:
        kv_cache: Either a list of (key, value) tuples or a DynamicCache object
        end_pos: Position to slice up to

    Returns:
        KV cache in the same format as input, sliced to end_pos
    """
    if kv_cache is None:
        return None

    # Check if it's a DynamicCache object (new transformers format)
    if hasattr(kv_cache, 'key_cache') and hasattr(kv_cache, 'value_cache'):
        # New DynamicCache format - create a new cache with sliced values
        from transformers.cache_utils import DynamicCache
        new_cache = DynamicCache()
        for layer_idx in range(len(kv_cache.key_cache)):
            key = kv_cache.key_cache[layer_idx][:, :, :end_pos, :].contiguous()
            value = kv_cache.value_cache[layer_idx][:, :, :end_pos, :].contiguous()
            new_cache.update(key, value, layer_idx)
        return new_cache
    elif isinstance(kv_cache, (list, tuple)):
        # Legacy list of tuples format - convert to DynamicCache for compatibility
        try:
            from transformers.cache_utils import DynamicCache
            new_cache = DynamicCache()
            for layer_idx, (k, v) in enumerate(kv_cache):
                key = k[:, :, :end_pos, :].contiguous()
                value = v[:, :, :end_pos, :].contiguous()
                new_cache.update(key, value, layer_idx)
            return new_cache
        except ImportError:
            # Fallback to list format if DynamicCache not available
            return [
                (k[:, :, :end_pos, :], v[:, :, :end_pos, :])
                for k, v in kv_cache
            ]
    else:
        # Unknown format, return as-is
        return kv_cache


class CoconutQwen(nn.Module):
    """
    Coconut (Chain of Continuous Thought) model wrapper for Qwen3.

    This model enables reasoning in continuous latent space by feeding
    hidden states back as input embeddings instead of decoding to discrete tokens.

    The model switches between:
    - Language mode: Standard autoregressive token generation
    - Latent mode: Uses last hidden states as next input embeddings

    Special tokens:
    - <|latent|>: Placeholder for continuous thought positions
    - <|start-latent|>: Marks beginning of latent reasoning
    - <|end-latent|>: Marks end of latent reasoning
    """

    def __init__(
        self,
        base_causallm,
        latent_token_id: int,
        start_latent_id: int,
        end_latent_id: int,
        eos_token_id: int,
    ):
        """
        Initialize the Coconut wrapper for Qwen3.

        Args:
            base_causallm: The base Qwen3 causal language model
            latent_token_id: Token ID for <|latent|> placeholder
            start_latent_id: Token ID for <|start-latent|> marker
            end_latent_id: Token ID for <|end-latent|> marker
            eos_token_id: Token ID for end-of-sequence
        """
        super(CoconutQwen, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        # Qwen3 uses get_input_embeddings() like most modern models
        self.embedding = self.base_causallm.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs
    ) -> Outputs:
        """
        Forward pass with continuous thought feedback mechanism.

        This method:
        1. Identifies positions of latent tokens in the input
        2. Performs iterative forward passes, replacing latent token embeddings
           with the hidden states from the previous position
        3. Uses KV-cache for efficiency across multiple passes
        4. Computes loss only on non-latent tokens

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for loss computation [batch_size, seq_len]
            position_ids: Position IDs for positional encoding [batch_size, seq_len]

        Returns:
            Outputs namedtuple with loss, inputs_embeds, and logits
        """
        logits = []

        # Find all latent token positions in the batch
        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        # Group latent positions by batch index
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (may differ across batch)

        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            # Start by processing tokens before the earliest latent token
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        # Iterative forward passes for each latent token
        for pass_idx in range(max_n_latents):

            if kv_cache is None:
                # First forward pass - no cache available
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = 0

            else:
                # Subsequent passes - reuse KV cache for efficiency
                past_key_values = extract_kv_cache_slice(kv_cache, next_compute_range[0])

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )

                hidden_states_offset = next_compute_range[0]
                # When using KV-cache for first k tokens,
                # hidden_states skips positions [0, k)
                # so we need this offset to correctly index hidden states

            logits.append(outputs.logits)

            # Update the range for the next forward pass
            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            # Get last layer hidden states
            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            # === Continuous Thought Feedback Mechanism ===
            # Replace latent token embeddings with hidden states from preceding position

            # Identify which positions need to be filled with continuous thoughts
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # Convert inputs_embeds to a list structure to avoid in-place operations
            # This is necessary for proper gradient computation
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # Replace latent token positions with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                # The continuous thought is the hidden state from the preceding position
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            # Reassemble inputs_embeds from the list
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # Final forward pass for remaining tokens after all latent positions
        past_key_values = extract_kv_cache_slice(kv_cache, next_compute_range[0]) if kv_cache else None

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        # Concatenate all logits and compute loss
        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self, mode: bool = True):
        """Set the model to training mode."""
        self.base_causallm.train(mode)
        return self

    def eval(self):
        """Set the model to evaluation mode."""
        self.base_causallm.eval()
        return self

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 16,
        output_embedding: bool = False,
        synced_gpus: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively after processing continuous thoughts.

        This method:
        1. Processes the input through the continuous thought mechanism
        2. Generates new tokens autoregressively using greedy decoding
        3. Stops on EOS token

        Args:
            input_ids: Input token IDs [1, seq_len] (batch_size must be 1)
            attention_mask: Attention mask (not actively used in generation)
            max_new_tokens: Maximum number of new tokens to generate
            output_embedding: If True, also return the final embeddings
            synced_gpus: If True, ensure same number of forward passes across GPUs

        Returns:
            Generated token IDs tensor, optionally with embeddings
        """
        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "Only batch_size == 1 is supported for generation"

        tokens = input_ids[0].detach().tolist()

        # Initial forward pass through continuous thought mechanism
        labels = input_ids.clone()  # Placeholder, not used for loss
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # Get the first generated token from the final logits
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # Continue generating tokens autoregressively
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # In FSDP, ensure the same number of forward passes across all devices
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            return torch.tensor(tokens, device=input_ids.device).view(1, -1), new_inputs_embeds
        else:
            return torch.tensor(tokens, device=input_ids.device).view(1, -1)

    def parameters(self, recurse: bool = True):
        """Return parameters of the underlying model."""
        return self.base_causallm.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        """Return the state dict with prefixed keys."""
        base_state = self.base_causallm.state_dict(*args, **kwargs)
        return {"base_causallm." + k: v for k, v in base_state.items()}

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict, handling the prefix."""
        # Check if the state dict has the prefix
        if any(k.startswith("base_causallm.") for k in state_dict.keys()):
            # Remove prefix for loading into base model
            new_state_dict = {
                k.replace("base_causallm.", ""): v
                for k, v in state_dict.items()
                if k.startswith("base_causallm.")
            }
            return self.base_causallm.load_state_dict(new_state_dict, strict=strict)
        else:
            # Load directly without prefix
            return self.base_causallm.load_state_dict(state_dict, strict=strict)
