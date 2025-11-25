# Adaptive Coconut implementation for Qwen3
# Extends CoconutQwen with confidence-based dynamic latent token count
#
# Key idea: The model learns to decide when it has "thought enough" by
# training a small classifier on hidden states to predict readiness.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from collections import namedtuple

from coconut_qwen import CoconutQwen, extract_kv_cache_slice

AdaptiveOutputs = namedtuple(
    "AdaptiveOutputs",
    ["loss", "inputs_embeds", "logits", "confidence_loss", "num_latent_used"]
)


class ConfidenceHead(nn.Module):
    """
    Small classifier that predicts whether the model is ready to answer.

    Takes hidden states as input and outputs a confidence score [0, 1].
    High confidence = ready to exit latent mode and generate answer.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, hidden_size] - last hidden state
        Returns:
            confidence: [batch_size, 1] - confidence scores
        """
        return self.classifier(hidden_states)


class AdaptiveCoconutQwen(nn.Module):
    """
    Coconut with adaptive latent token count based on confidence.

    During training:
    - Uses fixed latent count but trains confidence head to predict
      "ready" at the correct position (last latent before answer)

    During inference:
    - Dynamically adds latent tokens until confidence exceeds threshold
    - Stops early if confident, adds more if uncertain

    This allows the model to use more "thinking" for hard problems
    and less for easy ones.
    """

    def __init__(
        self,
        base_causallm,
        latent_token_id: int,
        start_latent_id: int,
        end_latent_id: int,
        eos_token_id: int,
        max_latent_tokens: int = 8,
        confidence_threshold: float = 0.7,
    ):
        super().__init__()

        # Base Coconut model
        self.coconut = CoconutQwen(
            base_causallm,
            latent_token_id=latent_token_id,
            start_latent_id=start_latent_id,
            end_latent_id=end_latent_id,
            eos_token_id=eos_token_id,
        )

        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id
        self.max_latent_tokens = max_latent_tokens
        self.confidence_threshold = confidence_threshold

        # Get hidden size from base model
        hidden_size = base_causallm.config.hidden_size

        # Confidence prediction head
        self.confidence_head = ConfidenceHead(hidden_size)

        # For accessing embeddings
        self.embedding = base_causallm.get_input_embeddings()
        self.base_causallm = base_causallm

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs
    ) -> AdaptiveOutputs:
        """
        Forward pass with confidence training.

        The confidence head is trained to output:
        - Low confidence (0) for early latent positions
        - High confidence (1) for the last latent position before answer
        """
        # Get base Coconut outputs
        coconut_outputs = self.coconut(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )

        # Get hidden states for confidence training
        # We need to identify latent positions and train confidence
        batch_size = input_ids.shape[0]
        device = input_ids.device

        confidence_loss = torch.tensor(0.0, device=device)
        num_latent = 0

        # Find latent token positions for each sample
        for batch_idx in range(batch_size):
            latent_mask = input_ids[batch_idx] == self.latent_token_id
            latent_positions = latent_mask.nonzero(as_tuple=True)[0]

            if len(latent_positions) > 0:
                num_latent = max(num_latent, len(latent_positions))

                # Get hidden states at latent positions
                # Note: We use the input embeddings that were modified by Coconut
                # For proper training, we'd need the actual hidden states
                # This is a simplified version - full impl would cache hidden states

                # Create target: 0 for all but last latent, 1 for last
                targets = torch.zeros(len(latent_positions), device=device)
                targets[-1] = 1.0  # Last latent should be confident

                # For now, use the embeddings as a proxy
                # In production, we'd extract actual hidden states from forward pass
                latent_embeds = coconut_outputs.inputs_embeds[batch_idx, latent_positions, :]

                # Predict confidence at each latent position
                confidence_preds = self.confidence_head(latent_embeds).squeeze(-1)

                # BCE loss for confidence prediction
                confidence_loss = confidence_loss + F.binary_cross_entropy(
                    confidence_preds, targets
                )

        if batch_size > 0 and num_latent > 0:
            confidence_loss = confidence_loss / batch_size

        return AdaptiveOutputs(
            loss=coconut_outputs.loss,
            inputs_embeds=coconut_outputs.inputs_embeds,
            logits=coconut_outputs.logits,
            confidence_loss=confidence_loss,
            num_latent_used=num_latent,
        )

    def generate_adaptive(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 32,
        min_latent: int = 1,
        max_latent: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate with adaptive latent token count.

        Starts with min_latent tokens, adds more until:
        - Confidence exceeds threshold, OR
        - max_latent is reached

        Args:
            input_ids: Input tokens (should NOT include latent tokens)
            attention_mask: Attention mask
            max_new_tokens: Max tokens to generate after latent reasoning
            min_latent: Minimum number of latent tokens
            max_latent: Maximum number of latent tokens
            confidence_threshold: Override default threshold
            verbose: Print confidence at each step

        Returns:
            (generated_ids, num_latent_used)
        """
        if max_latent is None:
            max_latent = self.max_latent_tokens
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        assert input_ids.shape[0] == 1, "Batch size must be 1 for adaptive generation"

        device = input_ids.device
        self.eval()

        # Start with question tokens
        current_tokens = input_ids[0].tolist()

        # Add start latent marker
        current_tokens.append(self.start_latent_id)

        # Iteratively add latent tokens and check confidence
        num_latent_used = 0

        for i in range(max_latent):
            # Add a latent token
            current_tokens.append(self.latent_token_id)
            num_latent_used += 1

            # Build input tensor
            input_tensor = torch.tensor([current_tokens], device=device)
            attention = torch.ones_like(input_tensor)
            position_ids = torch.arange(len(current_tokens), device=device).unsqueeze(0)

            # Forward pass to get hidden states
            with torch.no_grad():
                outputs = self.base_causallm(
                    input_ids=input_tensor,
                    attention_mask=attention,
                    position_ids=position_ids,
                    output_hidden_states=True,
                )

            # Get last hidden state at the last position
            last_hidden = outputs.hidden_states[-1][0, -1, :]  # [hidden_size]

            # Predict confidence
            confidence = self.confidence_head(last_hidden.unsqueeze(0)).item()

            if verbose:
                print(f"  Latent {num_latent_used}: confidence = {confidence:.3f}")

            # Check if we should stop
            if num_latent_used >= min_latent and confidence >= confidence_threshold:
                if verbose:
                    print(f"  -> Confident enough, stopping at {num_latent_used} latents")
                break

        # Add end latent marker
        current_tokens.append(self.end_latent_id)

        # Now generate the answer using the full Coconut mechanism
        input_tensor = torch.tensor([current_tokens], device=device)
        attention = torch.ones_like(input_tensor)

        with torch.no_grad():
            outputs = self.coconut.generate(
                input_tensor,
                attention,
                max_new_tokens=max_new_tokens,
            )

        return outputs, num_latent_used

    def train(self, mode: bool = True):
        self.coconut.train(mode)
        self.confidence_head.train(mode)
        return self

    def eval(self):
        self.coconut.eval()
        self.confidence_head.eval()
        return self

    def parameters(self, recurse: bool = True):
        """Return all parameters including confidence head."""
        for param in self.coconut.parameters(recurse=recurse):
            yield param
        for param in self.confidence_head.parameters(recurse=recurse):
            yield param

    def state_dict(self, *args, **kwargs):
        """Return combined state dict."""
        state = self.coconut.state_dict(*args, **kwargs)
        confidence_state = self.confidence_head.state_dict(*args, **kwargs)
        for k, v in confidence_state.items():
            state[f"confidence_head.{k}"] = v
        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict for both components."""
        coconut_state = {
            k: v for k, v in state_dict.items()
            if not k.startswith("confidence_head.")
        }
        confidence_state = {
            k.replace("confidence_head.", ""): v
            for k, v in state_dict.items()
            if k.startswith("confidence_head.")
        }

        self.coconut.load_state_dict(coconut_state, strict=strict)
        if confidence_state:
            self.confidence_head.load_state_dict(confidence_state, strict=strict)


def compute_problem_complexity(question: str) -> int:
    """
    Heuristic to estimate problem complexity for training data generation.
    Used to assign appropriate latent token counts during training.

    Returns estimated number of reasoning steps needed (1-5).
    """
    complexity = 1

    # More numbers = more complex
    import re
    numbers = re.findall(r'\d+', question)
    complexity += min(len(numbers) - 1, 2)

    # Certain keywords indicate complexity
    complex_keywords = ['total', 'remaining', 'difference', 'twice', 'half',
                       'percent', 'ratio', 'average', 'sum', 'product']
    for keyword in complex_keywords:
        if keyword.lower() in question.lower():
            complexity += 1
            break

    # Multiple sentences often mean multi-step
    sentences = question.count('.') + question.count('?')
    if sentences > 2:
        complexity += 1

    return min(max(complexity, 1), 5)
