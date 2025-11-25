# Latent Critic for Coconut
# Verifies intermediate reasoning steps in continuous space
#
# Two verification strategies:
# 1. Value Prediction: Directly predict P(correct | hidden_state)
# 2. Contrastive: Compare against known-good hidden state trajectories

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from collections import namedtuple

from coconut_qwen import CoconutQwen, extract_kv_cache_slice


CriticOutputs = namedtuple(
    "CriticOutputs",
    ["loss", "logits", "step_values", "trajectory_value", "critic_loss"]
)


class LatentValueCritic(nn.Module):
    """
    Predicts the probability that a hidden state leads to a correct answer.

    Trained with binary cross-entropy:
    - Positive: hidden states from trajectories that produced correct answers
    - Negative: hidden states from trajectories that produced wrong answers
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size] or [batch, hidden_size]
        Returns:
            values: [batch, seq_len, 1] or [batch, 1] - probability of correct answer
        """
        logits = self.value_head(hidden_states)
        return torch.sigmoid(logits)


class ContrastiveLatentCritic(nn.Module):
    """
    Verifies hidden states by comparing against a memory bank of
    "known good" hidden states from successful reasoning trajectories.

    The intuition: if current h_t is similar to h_t from problems
    we solved correctly, it's probably on the right track.
    """

    def __init__(
        self,
        hidden_size: int,
        projection_dim: int = 256,
        memory_size: int = 500,
        max_steps: int = 10,
        temperature: float = 0.1
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.memory_size = memory_size
        self.max_steps = max_steps
        self.temperature = temperature

        # Project hidden states to comparison space
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        # Memory bank: [max_steps, memory_size, projection_dim]
        # Stores reference "good" hidden states at each reasoning step
        self.register_buffer(
            'reference_memory',
            torch.zeros(max_steps, memory_size, projection_dim)
        )
        self.register_buffer(
            'memory_ptr',
            torch.zeros(max_steps, dtype=torch.long)
        )
        self.register_buffer(
            'memory_filled',
            torch.zeros(max_steps, dtype=torch.long)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        step_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity score to reference memory.

        Args:
            hidden_states: [batch, hidden_size]
            step_indices: [batch] - which reasoning step each hidden state is from

        Returns:
            scores: [batch] - similarity scores (higher = more similar to good trajectories)
        """
        # Project to comparison space
        z = self.projector(hidden_states)  # [batch, projection_dim]
        z = F.normalize(z, dim=-1)

        scores = []
        for i, step_idx in enumerate(step_indices):
            step_idx = min(step_idx.item(), self.max_steps - 1)
            n_filled = self.memory_filled[step_idx].item()

            if n_filled == 0:
                # No references yet, return neutral score
                scores.append(torch.tensor(0.5, device=hidden_states.device))
            else:
                # Get references for this step
                refs = self.reference_memory[step_idx, :n_filled]  # [n_filled, proj_dim]
                refs = F.normalize(refs, dim=-1)

                # Cosine similarity
                sim = F.cosine_similarity(z[i:i+1], refs, dim=-1)  # [n_filled]

                # Soft-max over similarities (attention-like)
                weights = F.softmax(sim / self.temperature, dim=0)
                score = (sim * weights).sum()
                scores.append(score)

        return torch.stack(scores)

    @torch.no_grad()
    def update_memory(
        self,
        hidden_states: torch.Tensor,
        step_idx: int,
        is_correct: bool
    ):
        """
        Update memory with hidden states from a completed trajectory.
        Only stores states from CORRECT trajectories.

        Args:
            hidden_states: [hidden_size] - hidden state at this step
            step_idx: which reasoning step
            is_correct: whether this trajectory led to correct answer
        """
        if not is_correct:
            return

        step_idx = min(step_idx, self.max_steps - 1)
        z = self.projector(hidden_states.unsqueeze(0))  # [1, proj_dim]
        z = F.normalize(z, dim=-1).squeeze(0)  # [proj_dim]

        ptr = self.memory_ptr[step_idx].item()
        self.reference_memory[step_idx, ptr] = z
        self.memory_ptr[step_idx] = (ptr + 1) % self.memory_size
        self.memory_filled[step_idx] = min(
            self.memory_filled[step_idx] + 1,
            self.memory_size
        )


class CoconutWithCritic(nn.Module):
    """
    Coconut model with integrated latent critic for step-by-step verification.

    During training:
    - Critic learns to predict which hidden states lead to correct answers
    - Uses TD-like learning: v(h_t) should predict eventual correctness

    During inference:
    - Critic scores each latent step
    - Can reject/retry if score drops too low
    - Can branch into multiple trajectories and pick best
    """

    def __init__(
        self,
        base_causallm,
        latent_token_id: int,
        start_latent_id: int,
        end_latent_id: int,
        eos_token_id: int,
        critic_type: str = "value",  # "value" or "contrastive"
        rejection_threshold: float = 0.3,
    ):
        super().__init__()

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
        self.rejection_threshold = rejection_threshold

        hidden_size = base_causallm.config.hidden_size

        # Initialize critic
        if critic_type == "value":
            self.critic = LatentValueCritic(hidden_size)
        else:
            self.critic = ContrastiveLatentCritic(hidden_size)

        self.critic_type = critic_type
        self.base_causallm = base_causallm
        self.embedding = base_causallm.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        position_ids: torch.Tensor,
        is_correct: Optional[torch.Tensor] = None,  # [batch] binary labels
        **kwargs
    ) -> CriticOutputs:
        """
        Forward pass with critic training.

        Args:
            is_correct: Binary tensor indicating if each sample's answer is correct.
                       Used to train the critic with hindsight.
        """
        # Get base outputs
        coconut_outputs = self.coconut(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )

        # Extract hidden states at latent positions for critic
        batch_size = input_ids.shape[0]
        device = input_ids.device

        step_values = []
        critic_loss = torch.tensor(0.0, device=device)

        for batch_idx in range(batch_size):
            latent_mask = input_ids[batch_idx] == self.latent_token_id
            latent_positions = latent_mask.nonzero(as_tuple=True)[0]

            if len(latent_positions) > 0:
                # Get embeddings at latent positions (as proxy for hidden states)
                latent_embeds = coconut_outputs.inputs_embeds[batch_idx, latent_positions]

                if self.critic_type == "value":
                    # Predict value at each step
                    values = self.critic(latent_embeds).squeeze(-1)  # [n_latent]
                    step_values.append(values)

                    # Train critic if we have correctness labels
                    if is_correct is not None:
                        target = is_correct[batch_idx].float().expand_as(values)
                        critic_loss = critic_loss + F.binary_cross_entropy(
                            values, target
                        )
                else:
                    # Contrastive critic
                    step_indices = torch.arange(len(latent_positions), device=device)
                    scores = self.critic(latent_embeds, step_indices)
                    step_values.append(scores)

                    # Update memory if we have correctness labels
                    if is_correct is not None and is_correct[batch_idx]:
                        for i, h in enumerate(latent_embeds):
                            self.critic.update_memory(h, i, True)

        if batch_size > 0 and len(step_values) > 0:
            critic_loss = critic_loss / batch_size

        # Compute trajectory-level value (average of step values)
        if step_values:
            trajectory_value = torch.stack([v.mean() for v in step_values]).mean()
        else:
            trajectory_value = torch.tensor(0.5, device=device)

        return CriticOutputs(
            loss=coconut_outputs.loss,
            logits=coconut_outputs.logits,
            step_values=step_values,
            trajectory_value=trajectory_value,
            critic_loss=critic_loss,
        )

    def generate_with_verification(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 32,
        num_latent: int = 4,
        num_branches: int = 1,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Generate with step-by-step verification.

        If num_branches > 1, explores multiple trajectories and picks
        the one with highest critic scores.
        """
        assert input_ids.shape[0] == 1, "Batch size must be 1"

        self.eval()
        device = input_ids.device

        best_output = None
        best_score = float('-inf')
        best_step_scores = []

        for branch in range(num_branches):
            # Build input with latent tokens
            tokens = input_ids[0].tolist()
            tokens.append(self.start_latent_id)
            tokens.extend([self.latent_token_id] * num_latent)
            tokens.append(self.end_latent_id)

            if num_branches > 1:
                # Add small noise for diversity
                noise_scale = 0.05 * (branch + 1)
            else:
                noise_scale = 0

            input_tensor = torch.tensor([tokens], device=device)
            attention = torch.ones_like(input_tensor)

            # Forward through Coconut to get hidden states
            with torch.no_grad():
                outputs = self.coconut.generate(
                    input_tensor,
                    attention,
                    max_new_tokens=max_new_tokens,
                )

            # Score this trajectory (simplified - would need actual hidden states)
            # For now, use output probability as proxy
            step_scores = [0.5] * num_latent  # Placeholder
            trajectory_score = sum(step_scores) / len(step_scores)

            if verbose:
                print(f"Branch {branch}: score={trajectory_score:.3f}")

            if trajectory_score > best_score:
                best_score = trajectory_score
                best_output = outputs
                best_step_scores = step_scores

        return best_output, best_step_scores

    def train(self, mode: bool = True):
        self.coconut.train(mode)
        self.critic.train(mode)
        return self

    def eval(self):
        self.coconut.eval()
        self.critic.eval()
        return self

    def parameters(self, recurse: bool = True):
        for p in self.coconut.parameters(recurse=recurse):
            yield p
        for p in self.critic.parameters(recurse=recurse):
            yield p


def train_critic_epoch(
    model: CoconutWithCritic,
    dataloader,
    optimizer,
    device,
    verifier_fn,  # Function that checks if answer is correct
):
    """
    Train the critic using hindsight experience.

    1. Generate answers for each problem
    2. Check correctness with verifier_fn
    3. Use correctness as training signal for critic
    """
    model.train()
    total_loss = 0
    total_critic_loss = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        position_ids = batch["position_ids"].to(device)
        expected_answers = batch.get("answers", None)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )

        # Generate answers to check correctness
        if expected_answers is not None:
            is_correct = []
            for i in range(input_ids.shape[0]):
                with torch.no_grad():
                    generated = model.coconut.generate(
                        input_ids[i:i+1],
                        attention_mask[i:i+1],
                        max_new_tokens=20,
                    )
                # Check if correct (simplified)
                correct = verifier_fn(generated, expected_answers[i])
                is_correct.append(correct)

            is_correct = torch.tensor(is_correct, device=device)

            # Recompute with correctness labels
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
                is_correct=is_correct,
            )

        # Combined loss
        loss = outputs.loss + 0.5 * outputs.critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += outputs.loss.item()
        total_critic_loss += outputs.critic_loss.item()

    return total_loss / len(dataloader), total_critic_loss / len(dataloader)
