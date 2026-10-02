"""Per-chunk BIO label head for hybrid OpenVLA-OFT.

Reads the same action-token hidden states as
``L1RegressionActionHead.predict_action``: ``(B, NUM_ACTIONS_CHUNK * ACTION_DIM, D)``
is reshaped to ``(B, NUM_ACTIONS_CHUNK, ACTION_DIM * D)``, then a linear layer
maps each step to ``num_label_classes`` logits. No extra prompt tokens.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

# B-MP, I-MP, B-L, I-L, E-MP. Matches FrameLabelEnum and SegmentPolicyConfigMixin.
DEFAULT_NUM_LABEL_CLASSES = 5


class ChunkLabelHead(nn.Module):
    """Linear BIO classifier on one concatenated action-token vector per chunk step."""

    def __init__(self, input_dim: int = 4096, num_label_classes: int = DEFAULT_NUM_LABEL_CLASSES):
        super().__init__()
        if num_label_classes < 2:
            raise ValueError(f"num_label_classes must be >= 2, got {num_label_classes}")
        self.input_dim = int(input_dim)
        self.num_label_classes = int(num_label_classes)
        self.num_actions_chunk = int(NUM_ACTIONS_CHUNK)
        self.action_dim = int(ACTION_DIM)
        self.proj = nn.Linear(self.input_dim * self.action_dim, self.num_label_classes)

    def forward(self, actions_hidden_states: Tensor) -> Tensor:
        """Return label logits ``(B, NUM_ACTIONS_CHUNK, num_label_classes)``.

        ``actions_hidden_states`` is the action-token slice of the last VLM hidden
        state, shape ``(B, NUM_ACTIONS_CHUNK * ACTION_DIM, input_dim)``.
        """
        if actions_hidden_states.ndim != 3:
            raise ValueError(
                "actions_hidden_states must be (B, NUM_ACTIONS_CHUNK * ACTION_DIM, D), "
                f"got ndim {actions_hidden_states.ndim}"
            )
        batch_size, num_tokens, hidden_dim = actions_hidden_states.shape
        expected_tokens = self.num_actions_chunk * self.action_dim
        if num_tokens != expected_tokens or hidden_dim != self.input_dim:
            raise ValueError(
                "actions_hidden_states must have shape "
                f"(B, {expected_tokens}, {self.input_dim}), got "
                f"{tuple(actions_hidden_states.shape)}"
            )
        rearranged = actions_hidden_states.reshape(batch_size, self.num_actions_chunk, -1)
        return self.proj(rearranged)

    @torch.no_grad()
    def predict_labels(self, actions_hidden_states: Tensor) -> Tensor:
        """Return argmax BIO class ids ``(B, NUM_ACTIONS_CHUNK)``."""
        return self.forward(actions_hidden_states).argmax(dim=-1)
