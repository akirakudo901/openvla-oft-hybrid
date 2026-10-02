"""L1 OpenVLA-OFT inference that also returns argmax BIO labels.

Uses ``OpenVLAForActionPrediction.predict_action`` so the prompt gets the same
empty action-token slots as upstream eval. The label head reads the action-token
hidden states that call already returns. Actions are recomputed from those
states so the batch dimension stays ``(B, NUM_ACTIONS_CHUNK, ACTION_DIM)``;
upstream ``predict_action`` reshapes them to a single chunk.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


@torch.inference_mode()
def predict(
    vla,
    action_head: nn.Module,
    label_head: nn.Module,
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
    pixel_values: Tensor,
    unnorm_key: str | None = None,
    proprio=None,
    proprio_projector: nn.Module | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(actions, labels)`` with shapes ``(B, k, action_dim)`` and ``(B, k)``.

    ``actions`` are unnormalized with the VLA's ``norm_stats`` when those stats
    are loaded. Otherwise the L1 head's normalized outputs are returned.
    """
    policy = _unwrap(vla)
    action_module = _unwrap(action_head)
    label_module = _unwrap(label_head)
    proprio_module = _unwrap(proprio_projector) if proprio_projector is not None else None

    _normalized_or_env, actions_hidden_states = policy.predict_action(
        input_ids=input_ids,
        unnorm_key=unnorm_key,
        proprio=proprio,
        proprio_projector=proprio_module,
        action_head=action_module,
        use_film=False,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
    )
    normalized_actions = action_module.predict_action(actions_hidden_states)
    label_dtype = next(label_module.parameters()).dtype
    labels = label_module.predict_labels(actions_hidden_states.to(dtype=label_dtype))
    actions = _to_env_actions(policy, normalized_actions, unnorm_key)
    return actions, labels


def _unwrap(module):
    return module.module if isinstance(module, nn.parallel.DistributedDataParallel) else module


def _to_env_actions(vla, normalized_actions: Tensor, unnorm_key: str | None) -> Tensor:
    if not getattr(vla, "norm_stats", None):
        return normalized_actions
    env_actions = vla._unnormalize_actions(
        normalized_actions.float().detach().cpu().numpy(),
        unnorm_key,
    )
    return torch.as_tensor(env_actions, device=normalized_actions.device, dtype=torch.float32)
