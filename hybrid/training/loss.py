"""MP/L-split chunk loss for hybrid OpenVLA-OFT.

Action L1 and BIO cross-entropy use :mod:`hybrid_eval.segment.losses`, so the
empty-mask and MP-vs-L means match ACT segment training. 
Cross-entropy runs in fp32 even when the label head is bf16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from hybrid_eval.segment.losses import (
    label_targets,
    label_valid_mask,
    masked_action_loss_mean,
    mp_l_action_masks,
    segment_label_ce,
)

LABEL_FEATURE_KEY = "frame_label_int"


def compute_hybrid_loss(
    actions_hat: Tensor,
    actions: Tensor,
    label_logits: Tensor,
    frame_labels: Tensor,
    *,
    is_pad: Tensor | None = None,
    mp_l1_weight: float = 1.0,
    mp_ce_weight: float = 1.0,
    l_ce_weight: float = 1.0,
    num_label_classes: int | None = None,
    label_feature_key: str = LABEL_FEATURE_KEY,
) -> tuple[Tensor, dict[str, float]]:
    """Return ``weighted_l1 + weighted_label_ce`` and detached component logs.

    ``weighted_l1 = l_l1 + mp_l1_weight * mp_l1``. Label CE is
    ``mp_ce_weight * mp_ce + l_ce_weight * l_ce``. An empty MP or L side
    contributes 0. ``is_pad`` is ``(B, T)`` with True on steps past the episode;
    those steps are dropped from both terms. When ``is_pad`` is omitted, every
    step is valid.

    ``actions_hat`` and ``actions`` are ``(B, T, act_dim)``. ``label_logits`` is
    ``(B, T, C)``. ``frame_labels`` is ``(B, T)`` or ``(B, T, 1)`` integer BIO ids.
    """
    if actions_hat.shape != actions.shape:
        raise ValueError(
            f"actions_hat shape {tuple(actions_hat.shape)} != actions shape {tuple(actions.shape)}"
        )
    if actions_hat.ndim != 3:
        raise ValueError(f"actions must be (B, T, act_dim), got ndim {actions_hat.ndim}")
    batch_size, chunk_len, _act_dim = actions_hat.shape
    if label_logits.ndim != 3 or label_logits.shape[:2] != (batch_size, chunk_len):
        raise ValueError(
            "label_logits must be "
            f"({batch_size}, {chunk_len}, C), got {tuple(label_logits.shape)}"
        )

    if is_pad is None:
        is_pad = torch.zeros(
            actions_hat.shape[:2], dtype=torch.bool, device=actions_hat.device
        )
    if is_pad.shape != (batch_size, chunk_len):
        raise ValueError(
            f"is_pad must be ({batch_size}, {chunk_len}), got {tuple(is_pad.shape)}"
        )
    is_pad = is_pad.to(device=actions_hat.device, dtype=torch.bool)

    batch: dict[str, Tensor] = {
        label_feature_key: frame_labels,
        "action_is_pad": is_pad,
    }
    abs_err = F.l1_loss(actions, actions_hat, reduction="none")
    action_valid_mask = (~is_pad).unsqueeze(-1)
    mp_action_mask, l_action_mask = mp_l_action_masks(
        batch,
        action_valid_mask,
        label_feature_key=label_feature_key,
    )
    mp_l1_loss = masked_action_loss_mean(abs_err, mp_action_mask)
    l_l1_loss = masked_action_loss_mean(abs_err, l_action_mask)
    weighted_l1_loss = l_l1_loss + mp_l1_weight * mp_l1_loss

    targets = label_targets(batch, label_feature_key)
    valid_labels = label_valid_mask(batch, label_feature_key)
    # Padded steps are masked out of the mean, but cross-entropy still indexes
    # every target. Fill those ids so an ignore value cannot fault the kernel.
    ce_targets = targets.masked_fill(~valid_labels, 0)
    weighted_label_ce_loss, label_loss_dict = segment_label_ce(
        label_logits.float(),
        ce_targets,
        valid_labels,
        mp_ce_weight=mp_ce_weight,
        l_ce_weight=l_ce_weight,
        num_label_classes=num_label_classes,
    )

    loss = weighted_l1_loss + weighted_label_ce_loss
    loss_dict = {
        "mp_l1_loss": mp_l1_loss.item(),
        "l_l1_loss": l_l1_loss.item(),
        "weighted_l1_loss": weighted_l1_loss.item(),
        **label_loss_dict,
    }
    return loss, loss_dict
