"""Collate OFT action batches plus per-step BIO labels and pad masks."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from torch import Tensor

from prismatic.util.data_utils import PaddedCollatorForActionPrediction

from .loss import LABEL_FEATURE_KEY


class HybridActionCollator:
    """Upstream action collator, plus stacked ``frame_label_int`` and ``is_pad``.

    ``actions`` may be a numpy array or a tensor. Pad masks default to all-valid
    when a sample does not provide ``is_pad`` or ``action_is_pad``.
    """

    def __init__(self, model_max_length: int, pad_token_id: int, padding_side: str = "right"):
        self.base = PaddedCollatorForActionPrediction(
            model_max_length, pad_token_id, padding_side=padding_side
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Tensor]:
        prepared = []
        for instance in instances:
            row = dict(instance)
            actions = row["actions"]
            if isinstance(actions, torch.Tensor):
                row["actions"] = actions.detach().cpu().numpy()
            prepared.append(row)
        batch = self.base(prepared)
        batch[LABEL_FEATURE_KEY] = torch.stack(
            [_as_label_row(instance[LABEL_FEATURE_KEY]) for instance in instances]
        )
        batch["is_pad"] = torch.stack([_as_pad_row(instance, batch[LABEL_FEATURE_KEY].shape[1]) for instance in instances])
        return batch


def _as_label_row(value) -> Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.long()
    if value.ndim == 2 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.ndim != 1:
        raise ValueError(f"{LABEL_FEATURE_KEY} must be (T,) or (T, 1), got {tuple(value.shape)}")
    return value


def _as_pad_row(instance: Dict, chunk_len: int) -> Tensor:
    raw = instance.get("is_pad", instance.get("action_is_pad"))
    if raw is None:
        return torch.zeros(chunk_len, dtype=torch.bool)
    if not isinstance(raw, torch.Tensor):
        raw = torch.as_tensor(np.asarray(raw))
    raw = raw.bool()
    if raw.ndim == 2 and raw.shape[-1] == 1:
        raw = raw.squeeze(-1)
    if raw.shape != (chunk_len,):
        raise ValueError(f"is_pad must be ({chunk_len},), got {tuple(raw.shape)}")
    return raw
