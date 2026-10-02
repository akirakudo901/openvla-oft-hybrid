"""Turn an efficient or augmentation-ready LeRobot dataset into OFT training samples.

Each sample matches ``RLDSBatchTransform``: current-frame image, action-token
prompt, and a chunk of continuous actions. BIO labels and the LeRobot pad mask
are passed through for the hybrid loss. Normalization is OpenVLA bounds
scaling, fit on per-step actions after the gripper flip, not LeRobot mean/std.
``action_norm=quantile`` uses the 1st/99th percentiles. ``action_norm=minmax``
uses min/max and stores those in ``q01``/``q99`` so LIBERO unnormalization
inverts the same map.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import IGNORE_INDEX, NUM_ACTIONS_CHUNK

from .normalize import (
    NORM_QUANTILE,
    bounds_q99,
    dataset_statistics,
    libero_gripper_action,
    maybe_reorder_state_to_oft,
    resolve_norm_mode,
)

PRIMARY_IMAGE_KEY = "observation.images.image"
WRIST_IMAGE_KEY = "observation.images.image2"
STATE_KEY = "observation.state"
LABEL_KEY = "frame_label_int"
LEGACY_STATE_FORMAT = "efficient_libero"


class HybridLeRobotOFTDataset(Dataset):
    """Map one LeRobot (or MP-aug-ready) window to an OpenVLA-OFT collator sample."""

    def __init__(
        self,
        source,
        processor,
        *,
        dataset_name: str,
        use_wrist_image: bool,
        use_proprio: bool,
        action_norm: str = NORM_QUANTILE,
    ):
        self.source = source
        self.dataset_name = dataset_name
        self.use_wrist_image = use_wrist_image
        self.use_proprio = use_proprio
        self.action_norm = resolve_norm_mode(action_norm)
        self.action_tokenizer = ActionTokenizer(processor.tokenizer)
        self.base_tokenizer = processor.tokenizer
        self.image_transform = processor.image_processor.apply_transform
        self._legacy_state = _legacy_efficient_state(source)
        actions, proprios, num_trajectories = _collect_step_rows(source, legacy_state=self._legacy_state)
        self.dataset_statistics = dataset_statistics(
            dataset_name=dataset_name,
            actions=actions,
            proprios=proprios,
            num_trajectories=num_trajectories,
            norm_mode=self.action_norm,
        )
        stats = self.dataset_statistics[dataset_name]
        self._action_stats = stats["action"]
        self._proprio_stats = stats["proprio"]

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int) -> dict:
        sample = self.source[int(idx)]
        actions = _as_numpy(sample["action"])
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        actions = libero_gripper_action(actions)
        actions = bounds_q99(
            actions,
            q01=self._action_stats["q01"],
            q99=self._action_stats["q99"],
            mask=self._action_stats["mask"],
            min_values=self._action_stats["min"],
            max_values=self._action_stats["max"],
        )
        if actions.shape[0] != NUM_ACTIONS_CHUNK:
            raise ValueError(
                f"Expected an action chunk of length {NUM_ACTIONS_CHUNK}, got {actions.shape[0]}"
            )

        lang = str(sample["task"]).lower()
        prompt_builder = PurePromptBuilder("openvla")
        action_chunk_string = "".join(self.action_tokenizer(actions))
        for role, value in (
            ("human", f"What action should the robot take to {lang}?"),
            ("gpt", action_chunk_string),
        ):
            prompt_builder.add_turn(role, value)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = torch.tensor(input_ids)
        input_ids = torch.tensor(input_ids)
        labels[: -(len(action_chunk_string) + 1)] = IGNORE_INDEX

        pixel_values = self.image_transform(_image_at(sample[PRIMARY_IMAGE_KEY], 0))
        row = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": self.dataset_name,
            "actions": np.asarray(actions, dtype=np.float32),
            LABEL_KEY: _label_chunk(sample[LABEL_KEY]),
            "is_pad": _pad_chunk(sample),
        }
        if self.use_wrist_image:
            row["pixel_values_wrist"] = self.image_transform(_image_at(sample[WRIST_IMAGE_KEY], 0))
        if self.use_proprio:
            state = maybe_reorder_state_to_oft(
                _as_numpy(sample[STATE_KEY]), legacy_efficient_layout=self._legacy_state
            )
            if state.ndim == 2:
                state = state[0]
            row["proprio"] = bounds_q99(
                state,
                q01=self._proprio_stats["q01"],
                q99=self._proprio_stats["q99"],
                mask=self._proprio_stats["mask"],
                min_values=self._proprio_stats["min"],
                max_values=self._proprio_stats["max"],
            )
        return row


def build_hybrid_lerobot_dataset(cfg, processor) -> HybridLeRobotOFTDataset:
    """Load ``cfg.lerobot_dataset_root`` and wrap it when augmentation-ready meta exists."""
    from dataset.loaders.mp_aug_ready_train_dataset import (
        is_augmentation_ready_dataset,
        wrap_mp_aug_ready_dataset,
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = _dataset_root(cfg.lerobot_dataset_root, cfg.dataset_name)
    meta_fps = _read_fps(dataset_root)
    dt = 1.0 / meta_fps
    delta_timestamps = {
        "action": [i * dt for i in range(NUM_ACTIONS_CHUNK)],
        LABEL_KEY: [i * dt for i in range(NUM_ACTIONS_CHUNK)],
    }
    base = LeRobotDataset(
        repo_id=dataset_root.name,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
    )
    source = base

    if is_augmentation_ready_dataset(dataset_root):
        # Efficient virtual indices and relabeled actions, without extra MP-shift hops.
        source = wrap_mp_aug_ready_dataset(
            base,
            chunk_size=NUM_ACTIONS_CHUNK,
            mp_shift_max=0,
            enable_augmentation=False,
        )
    return HybridLeRobotOFTDataset(
        source,
        processor,
        dataset_name=cfg.dataset_name,
        use_wrist_image=cfg.num_images_in_input > 1,
        use_proprio=cfg.use_proprio,
        action_norm=getattr(cfg, "action_norm", NORM_QUANTILE),
    )


def _dataset_root(root: Path, dataset_name: str) -> Path:
    root = Path(root)
    if (root / "meta").is_dir():
        return root
    nested = root / dataset_name
    if (nested / "meta").is_dir():
        return nested
    raise FileNotFoundError(
        f"No LeRobot dataset at {root} or {nested} (expected a meta/ directory)."
    )


def _read_fps(dataset_root: Path) -> float:
    info_path = dataset_root / "meta" / "info.json"
    import json

    with info_path.open() as handle:
        info = json.load(handle)
    return float(info["fps"])


def _legacy_efficient_state(source) -> bool:
    meta = getattr(source, "meta", None)
    if meta is None and hasattr(source, "_base"):
        meta = source._base.meta
    features = getattr(meta, "features", {}) or {}
    state = features.get(STATE_KEY, {})
    layout = str(state.get("libero_state_format", "")).lower()
    if layout == LEGACY_STATE_FORMAT:
        return True
    names = state.get("names") or []
    return bool(names) and str(names[0]).startswith("gripper")


def _collect_step_rows(source, *, legacy_state: bool):
    """Per-step actions and proprio after the gripper flip and state reorder."""
    if hasattr(source, "_action_for_wrapper_idx"):
        actions = np.stack([np.asarray(row, dtype=np.float32) for row in source._action_for_wrapper_idx], axis=0)
        states = []
        for wrapper_idx, base_row in enumerate(source._base_hf_row_for_wrapper_idx):
            del wrapper_idx
            states.append(_as_numpy(source._base.hf_dataset[int(base_row)][STATE_KEY]))
        proprios = np.stack(states, axis=0)
        num_trajectories = len(getattr(source, "_virtual_segments", [])) or 1
    else:
        hf = source.hf_dataset
        actions = np.stack([_as_numpy(row["action"]) for row in hf], axis=0)
        proprios = np.stack([_as_numpy(row[STATE_KEY]) for row in hf], axis=0)
        num_trajectories = int(getattr(source, "num_episodes", 1))
    actions = libero_gripper_action(actions)
    proprios = maybe_reorder_state_to_oft(proprios, legacy_efficient_layout=legacy_state)
    return actions, proprios, num_trajectories


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image_at(value, index: int) -> Image.Image:
    array = _as_numpy(value)
    if array.ndim == 4:
        array = array[index]
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255.0).round().astype(np.uint8)
    else:
        array = array.astype(np.uint8, copy=False)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    return Image.fromarray(array)


def _label_chunk(value) -> torch.Tensor:
    labels = torch.as_tensor(_as_numpy(value)).long().reshape(-1)
    if labels.numel() != NUM_ACTIONS_CHUNK:
        raise ValueError(f"{LABEL_KEY} chunk must have length {NUM_ACTIONS_CHUNK}, got {labels.numel()}")
    return labels


def _pad_chunk(sample) -> torch.Tensor:
    raw = sample.get("action_is_pad", sample.get("is_pad"))
    if raw is None:
        return torch.zeros(NUM_ACTIONS_CHUNK, dtype=torch.bool)
    pad = torch.as_tensor(_as_numpy(raw)).bool().reshape(-1)
    if pad.numel() != NUM_ACTIONS_CHUNK:
        raise ValueError(f"action_is_pad must have length {NUM_ACTIONS_CHUNK}, got {pad.numel()}")
    return pad
