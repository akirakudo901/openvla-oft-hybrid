"""Numpy ``bounds_q99`` and the LIBERO gripper flip used by the RLDS path.

Arm action dimensions are mapped into [-1, 1]. ``quantile`` uses the 1st/99th
percentiles. ``minmax`` uses the dataset min and max, then copies those into
``q01`` and ``q99`` so LIBERO unnormalization (which always reads those keys)
inverts the same map. The gripper dimension stays absolute. Dimensions whose
min equals max become 0.
"""

from __future__ import annotations

import numpy as np

# Legacy efficient layout is gripper(2) + ee_pos(3) + ee_ori(3). OFT proprio is
# ee_pos(3) + ee_ori(3) + gripper(2), which is also the current LeRobot export.
EFFICIENT_TO_OFT_STATE = (2, 3, 4, 5, 6, 7, 0, 1)
ARM_ACTION_DIMS = 6
NORM_QUANTILE = "quantile"
NORM_MINMAX = "minmax"
NORM_MODES = (NORM_QUANTILE, NORM_MINMAX)


def libero_gripper_action(actions: np.ndarray) -> np.ndarray:
    """Clip the last action dim to [0, 1] and invert it so +1 is open and 0 is closed."""
    out = np.array(actions, dtype=np.float32, copy=True)
    gripper = np.clip(out[..., -1], 0.0, 1.0)
    out[..., -1] = 1.0 - gripper
    return out


def maybe_reorder_state_to_oft(state: np.ndarray, *, legacy_efficient_layout: bool) -> np.ndarray:
    """Reorder gripper-first state into the OFT proprio order when the export is legacy."""
    state = np.asarray(state, dtype=np.float32)
    if not legacy_efficient_layout:
        return state
    return state[..., list(EFFICIENT_TO_OFT_STATE)]


def bounds_q99(
    values: np.ndarray,
    *,
    q01: np.ndarray,
    q99: np.ndarray,
    mask: np.ndarray,
    min_values: np.ndarray,
    max_values: np.ndarray,
) -> np.ndarray:
    """Map masked dims with ``2 * (x - q01) / (q99 - q01) - 1``, clipped to [-1, 1]."""
    values = np.asarray(values, dtype=np.float32)
    low = np.asarray(q01, dtype=np.float32)
    high = np.asarray(q99, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    scaled = np.clip(2.0 * (values - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0)
    out = np.where(mask, scaled, values)
    zeros = np.asarray(min_values) == np.asarray(max_values)
    return np.where(zeros, np.float32(0.0), out).astype(np.float32, copy=False)


def fit_feature_stats(rows: np.ndarray) -> dict:
    """Per-dimension mean, std, min, max, q01, and q99 for one feature."""
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(-1, 1)
    return {
        "mean": rows.mean(axis=0),
        "std": rows.std(axis=0),
        "min": rows.min(axis=0),
        "max": rows.max(axis=0),
        "q01": np.quantile(rows, 0.01, axis=0).astype(np.float32),
        "q99": np.quantile(rows, 0.99, axis=0).astype(np.float32),
    }


def action_norm_mask(action_dim: int) -> np.ndarray:
    """Normalize the arm and leave the last (gripper) dimension absolute."""
    mask = np.zeros(action_dim, dtype=bool)
    mask[: min(ARM_ACTION_DIMS, action_dim)] = True
    if action_dim > ARM_ACTION_DIMS:
        mask[ARM_ACTION_DIMS:] = False
    return mask


def resolve_norm_mode(mode: str) -> str:
    """Return ``quantile`` or ``minmax``."""
    key = str(mode).strip().lower()
    if key not in NORM_MODES:
        raise ValueError(f"action_norm must be one of {NORM_MODES}, got {mode!r}")
    return key

# TODO this is very close to hacking the code... make modifications that are more appropriate later
def bind_unnorm_bounds(feature_stats: dict, mode: str) -> dict:
    """Make ``q01``/``q99`` the bounds training will apply.

    LIBERO unnormalization always inverts ``2 * (x - q01) / (q99 - q01) - 1``
    using the saved ``q01`` and ``q99``. Quantile mode leaves the percentiles.
    Min-max mode copies ``min`` and ``max`` into those keys so the existing
    inverse matches, without changing unnormalization code. Empirical
    percentiles are kept on ``quantile_01`` and ``quantile_99``.
    """
    mode = resolve_norm_mode(mode)
    feature_stats["quantile_01"] = np.array(feature_stats["q01"], dtype=np.float32, copy=True)
    feature_stats["quantile_99"] = np.array(feature_stats["q99"], dtype=np.float32, copy=True)
    if mode == NORM_MINMAX:
        feature_stats["q01"] = np.array(feature_stats["min"], dtype=np.float32, copy=True)
        feature_stats["q99"] = np.array(feature_stats["max"], dtype=np.float32, copy=True)
    return feature_stats


def dataset_statistics(
    *,
    dataset_name: str,
    actions: np.ndarray,
    proprios: np.ndarray,
    num_trajectories: int,
    norm_mode: str = NORM_QUANTILE,
) -> dict:
    """Stats dict shaped like the RLDS ``dataset_statistics.json`` payload."""
    mode = resolve_norm_mode(norm_mode)
    action_stats = bind_unnorm_bounds(fit_feature_stats(actions), mode)
    action_stats["mask"] = action_norm_mask(actions.shape[-1])
    proprio_stats = bind_unnorm_bounds(fit_feature_stats(proprios), mode)
    proprio_stats["mask"] = np.ones(proprios.shape[-1], dtype=bool)
    return {
        dataset_name: {
            "action": action_stats,
            "proprio": proprio_stats,
            "num_transitions": int(actions.shape[0]),
            "num_trajectories": int(num_trajectories),
        }
    }
