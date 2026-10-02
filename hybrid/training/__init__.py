"""MP/L-split action L1, segment-label cross-entropy, and label-head checkpoints."""

from .loss import LABEL_FEATURE_KEY, compute_hybrid_loss

__all__ = ["LABEL_FEATURE_KEY", "compute_hybrid_loss"]
