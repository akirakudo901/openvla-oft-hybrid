"""Hybrid model pieces.

``ChunkLabelHead`` maps the same per-chunk action-token hidden states used by
``prismatic.models.action_heads.L1RegressionActionHead`` to BIO class logits.
"""

from .label_head import DEFAULT_NUM_LABEL_CLASSES, ChunkLabelHead

__all__ = ["ChunkLabelHead", "DEFAULT_NUM_LABEL_CLASSES"]
