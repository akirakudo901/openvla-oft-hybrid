"""Hybrid OpenVLA-OFT: a BIO label head and MP/L losses on the L1 LoRA fine-tune.

Upstream RLDS training (``vla-scripts/finetune.py`` and ``prismatic.vla.datasets.rlds``)
is unchanged. This package reads an augmentation-ready or efficient LeRobot export
and will emit the batch dict that the existing OFT forward pass expects.
"""
