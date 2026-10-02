"""LeRobot window to OpenVLA-OFT batch adapter.

Does not modify the RLDS loader. Subsampling and action relabeling stay in the
hybrid-motion-planner efficient / augmentation-ready export.
"""

from .lerobot_batch import HybridLeRobotOFTDataset, build_hybrid_lerobot_dataset

__all__ = ["HybridLeRobotOFTDataset", "build_hybrid_lerobot_dataset"]
