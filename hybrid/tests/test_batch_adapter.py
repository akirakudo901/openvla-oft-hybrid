"""Fake LeRobot window mapped to an OFT training sample."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from hybrid.datasets.lerobot_batch import PRIMARY_IMAGE_KEY, HybridLeRobotOFTDataset
from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK


class _Tokenizer:
    vocab_size = 32000

    def decode(self, token_ids):
        return "a" * len(token_ids)

    def batch_decode(self, rows):
        return [self.decode(row) for row in rows]

    def __call__(self, _text, add_special_tokens=True):
        del add_special_tokens
        return SimpleNamespace(input_ids=list(range(80)))


class _Processor:
    tokenizer = _Tokenizer()

    class image_processor:
        @staticmethod
        def apply_transform(image: Image.Image) -> torch.Tensor:
            return torch.tensor(np.asarray(image).shape)


class _Source:
    def __init__(self, window: dict, rows: list[dict]) -> None:
        self._window = window
        self.hf_dataset = rows
        self.num_episodes = 1
        self.meta = SimpleNamespace(
            features={"observation.state": {"names": ["ee_x", "ee_y", "ee_z", "ori_x", "ori_y", "ori_z", "g0", "g1"]}}
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        del idx
        return self._window


class HybridBatchAdapterTest(unittest.TestCase):
    def test_fake_window_becomes_oft_sample(self) -> None:
        chunk = NUM_ACTIONS_CHUNK
        action = np.zeros((chunk, ACTION_DIM), dtype=np.float32)
        action[:, 0] = np.linspace(0.0, 10.0, chunk)
        action[:, -1] = np.linspace(0.0, 1.0, chunk)
        state = np.zeros((chunk, 8), dtype=np.float32)
        labels = np.array([0, 2] + [1] * (chunk - 2), dtype=np.int64)
        pad = np.array([False] * (chunk - 1) + [True])
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        window = {
            "action": action,
            "task": "Pick Up The Cup",
            PRIMARY_IMAGE_KEY: image,
            "frame_label_int": labels,
            "action_is_pad": pad,
            "observation.state": state[0],
        }
        rows = [
            {"action": action[i], "observation.state": state[i]}
            for i in range(chunk)
        ]
        dataset = HybridLeRobotOFTDataset(
            _Source(window, rows),
            _Processor(),
            dataset_name="libero_spatial",
            use_wrist_image=False,
            use_proprio=False,
        )
        sample = dataset[0]

        self.assertEqual(sample["actions"].shape, (chunk, ACTION_DIM))
        self.assertEqual(sample["actions"].dtype, np.float32)
        flipped_gripper = 1.0 - np.clip(np.linspace(0.0, 1.0, chunk), 0.0, 1.0)
        self.assertTrue(np.allclose(sample["actions"][:, -1], flipped_gripper))
        # Arm dim 0 is scaled into [-1, 1]. Constant arm dims collapse to 0.
        self.assertTrue(np.all(sample["actions"][:, 0] >= -1.0))
        self.assertTrue(np.all(sample["actions"][:, 0] <= 1.0))
        self.assertTrue(np.allclose(sample["actions"][:, 1:-1], 0.0))
        self.assertTrue(torch.equal(sample["frame_label_int"], torch.tensor(labels)))
        self.assertTrue(torch.equal(sample["is_pad"], torch.tensor(pad)))
        self.assertEqual(sample["dataset_name"], "libero_spatial")
        self.assertTrue(torch.all(sample["labels"][:2] == IGNORE_INDEX))
        self.assertEqual(tuple(sample["pixel_values"].tolist()), (4, 4, 3))
        self.assertIn("libero_spatial", dataset.dataset_statistics)
        self.assertFalse(bool(dataset.dataset_statistics["libero_spatial"]["action"]["mask"][-1]))


class EpisodeSplitTest(unittest.TestCase):
    def test_holds_out_the_tail_and_keeps_a_train_episode(self) -> None:
        from hybrid.datasets.lerobot_batch import split_episode_ids

        train_ids, val_ids = split_episode_ids(10, 0.1)
        self.assertEqual(train_ids, list(range(9)))
        self.assertEqual(val_ids, [9])
        train_ids, val_ids = split_episode_ids(2, 0.9)
        self.assertEqual(train_ids, [0])
        self.assertEqual(val_ids, [1])


if __name__ == "__main__":
    unittest.main()
