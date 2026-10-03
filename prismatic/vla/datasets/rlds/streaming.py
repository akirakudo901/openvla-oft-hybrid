"""
streaming.py

Streaming RLDS source for rolling-window training: samples trajectories from whichever RLDS datasets are currently
"live" in a window directory, so a background producer can add new data shards and retire old ones during a single
long training run (used for MolmoBot-Data; see experiments/robot/molmospaces/data/stream_producer.py).

Window protocol (the producer owns the directory, this module only reads it):
    <window_dir>/active/<dataset_name>/<version>/   a complete TFDS dataset, moved in atomically once fully written
    <window_dir>/active/<dataset_name>/RETIRED      marker: stop sampling it (the producer deletes it after a grace
                                                    period much longer than RESCAN_SECONDS, so open readers are safe)

Every live dataset is sampled with equal probability, which gives each shard the same exposure while it is live.
"""

import os
import time
from typing import Iterator, List

import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from dlimp.dataset import _broadcast_metadata_rlds, _wrap

RETIRED_MARKER = "RETIRED"
RESCAN_SECONDS = 60


def list_live_datasets(window_dir: str) -> List[str]:
    """Names of the datasets under `<window_dir>/active` that are not marked as retired."""
    active_dir = os.path.join(window_dir, "active")
    if not os.path.isdir(active_dir):
        return []
    return [
        name
        for name in sorted(os.listdir(active_dir))
        if os.path.isdir(os.path.join(active_dir, name))
        and not os.path.exists(os.path.join(active_dir, name, RETIRED_MARKER))
    ]


def _episode_dataset(window_dir: str, name: str, split: str) -> tf.data.Dataset:
    """One shard's episodes, read the same way as `dl.DLataset.from_rlds` (steps left undecoded)."""
    dataset_dir = os.path.join(window_dir, "active", name)
    version = sorted(v for v in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, v)))[-1]
    builder = tfds.builder_from_directory(os.path.join(dataset_dir, version))
    return builder.as_dataset(
        split=split,
        shuffle_files=True,
        decoders={"steps": tfds.decode.SkipDecoding()},
        read_config=tfds.ReadConfig(skip_prefetch=True),
    )


def _trajectory_generator(window_dir: str, split: str, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    iterators, last_scan = {}, 0.0
    while True:
        if not iterators or time.time() - last_scan > RESCAN_SECONDS:
            live = set(list_live_datasets(window_dir))
            for name in set(iterators) - live:
                del iterators[name]
            for name in sorted(live - set(iterators)):
                iterators[name] = iter(tfds.as_numpy(_episode_dataset(window_dir, name, split).repeat()))
            last_scan = time.time()
            if not iterators:
                print(f"[streaming] no live datasets in {window_dir}; waiting for the producer")
                time.sleep(30)
                continue
        names = sorted(iterators)
        yield next(iterators[names[rng.integers(len(names))]])


def make_streaming_rlds_dataset(window_dir: str, split: str = "train") -> dl.DLataset:
    """
    Returns a DLataset of trajectories equivalent to `dl.DLataset.from_rlds(builder, split)`, but drawn from the
    live datasets of a rolling window. Never ends; picks up newly activated datasets within RESCAN_SECONDS.
    """
    live = list_live_datasets(window_dir)
    if not live:
        raise ValueError(f"No live datasets in {window_dir}/active; start the producer first")
    element_spec = _episode_dataset(window_dir, live[0], split).element_spec

    # Different sampling order per training process (torchrun sets RANK)
    seed = int(os.environ.get("RANK", 0))
    dataset = tf.data.Dataset.from_generator(
        lambda: _trajectory_generator(window_dir, split, seed), output_signature=element_spec
    )
    dataset = _wrap(lambda: dataset, False)()._apply_options()
    return dataset.enumerate().traj_map(_broadcast_metadata_rlds)
