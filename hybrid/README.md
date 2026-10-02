# Hybrid OpenVLA-OFT

This directory is the hybrid-motion-planner addition to OpenVLA-OFT. Upstream files stay as they are:

- `vla-scripts/finetune.py`
- `prismatic/models/action_heads.py`
- `prismatic/vla/datasets/rlds/`

The hybrid fine-tune is still LoRA on the VLM plus a full L1 action head. The extra pieces are a per-chunk BIO label head and MP/L-split losses. Later modules import the existing helpers (`init_module`, `wrap_ddp`, LoRA setup, `L1RegressionActionHead`) instead of forking them.

## Two data paths

### Unchanged RLDS path

`libero_*_no_noops` training is untouched. One TFDS episode is a trajectory with images, `observation/state`, a 7-D action, and `language_instruction`. The existing loader then:

1. `libero_dataset_transform` keeps arm deltas, clips the gripper to `[0, 1]`, and inverts it so +1 is open and 0 is close. `EEF_state` is `state[:, :6]` (xyz + axis-angle); `gripper_state` is `state[:, -2:]`.
2. `restructure` keeps images, an 8-D proprio, the action, language, and `dataset_name`.
3. `bounds_q99` normalizes the arm action. The gripper is absolute and is not normalized.
4. `chunk_act_obs` builds an action window of length `NUM_ACTIONS_CHUNK` (8 on LIBERO). Steps past the end of the episode repeat the last action. There is no pad mask.
5. `RLDSBatchTransform` emits `pixel_values`, token ids, action-token labels, and float `actions`.

That path has no `frame_label_int`, and its timesteps are not the labeled HDF5 timeline (no-op frames were dropped when the RLDS set was built). This package does not subsample or relabel inside that TF graph.

### LeRobot adapter (what this package trains on)

Subsampling and pose-endpoint relabeling already live in hybrid-motion-planner (`dataset/core/frame_subsample.py`, `dataset/core/action_relabel.py`). Their output is an efficient or augmentation-ready LeRobot export. `datasets/` will only read that export and emit the same batch dict the OFT forward pass expects:

- Images through the Prismatic image transform.
- Proprio reordered to OFT layout: xyz + axis-angle, then the 2-D gripper.
- Actions and `frame_label_int` over the same future window.
- Gripper convention matched to `libero_dataset_transform` before tokenization and before the L1 target.
- Arm actions normalized with bounds fit on this export (`--action_norm quantile` or `minmax`). Quantile uses the 1st/99th percentiles. Min-max uses the dataset min and max, then copies those into `q01`/`q99` in `dataset_statistics.json`. LIBERO unnormalization always inverts `q01`/`q99`, so both modes decode with the existing code. LeRobot mean/std is not applied on top.
- Past-the-end steps use the LeRobot pad mask. The hybrid loss ignores those steps.

`MpAugReadyTrainDataset` is used when the export has augmentation meta. Otherwise the adapter reads the efficient LeRobot dataset directly.

## Layout

| Path | Role |
| --- | --- |
| `models/` | Per-chunk BIO label head on the L1 action-token hidden states |
| `datasets/` | LeRobot window to OFT batch, plus quantile or min-max bounds on arm dims |
| `training/` | MP/L L1 and segment-label cross-entropy; label-head checkpoint save/load |
| `scripts/` | Fine-tune entry. Same LoRA + L1 loop, with the extra losses |
| `inference/` | Returns an action chunk and argmax BIO labels |
| `tests/` | Head shape, loss masks, and the batch adapter |

`run_libero_eval.py` and the hybrid rollout wrapper are not wired from here.
