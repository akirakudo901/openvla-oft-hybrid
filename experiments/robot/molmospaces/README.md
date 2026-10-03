# OpenVLA-OFT on MolmoSpaces

Fine-tune OpenVLA-OFT on [MolmoBot-Data](https://huggingface.co/datasets/allenai/molmobot-data) (Franka, simulated
demos) and evaluate it on the [MolmoSpaces](https://github.com/allenai/molmospaces) benchmarks (held-out houses
and objects). Code lives here and is edited on the laptop; data, models and runs live only on the GPU server.

## Layout

```
molmospaces/
├── molmospaces_utils.py     shared train/eval conventions: cameras, 224x224 letterbox, state & action format
├── data/                    MolmoBot-Data -> RLDS
│   ├── inspect_molmobot.py       print the raw HDF5/MP4 format of a downloaded shard
│   ├── convert_molmobot_to_rlds.py
│   ├── check_rlds.py             counts, value ranges, sample-frame grid of a converted dataset
│   ├── stream_producer.py        rolling window of shards for one long run (download, convert, rotate)
│   └── compute_stream_stats.py   fixed normalization statistics for a stream
├── eval/                    MolmoSpaces benchmark evaluation
│   ├── oft_policy_server.py      `oft` env: serves a run dir over HTTP (LoRA merged in GPU memory)
│   ├── molmospaces_oft_policy.py `mlspaces` env: MolmoSpaces policy + eval config that queries the server
│   └── inspect_benchmarks.py     summary of every installed benchmark
└── scripts/                 server entry points (all source common.sh)
    ├── common.sh                 paths, mirrors, helpers  <- the single source of truth for the server layout
    ├── run_detached.sh           start any step detached from SSH, log to $WS/logs
    ├── 01_download_inspect.sh    `mlspaces` env + 1 raw shard + format report
    ├── 02_install_benchmark.sh   benchmarks, EGL, benchmark report, no-op policy episode
    ├── 03_convert_dev.sh         `oft` env (TF part) + RLDS dataset molmobot_pick_dev
    ├── 04_install_oft_smoke_train.sh  OpenVLA-OFT install, OpenVLA-7B download, 200-step training
    ├── 05_eval.sh                evaluate a run dir on a benchmark
    ├── 06_stream_train.sh        long streaming run (SMOKE=1 for a quick test)
    ├── stream_status.sh          read-only status of a streaming run
    ├── migrate_server_layout.sh  one-time move of the old server workspace to this layout
    ├── oft_constraints.txt       pip pins for the `oft` env
    └── sync_to_server.sh         LAPTOP: retrying rsync of the repo to the server
```

Changes outside this folder (OpenVLA-OFT itself): `prismatic/vla/constants.py` (MOLMOSPACES constants),
`prismatic/vla/datasets/rlds/oxe/{configs,transforms,materialize}.py` (dataset registration, absolute joint
actions), `prismatic/vla/datasets/rlds/{dataset,streaming}.py` (streaming source, no GCS lookups),
`vla-scripts/finetune.py` (step heartbeat for the stream producer).

## Server workspace (`$WS` = `~/HPL_workspace/molmospaces_ws` -> `/ssd/prime/molmospaces_ws`)

```
models/openvla-7b/        data/raw/  data/rlds/  data/streams/{pick,pick_smoke}/
runs/                     eval/<run>/<benchmark>_<N>ep/
assets/ mlspaces_cache/   (MolmoSpaces simulator; managed by molmospaces)
cache/{hf_home,wandb}/    setup/{bulk_download.py,oft_deps/}    logs/
```

## Data and conventions

| | |
|---|---|
| Images | 1 third-person camera (one of 4, random per episode) + `wrist_camera_zed_mini`, letterboxed to 224x224 |
| State (8) | 7 arm joint angles [rad] + gripper opening in [0, 1] |
| Action (8) | 7 absolute target joint angles [rad] + gripper {0 open, 1 close}; chunks of 8 at 15 Hz |
| Normalization | joints -> [-1, 1] by 1st/99th percentile; gripper left as is |

## Running (on the server)

```bash
S=~/HPL_workspace/openvla-oft/experiments/robot/molmospaces/scripts
bash $S/run_detached.sh 05_eval.sh ~/HPL_workspace/molmospaces_ws/runs/<run dir># 5 Pick-Classic episodes
SMOKE=1 bash $S/run_detached.sh 06_stream_train.sh                          # streaming test
bash $S/run_detached.sh 06_stream_train.sh                                  # full run
bash $S/stream_status.sh pick_smoke                                         # progress
```

Network notes: Hugging Face via `hf-mirror.com`, PyPI via the Tsinghua mirror, GitHub-only packages copied from
the laptop into `setup/oft_deps/`, wandb offline. All set in `common.sh`.
