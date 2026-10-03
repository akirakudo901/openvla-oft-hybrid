# Shared paths and environment for the MolmoSpaces x OpenVLA-OFT server scripts. Source it, don't run it:
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# Server workspace layout ($WS; ~/HPL_workspace/molmospaces_ws is a symlink to /ssd/prime/molmospaces_ws):
#   models/openvla-7b/            base model
#   data/raw/<TaskConfig>/        extracted MolmoBot-Data shards (HDF5 + MP4)
#   data/rlds/<name>/             converted RLDS datasets (e.g. molmobot_pick_dev)
#   data/streams/<name>/          rolling windows for streaming training (managed by stream_producer.py)
#   runs/                         fine-tuning runs and checkpoints
#   eval/                         MolmoSpaces benchmark results
#   assets/, mlspaces_cache/      MolmoSpaces simulator assets (managed by molmospaces; internal links use these paths)
#   cache/                        Hugging Face and wandb caches
#   setup/                        install helpers (pip constraints, bulk_download.py, GitHub-only deps)
#   logs/                         one log per script run, plus inspection reports

OFT_DIR="${OFT_DIR:-$HOME/HPL_workspace/openvla-oft}"
MS_DIR="$OFT_DIR/experiments/robot/molmospaces"
WS="${WS:-$HOME/HPL_workspace/molmospaces_ws}"

MODEL_DIR="${MODEL_DIR:-$WS/models/openvla-7b}"
RAW_DIR="$WS/data/raw"
RLDS_DIR="$WS/data/rlds"
STREAMS_DIR="$WS/data/streams"
RUNS_DIR="$WS/runs"
EVAL_DIR="$WS/eval"
SETUP_DIR="$WS/setup"
LOG_DIR="$WS/logs"
mkdir -p "$RAW_DIR" "$RLDS_DIR" "$STREAMS_DIR" "$RUNS_DIR" "$EVAL_DIR" "$SETUP_DIR" "$LOG_DIR" "$WS/cache"

# MolmoSpaces simulator assets
export MLSPACES_ASSETS_DIR="$WS/assets"
export MLSPACES_CACHE_DIR="$WS/mlspaces_cache"
export MUJOCO_GL=egl

# Blocked or slow from this server -> mirrors / offline
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$WS/cache/hf_home"
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_PROGRESS_BAR=off
export WANDB_MODE=offline
export WANDB_DIR="$WS/cache/wandb"

# `experiments.robot...` modules are imported from the repo root
export PYTHONPATH="$OFT_DIR${PYTHONPATH:+:$PYTHONPATH}"

source ~/miniforge3/etc/profile.d/conda.sh

# GPUs with < 2 GB in use (the machine is shared), comma-separated, at most $1 of them
idle_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 2000 {print $1}' | head -n "${1:-8}" | paste -sd, -
}

# Free space in GB on the disk holding $1
free_gb() {
  df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'
}

require_free_gb() {
  local need=$1 have
  have=$(free_gb "$WS")
  if (( have < need )); then
    echo "ERROR: need at least ${need} GB free in $WS, have ${have} GB." >&2
    exit 1
  fi
}
