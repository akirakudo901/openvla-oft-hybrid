#!/usr/bin/env bash
# Step 1: create the MolmoSpaces env (`mlspaces`), download ONE train shard + the val shard of a MolmoBot-Data
# task config, and print its on-disk format. Data lands in $WS/data/raw (never on the laptop).
#   bash scripts/run_detached.sh 01_download_inspect.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TASK_CONFIG="${TASK_CONFIG:-FrankaPickOmniCamConfig}"
require_free_gb 20

if ! conda env list | grep -qE "^mlspaces\s"; then
  conda create -y -n mlspaces python=3.11
fi
conda activate mlspaces
pip install zstandard datasets huggingface_hub tqdm h5py "imageio[ffmpeg]"

if [[ ! -f "$SETUP_DIR/bulk_download.py" ]]; then
  hf download allenai/molmobot-data bulk_download.py --repo-type dataset --local-dir "$SETUP_DIR"
fi
python "$SETUP_DIR/bulk_download.py" "$RAW_DIR" --config "$TASK_CONFIG" --split all --part 0 --max_part_shards 1 -y
du -sh "$RAW_DIR/$TASK_CONFIG"/*

python "$MS_DIR/data/inspect_molmobot.py" "$RAW_DIR/$TASK_CONFIG" | tee "$LOG_DIR/inspect_${TASK_CONFIG}.txt"

# MolmoSpaces itself (torch 2.7; needed from step 2 on)
pip install "molmo-spaces[mujoco]==0.2.9"
python -c "import molmo_spaces, mujoco; print('molmo_spaces OK, mujoco', mujoco.__version__)"
echo "DONE"
