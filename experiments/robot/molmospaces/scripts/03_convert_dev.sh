#!/usr/bin/env bash
# Step 3: create the OpenVLA-OFT env (`oft`, TF/TFDS part only), convert the raw shard from step 1
# (part0 train + val) to the RLDS dataset `molmobot_pick_dev`, and sanity-check it.
#   bash scripts/run_detached.sh 03_convert_dev.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DATASET_NAME="${DATASET_NAME:-molmobot_pick_dev}"
RAW="$RAW_DIR/FrankaPickOmniCamConfig/part0"
require_free_gb 8

if ! conda env list | grep -qE "^oft\s"; then
  conda create -y -n oft python=3.10
fi
conda activate oft
# tensorflow_metadata/protobuf pinned to the TF 2.15 era; newer metadata needs protobuf>=5 (breaks TF 2.15)
pip install "tensorflow==2.15.0" "tensorflow_datasets==4.9.3" "tensorflow_metadata==1.14.0" "protobuf==3.20.3" \
  "numpy<2" h5py av "imageio[pyav]" pillow

# Smoke test: 8 episodes into a throwaway dir, fails fast on reader/TFDS errors
SMOKE_DIR="$RLDS_DIR/_smoke"
rm -rf "$SMOKE_DIR"
python "$MS_DIR/data/convert_molmobot_to_rlds.py" --name molmobot_pick_smoke --train_dirs "$RAW/train" \
  --val_dirs "$RAW/val" --out_dir "$SMOKE_DIR" --max_episodes 8 --num_workers 4
python "$MS_DIR/data/check_rlds.py" --name molmobot_pick_smoke --data_dir "$SMOKE_DIR" --grid_episodes 2
rm -rf "$SMOKE_DIR"

time python "$MS_DIR/data/convert_molmobot_to_rlds.py" --name "$DATASET_NAME" \
  --train_dirs "$RAW/train" --val_dirs "$RAW/val" --out_dir "$RLDS_DIR"
python "$MS_DIR/data/check_rlds.py" --name "$DATASET_NAME" --data_dir "$RLDS_DIR" \
  --out_png "$LOG_DIR/${DATASET_NAME}_samples.png" | tee "$LOG_DIR/check_${DATASET_NAME}.txt"
echo "DONE"
