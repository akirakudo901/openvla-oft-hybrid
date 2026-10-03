#!/usr/bin/env bash
# Step 4: install OpenVLA-OFT into the `oft` env, download OpenVLA-7B, and run a short fine-tuning smoke test on
# `molmobot_pick_dev` (checks the pipeline end to end and measures seconds/step).
#
# The server cannot reach GitHub: first copy these from the laptop into $WS/setup/oft_deps/
#   transformers-openvla-oft/   (git clone https://github.com/moojink/transformers-openvla-oft)
#   dlimp_openvla/              (git clone https://github.com/moojink/dlimp_openvla)
#   flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl  (flash-attention v2.5.5 release)
#
#   bash scripts/run_detached.sh 04_install_oft_smoke_train.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DATASET_NAME="${DATASET_NAME:-molmobot_pick_dev}"
SMOKE_STEPS="${SMOKE_STEPS:-200}"
DEPS="$SETUP_DIR/oft_deps"
CONSTRAINTS="$MS_DIR/scripts/oft_constraints.txt"
conda activate oft

if ! python -c "import prismatic, flash_attn, dlimp, transformers" 2>/dev/null; then
  pip install -c "$CONSTRAINTS" torch torchvision torchaudio
  pip install -c "$CONSTRAINTS" "$DEPS/transformers-openvla-oft"
  pip install --no-deps "$DEPS/dlimp_openvla"
  # openvla-oft's own git dependencies are satisfied above, so install it without resolving deps
  pip install --no-deps -e "$OFT_DIR"
  pip install -c "$CONSTRAINTS" accelerate "draccus==0.8.0" einops huggingface_hub json-numpy jsonlines matplotlib \
    "peft==0.11.1" rich "sentencepiece==0.1.99" "timm==0.9.10" tokenizers wandb "tensorflow_graphics==2021.12.3" \
    "diffusers==0.30.3" imageio uvicorn fastapi packaging ninja zstandard pyarrow
  pip install --no-deps "$DEPS"/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
fi
python -c "import torch, transformers, flash_attn, dlimp; print('torch', torch.__version__, 'gpus', torch.cuda.device_count())"

if [[ ! -f "$MODEL_DIR/config.json" ]] || ls "$MODEL_DIR"/*.incomplete >/dev/null 2>&1; then
  require_free_gb 20
  huggingface-cli download openvla/openvla-7b --local-dir "$MODEL_DIR"
fi

GPUS=$(idle_gpus "${MAX_GPUS:-8}")
[[ -n "$GPUS" ]] || { echo "ERROR: no idle GPU found" >&2; exit 1; }
NUM_GPUS=$(awk -F, '{print NF}' <<< "$GPUS")
echo "Using GPUs: $GPUS ($NUM_GPUS)"

cd "$OFT_DIR"
CUDA_VISIBLE_DEVICES="$GPUS" torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" vla-scripts/finetune.py \
  --vla_path "$MODEL_DIR" \
  --data_root_dir "$RLDS_DIR" \
  --dataset_name "$DATASET_NAME" \
  --run_root_dir "$RUNS_DIR" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps "$SMOKE_STEPS" \
  --save_freq "$SMOKE_STEPS" \
  --save_latest_checkpoint_only True \
  --merge_lora_during_training False \
  --shuffle_buffer_size 20000 \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity none \
  --wandb_project molmospaces \
  --run_id_note "smoke--${NUM_GPUS}gpu"
echo "DONE"
