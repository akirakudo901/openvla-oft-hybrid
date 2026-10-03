#!/usr/bin/env bash
# Step 6: long OpenVLA-OFT fine-tuning run on a rolling window of MolmoBot-Data shards.
#   1. starts data/stream_producer.py (download -> convert -> activate/retire shards), detached from this shell
#   2. waits until the window holds WINDOW_SIZE live shards, computes normalization statistics once
#   3. runs finetune.py on dataset `molmobot_pick_stream` on the GPUs that are idle at that moment
# The learning rate drops 10x at DECAY_STEPS (default 2/3 of MAX_STEPS, as in OFT's LIBERO recipe 100k/150k).
#
#   SMOKE=1 bash scripts/run_detached.sh 06_stream_train.sh     # test: 2 shards rotating every 100 steps, 400 steps
#   bash scripts/run_detached.sh 06_stream_train.sh             # full run: 6 shards, 100k steps
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ "${SMOKE:-0}" == 1 ]]; then
  STREAM_DIR="${STREAM_DIR:-$STREAMS_DIR/pick_smoke}"
  WINDOW_SIZE=2; MAX_STEPS=400; SAVE_FREQ=400; SHUFFLE_BUFFER=20000; RUN_NOTE=stream-smoke
  PRODUCER_EXTRA=(--residency_steps 100 --retire_grace_min 2 --max_shards 4)
else
  STREAM_DIR="${STREAM_DIR:-$STREAMS_DIR/pick}"
  WINDOW_SIZE="${WINDOW_SIZE:-6}"; MAX_STEPS="${MAX_STEPS:-100000}"; SAVE_FREQ="${SAVE_FREQ:-10000}"
  SHUFFLE_BUFFER=100000; RUN_NOTE="${RUN_NOTE:-stream}"
  PRODUCER_EXTRA=()
fi
export MOLMOBOT_STREAM_DIR="$STREAM_DIR"
mkdir -p "$STREAM_DIR"
conda activate oft
python -c "import zstandard, pyarrow" 2>/dev/null || pip install -c "$MS_DIR/scripts/oft_constraints.txt" zstandard pyarrow

# ---- 1. producer (keeps running if this script or the SSH session dies) -------------------------------------
PRODUCER_PID_FILE="$STREAM_DIR/producer.pid"
if [[ -f "$PRODUCER_PID_FILE" ]] && kill -0 "$(cat "$PRODUCER_PID_FILE")" 2>/dev/null; then
  echo "producer already running (pid $(cat "$PRODUCER_PID_FILE"))"
else
  setsid nohup python "$MS_DIR/data/stream_producer.py" --window_dir "$STREAM_DIR" --window_size "$WINDOW_SIZE" \
    "${PRODUCER_EXTRA[@]}" >> "$STREAM_DIR/producer.log" 2>&1 < /dev/null &
  echo $! > "$PRODUCER_PID_FILE"
  echo "started producer (pid $!), log: $STREAM_DIR/producer.log"
fi

# ---- 2. wait for a full window, then fix the normalization statistics ---------------------------------------
live_count() { python -c "from prismatic.vla.datasets.rlds.streaming import list_live_datasets as l; print(len(l('$STREAM_DIR')))" 2>/dev/null; }
until [[ "$(live_count)" -ge "$WINDOW_SIZE" ]]; do
  kill -0 "$(cat "$PRODUCER_PID_FILE")" 2>/dev/null \
    || { echo "ERROR: producer died; see $STREAM_DIR/producer.log" >&2; tail -20 "$STREAM_DIR/producer.log" >&2; exit 1; }
  echo "$(date +%H:%M) waiting for window: $(live_count)/$WINDOW_SIZE live shards"
  sleep 120
done
if [[ ! -f "$STREAM_DIR/dataset_statistics.json" ]]; then
  python "$MS_DIR/data/compute_stream_stats.py" --window_dir "$STREAM_DIR" --num_shards "$WINDOW_SIZE"
fi

# ---- 3. train on the GPUs that are idle now ------------------------------------------------------------------
GPUS=$(idle_gpus "${MAX_GPUS:-8}")
[[ -n "$GPUS" ]] || { echo "ERROR: no idle GPU found" >&2; exit 1; }
NUM_GPUS=$(awk -F, '{print NF}' <<< "$GPUS")
echo "training on GPUs $GPUS ($NUM_GPUS x batch 8 = $((8 * NUM_GPUS)) frames/step)"

cd "$OFT_DIR"
rm -f "$STREAM_DIR/trainer_step.txt"  # a new run counts from step 0; the producer rebases on the first report
CUDA_VISIBLE_DEVICES="$GPUS" torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" vla-scripts/finetune.py \
  --vla_path "$MODEL_DIR" \
  --data_root_dir "$STREAM_DIR" \
  --dataset_name molmobot_pick_stream \
  --run_root_dir "$RUNS_DIR" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay "${DECAY_STEPS:-$((MAX_STEPS * 2 / 3))}" \
  --max_steps "$MAX_STEPS" \
  --save_freq "$SAVE_FREQ" \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training False \
  --shuffle_buffer_size "$SHUFFLE_BUFFER" \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity none \
  --wandb_project molmospaces \
  --run_id_note "${RUN_NOTE}--${NUM_GPUS}gpu"

kill "$(cat "$PRODUCER_PID_FILE")" 2>/dev/null || true
echo "DONE"
