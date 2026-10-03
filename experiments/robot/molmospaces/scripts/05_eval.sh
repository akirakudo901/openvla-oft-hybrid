#!/usr/bin/env bash
# Step 5: evaluate an OpenVLA-OFT fine-tuning run on a MolmoSpaces benchmark. Serves the run with
# eval/oft_policy_server.py (`oft` env, one idle GPU) and runs MolmoSpaces' eval_main against it (`mlspaces` env).
#
#   bash scripts/run_detached.sh 05_eval.sh <RUN_DIR>                       # 5 Pick-Classic episodes
#   NUM_EPISODES=200 BENCH=pick_classic_200 bash scripts/run_detached.sh 05_eval.sh <RUN_DIR>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN_DIR="${1:?usage: 05_eval.sh <RUN_DIR> (a fine-tuning run dir under $RUNS_DIR)}"
RUN_DIR="${RUN_DIR%/}"
NUM_EPISODES="${NUM_EPISODES:-5}"
BENCH="${BENCH:-pick_classic}"
PORT="${PORT:-8777}"
# Run ids look like openvla-7b+<dataset_name>+b8+...; the dataset name is the action un-normalization key
UNNORM_KEY="${UNNORM_KEY:-$(basename "$RUN_DIR" | cut -d+ -f2)}"

V2="$MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-objaverse"
case "$BENCH" in
  pick_classic)     BENCH_DIR="$V2/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark" ;;     # 1000 episodes
  pick_classic_200) BENCH_DIR="$V2/FrankaPickHardBench/FrankaPickHardBench_20260212_200ep_json_benchmark" ;;
  *) echo "ERROR: unknown BENCH=$BENCH" >&2; exit 1 ;;
esac
OUT_DIR="$EVAL_DIR/$(basename "$RUN_DIR")/${BENCH}_${NUM_EPISODES}ep"
disk_report() { echo "[disk] $1: free=$(free_gb "$WS")G objaverse_cache=$(du -sh "$MLSPACES_CACHE_DIR/objects/objaverse" | cut -f1)"; }

# ---- policy server on one idle GPU ----------------------------------------------------------------------------
GPU=$(idle_gpus 1)
[[ -n "$GPU" ]] || { echo "ERROR: no idle GPU found" >&2; exit 1; }
echo "Serving $RUN_DIR (unnorm key $UNNORM_KEY) on GPU $GPU, port $PORT"
conda activate oft
SERVER_LOG="$LOG_DIR/oft_server_${PORT}.log"
CUDA_VISIBLE_DEVICES="$GPU" python "$MS_DIR/eval/oft_policy_server.py" \
  --base_checkpoint "$MODEL_DIR" --pretrained_checkpoint "$RUN_DIR" --unnorm_key "$UNNORM_KEY" --port "$PORT" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  curl -sf "http://localhost:$PORT/health" >/dev/null && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: policy server exited; see $SERVER_LOG" >&2; tail -30 "$SERVER_LOG" >&2; exit 1; }
  sleep 5
done
echo "server ready"

# ---- benchmark episodes ---------------------------------------------------------------------------------------
conda activate mlspaces
python -c "import json_numpy" 2>/dev/null || pip install -q json-numpy
export MUJOCO_EGL_DEVICE_ID="$GPU"
export OFT_SERVER_URL="http://localhost:$PORT"
disk_report "before eval"
time python -m molmo_spaces.evaluation.eval_main \
  experiments.robot.molmospaces.eval.molmospaces_oft_policy:OFTPolicyEvalConfig \
  --benchmark_dir "$BENCH_DIR" --max_episodes "$NUM_EPISODES" --no_wandb --output_dir "$OUT_DIR"
disk_report "after eval"
echo "DONE"
