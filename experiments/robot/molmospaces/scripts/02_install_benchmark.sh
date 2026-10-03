#!/usr/bin/env bash
# Step 2: install the MolmoSpaces benchmarks (objects are fetched lazily per house), set up headless EGL
# rendering, summarize every benchmark, and run one FrankaPickHardBench episode with the built-in no-op policy.
#   bash scripts/run_detached.sh 02_install_benchmark.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SMOKE_BENCH=benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
require_free_gb 15
conda activate mlspaces
mkdir -p "$MLSPACES_ASSETS_DIR" "$MLSPACES_CACHE_DIR"

if [[ ! -e "$MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2" ]]; then
  time python -m molmo_spaces.molmo_spaces_constants
fi

# EGL loader for headless MuJoCo (the system has only libEGL_nvidia, no libEGL.so.1)
if [[ ! -e "$CONDA_PREFIX/lib/libEGL.so.1" ]]; then
  conda install -y -n mlspaces -c conda-forge libegl libgl libglvnd
fi
python -c "import mujoco; ctx = mujoco.GLContext(64, 64); ctx.make_current(); print('EGL rendering OK')"

python "$MS_DIR/eval/inspect_benchmarks.py" "$MLSPACES_ASSETS_DIR/benchmarks" | tee "$LOG_DIR/inspect_benchmarks.txt"

time python -m molmo_spaces.evaluation.eval_main \
  molmo_spaces.evaluation.configs.evaluation_configs:DummyBenchmarkEvalConfig \
  --benchmark_dir "$MLSPACES_ASSETS_DIR/$SMOKE_BENCH" --max_episodes 1 --no_wandb \
  --output_dir "$EVAL_DIR/dummy_policy_smoke"
echo "DONE"
