#!/usr/bin/env bash
# Starts one of the step scripts in the background, detached from the SSH session (survives disconnects),
# logging to $WS/logs/<script>_<timestamp>.log. Environment variables are passed through.
#   bash ~/HPL_workspace/openvla-oft/experiments/robot/molmospaces/scripts/run_detached.sh 06_stream_train.sh [args]
#   SMOKE=1 bash .../run_detached.sh 06_stream_train.sh
set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPTS_DIR/common.sh"

SCRIPT="${1:?usage: run_detached.sh <script.sh> [args]}"
shift
LOG="$LOG_DIR/${SCRIPT%.sh}_$(date +%Y%m%d-%H%M%S).log"
setsid nohup bash "$SCRIPTS_DIR/$SCRIPT" "$@" > "$LOG" 2>&1 < /dev/null &
echo "started $SCRIPT (pid $!)"
echo "log: $LOG"
echo "follow: tail -f $LOG"
