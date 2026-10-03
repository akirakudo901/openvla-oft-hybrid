#!/usr/bin/env bash
# LAPTOP -> GPU server code sync that survives a flaky connection: keepalives, short connect timeout, resumable
# transfers, retries with backoff. Only code moves this way; datasets are downloaded on the server.
# The molmospaces folder is mirrored exactly (files moved/deleted on the laptop are removed on the server);
# the rest of the repo is only added to, so the server's pip-install artifacts (*.egg-info) stay intact.
#   bash experiments/robot/molmospaces/scripts/sync_to_server.sh
set -uo pipefail

KEY="${KEY:-$HOME/Downloads/prime.key}"
HOST="${HOST:-prime@36.103.234.242}"
PORT="${PORT:-2337}"
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
DEST="${DEST:-HPL_workspace/openvla-oft}"
SSH="ssh -i $KEY -p $PORT -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
MS=experiments/robot/molmospaces

retry() {
  for attempt in $(seq 1 10); do
    "$@" && return 0
    echo "attempt $attempt failed; retrying in $((attempt * 20))s" >&2
    sleep $((attempt * 20))
  done
  echo "giving up after 10 attempts: $*" >&2
  return 1
}

retry rsync -az --partial --timeout=120 --exclude .git --exclude __pycache__ --exclude "*.code-workspace" \
  -e "$SSH" "$SRC/" "$HOST:$DEST/" || exit 1
retry rsync -az --partial --timeout=120 --delete --exclude __pycache__ --exclude "*.code-workspace" \
  -e "$SSH" "$SRC/$MS/" "$HOST:$DEST/$MS/" || exit 1
echo "synced $SRC -> $HOST:$DEST"
