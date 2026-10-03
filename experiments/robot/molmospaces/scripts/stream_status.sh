#!/usr/bin/env bash
# Read-only snapshot of a step-6 streaming run: trainer step, window contents, producer progress, disk.
#   bash scripts/stream_status.sh            # full run ($WS/data/streams/pick)
#   bash scripts/stream_status.sh pick_smoke # smoke test
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
STREAM_DIR="$STREAMS_DIR/${1:-pick}"

echo "== $STREAM_DIR"
echo "== trainer (step, global batch): $(cat "$STREAM_DIR/trainer_step.txt" 2>/dev/null || echo 'not started')"
pid=$(cat "$STREAM_DIR/producer.pid" 2>/dev/null)
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then echo "== producer running (pid $pid)"; else echo "== producer NOT running"; fi
echo "== active/:"
for d in "$STREAM_DIR"/active/*/; do
  [[ -d "$d" ]] || continue
  [[ -f "$d/RETIRED" ]] && state=retired || state=live
  echo "   $(basename "$d")  $state  $(du -sh "$d" | cut -f1)"
done
echo "== ready/: $(ls "$STREAM_DIR/ready" 2>/dev/null | tr '\n' ' ')"
python3 - "$STREAM_DIR/manifest.json" <<'EOF' 2>/dev/null
import json, sys
m = json.load(open(sys.argv[1]))
print(f"== shards prepared: {m['next_index']}/{len(m['order'])}, live: {len(m['live'])}, retired: {len(m['retired'])}")
EOF
echo "== disk: $(df -h "$STREAM_DIR" | tail -1 | awk '{print $4 " free of " $2}')"
echo "== producer log:"
tail -5 "$STREAM_DIR/producer.log" 2>/dev/null
