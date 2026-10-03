#!/usr/bin/env bash
# ONE-TIME (server): move the MolmoSpaces workspace to /ssd/prime, leave a symlink at the old path, and sort its
# contents into the layout described in common.sh. Safe to re-run: every move is skipped once done.
#   bash ~/HPL_workspace/openvla-oft/experiments/robot/molmospaces/scripts/migrate_server_layout.sh
set -euo pipefail

OLD_WS="$HOME/HPL_workspace/molmospaces_ws"
NEW_WS="/ssd/prime/molmospaces_ws"

if pgrep -u "$USER" -f "[s]tream_producer|[f]inetune.py|[o]ft_policy_server|[e]val_main|[c]onvert_molmobot" >/dev/null; then
  echo "ERROR: a MolmoSpaces/OFT job is still running; stop it first:" >&2
  pgrep -u "$USER" -af "[s]tream_producer|[f]inetune.py|[o]ft_policy_server|[e]val_main|[c]onvert_molmobot" >&2
  exit 1
fi

# ---- 1. workspace onto /ssd, symlink at the old path ----------------------------------------------------------
if [[ -d "$OLD_WS" && ! -L "$OLD_WS" ]]; then
  echo "copying $OLD_WS -> $NEW_WS (keeps symlinks as they are; original removed only after a complete copy)"
  mkdir -p "$NEW_WS"
  rsync -a "$OLD_WS/" "$NEW_WS/"
  rm -rf "$OLD_WS"
  ln -s "$NEW_WS" "$OLD_WS"
fi
WS="$OLD_WS"
cd "$WS"

# ---- 2. new layout --------------------------------------------------------------------------------------------
move() {  # move <src> <dst>: only if src exists and dst does not
  if [[ -e "$1" && ! -e "$2" ]]; then
    mkdir -p "$(dirname "$2")"
    mv "$1" "$2"
    echo "moved $1 -> $2"
  fi
}
move mbdata data/raw
move rlds data/rlds
move eval_output eval
move hf_home cache/hf_home
move wandb cache/wandb
move bulk_download.py setup/bulk_download.py
move "$HOME/HPL_workspace/oft_deps" setup/oft_deps
rm -f oft_constraints.txt  # now versioned in the repo: scripts/oft_constraints.txt
rm -rf rlds_smoke

mkdir -p logs/old_runs
for f in inspect_*.txt check_*.txt oft_server_*.log data/rlds/*_samples.png; do
  [[ -e "$f" ]] && mv "$f" logs/ && echo "moved $f -> logs/"
done
for f in "$HOME"/HPL_workspace/logs/molmospaces_step*.log; do
  [[ -e "$f" ]] && mv "$f" logs/old_runs/ && echo "moved $f -> logs/old_runs/"
done

# ---- 3. report ------------------------------------------------------------------------------------------------
echo
ls -ld "$OLD_WS"
for d in models data/raw data/rlds data/streams runs eval assets mlspaces_cache cache setup logs; do
  [[ -e "$d" ]] && printf "  %-16s %s\n" "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done
df -h / /ssd | tail -2
