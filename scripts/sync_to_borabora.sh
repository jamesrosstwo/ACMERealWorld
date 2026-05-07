#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

LOCAL_NAME="${1:-episodes}"
REMOTE_PATH="${2:-}"
if [[ -z "$REMOTE_PATH" ]]; then
  echo "Usage: $0 [local_subdirectory] <remote_subdirectory>" >&2
  exit 2
fi

LOCAL_BASE="/mnt/ssd/james"
LOCAL_SRC="$LOCAL_BASE/$LOCAL_NAME"
if [[ ! -d "$LOCAL_SRC" ]]; then
  echo "ERROR: local source $LOCAL_SRC does not exist" >&2
  exit 1
fi

REMOTE_BASE="/home/james/ACME/data"
REMOTE_DEST="$REMOTE_BASE/$REMOTE_PATH"
REMOTE_HOST="james@142.1.46.125"
SSH_CMD="ssh -p 2233 -i ~/.ssh/id_ed25519 -J james@142.1.44.183:2233"

human() { numfmt --to=iec --suffix=B "$1"; }

# Episodes that already have COMPLETE on borabora are excluded from transfer.
EXCLUDES_RAW=$($SSH_CMD "$REMOTE_HOST" \
  "mkdir -p $REMOTE_DEST && for d in $REMOTE_DEST/*/; do [ -f \"\$d/COMPLETE\" ] && basename \"\$d\"; done" \
  2>/dev/null || true)

RSYNC_EXCLUDES=()
declare -A EXCLUDED
while IFS= read -r d; do
  [[ -z "$d" ]] && continue
  RSYNC_EXCLUDES+=("--exclude=$d")
  EXCLUDED["$d"]=1
done <<<"$EXCLUDES_RAW"

NEEDED_BYTES=0
for entry in "$LOCAL_SRC"/*; do
  name=$(basename "$entry")
  [[ -n "${EXCLUDED[$name]:-}" ]] && continue
  size=$(du -sb "$entry" | awk '{print $1}')
  NEEDED_BYTES=$((NEEDED_BYTES + size))
done

AVAIL_BYTES=$($SSH_CMD "$REMOTE_HOST" \
  "df -B1 --output=avail $REMOTE_DEST | tail -n1 | tr -d ' '")

echo "Transfer size: $(human "$NEEDED_BYTES")"
echo "Borabora free: $(human "$AVAIL_BYTES") at $REMOTE_DEST"

if (( AVAIL_BYTES < NEEDED_BYTES )); then
  echo "ERROR: not enough free space on borabora ($(human "$AVAIL_BYTES") available, $(human "$NEEDED_BYTES") needed)" >&2
  exit 1
fi

LOG="/home/james/sync_${LOCAL_NAME}_to_${REMOTE_PATH}_$(date +%Y%m%d_%H%M%S).log"
setsid -f rsync -av --partial --progress "${RSYNC_EXCLUDES[@]}" \
  -e "$SSH_CMD" \
  "$LOCAL_SRC/" "$REMOTE_HOST:$REMOTE_DEST" \
  >"$LOG" 2>&1 </dev/null
echo "tail -f $LOG"
