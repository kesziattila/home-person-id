#!/usr/bin/env bash
# Pull identity audit samples (annotated frames + track crops + JSONL) of one UTC day
# for spot-check review (skill review-identities). Lands in local/remote/metrics/.
#
# Usage: scripts/remote/fetch_audit.sh [YYYY-MM-DD]   (default: today, UTC)
#   then: PYTHONPATH=. python tools/audit_sheet.py local/remote/metrics --since <ISO ts>
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

: "${REMOTE_METRICS_DIR:=data/metrics}"
DAY="${1:-$(date -u +%F)}"
DEST="$PROJECT_ROOT/$LOCAL_PULL_DIR/metrics"
mkdir -p "$DEST/audit" "$DEST/audit_frames/$DAY"

rsync -az -e "ssh ${SSH_OPTS[*]}" \
    "$JETSON_HOST:$REMOTE_PROJECT_DIR/$REMOTE_METRICS_DIR/audit/$DAY.jsonl" "$DEST/audit/" || exit 1
rsync -az -e "ssh ${SSH_OPTS[*]}" \
    "$JETSON_HOST:$REMOTE_PROJECT_DIR/$REMOTE_METRICS_DIR/audit_frames/$DAY/" "$DEST/audit_frames/$DAY/"
echo "Pulled audit $DAY -> $DEST ($(wc -l < "$DEST/audit/$DAY.jsonl") samples, $(ls "$DEST/audit_frames/$DAY" | wc -l) images)"
