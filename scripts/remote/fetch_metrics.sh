#!/usr/bin/env bash
# Pull the on-disk metrics (face score log, ...) from the Jetson for offline tuning.
# Lands in local/remote/metrics/ (gitignored). Incremental: only new/changed files.
#
# Usage: scripts/remote/fetch_metrics.sh
#   then: PYTHONPATH=. python tools/analyze_face_scores.py local/remote/metrics/face_scores
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

: "${REMOTE_METRICS_DIR:=data/metrics}"
DEST="$PROJECT_ROOT/$LOCAL_PULL_DIR/metrics"
mkdir -p "$DEST"

# Audit images are large; scripts/remote/fetch_audit.sh pulls them per day
rsync -az -e "ssh ${SSH_OPTS[*]}" --exclude 'audit_frames/' \
    "$JETSON_HOST:$REMOTE_PROJECT_DIR/$REMOTE_METRICS_DIR/" "$DEST/"
echo "Pulled metrics -> $DEST"
find "$DEST" -name '*.jsonl' -printf '  %P  %s bytes\n' | sort
