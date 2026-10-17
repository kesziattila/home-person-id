#!/usr/bin/env bash
# Time-limited full-frame capture on the Jetson for offline detector tuning
# (src/tracking/detection_log.FrameCapture). The running app picks up the trigger
# within a second; no restart, no config change. Size cap: metrics.frame_capture_max_gb.
#
# Usage: scripts/remote/capture_frames.sh start <minutes>   capture every logged frame for N minutes
#        scripts/remote/capture_frames.sh stop
#        scripts/remote/capture_frames.sh status            trigger, frames per day, disk use
#        scripts/remote/capture_frames.sh fetch [YYYY-MM-DD] frames + detection log of a UTC day
#   then: PYTHONPATH=. python tools/redetect.py local/remote/metrics --config ... --out ...
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

: "${REMOTE_METRICS_DIR:=data/metrics}"
DET="$REMOTE_PROJECT_DIR/$REMOTE_METRICS_DIR/detections"

case "${1:-status}" in
    start)
        minutes="${2:?usage: capture_frames.sh start <minutes>}"
        [[ "$minutes" =~ ^[0-9]+$ ]] || { echo "minutes must be an integer" >&2; exit 2; }
        ssh_jetson "mkdir -p '$DET' && echo \$(( \$(date +%s) + $minutes * 60 )) > '$DET/capture_until'"
        echo "Capturing for $minutes min (until $(date -d "+$minutes min" +%H:%M) local)"
        ;;
    stop)
        ssh_jetson "rm -f '$DET/capture_until'"
        echo "Capture stopped"
        ;;
    status)
        ssh_jetson "DET='$DET' bash -s" <<'REMOTE'
if [[ -f "$DET/capture_until" ]] && (( $(cat "$DET/capture_until") > $(date +%s) )); then
    echo "capturing until $(date -d @"$(cat "$DET/capture_until")" +%H:%M:%S) local"
else
    echo "not capturing"
fi
for day in "$DET"/frames/*/; do
    [[ -d "$day" ]] && echo "  $(basename "$day"): $(find "$day" -name '*.jpg' | wc -l) frames, $(du -sh "$day" | cut -f1)"
done
df -h "$DET" | tail -1 | awk '{print "  disk: " $4 " free of " $2}'
REMOTE
        ;;
    fetch)
        DAY="${2:-$(date -u +%F)}"
        DEST="$PROJECT_ROOT/$LOCAL_PULL_DIR/metrics/detections"
        mkdir -p "$DEST/frames/$DAY"
        rsync -az -e "ssh ${SSH_OPTS[*]}" "$JETSON_HOST:$DET/$DAY.jsonl" "$DEST/"
        rsync -az -e "ssh ${SSH_OPTS[*]}" "$JETSON_HOST:$DET/frames/$DAY/" "$DEST/frames/$DAY/"
        echo "Pulled frames $DAY -> $DEST/frames/$DAY ($(find "$DEST/frames/$DAY" -name '*.jpg' | wc -l) frames)"
        ;;
    *)
        sed -n '2,11p' "$0"; exit 2 ;;
esac
