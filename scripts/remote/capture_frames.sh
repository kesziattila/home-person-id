#!/usr/bin/env bash
# Time-limited full-frame capture on the Jetson for offline detector tuning
# (src/tracking/detection_log.FrameCapture). The running app picks up the trigger
# within a second; no restart, no config change. Size cap: metrics.frame_capture_max_gb.
#
# Usage: scripts/remote/capture_frames.sh start <minutes>   capture every logged frame for N minutes
#        scripts/remote/capture_frames.sh stop
#        scripts/remote/capture_frames.sh status            trigger, frames per day, disk use
#        scripts/remote/capture_frames.sh fetch [YYYY-MM-DD] [HHMM-HHMM]
#                                       frames (+ detection log) of a UTC day, optionally only a
#                                       UTC time window — frames are ~0.7 MB each, mind local disk
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
        WINDOW="${3:-}"
        DEST="$PROJECT_ROOT/$LOCAL_PULL_DIR/metrics/detections"
        mkdir -p "$DEST/frames/$DAY"
        filter=()
        if [[ -n "$WINDOW" ]]; then
            [[ "$WINDOW" =~ ^([0-9]{4})-([0-9]{4})$ ]] || { echo "window must be HHMM-HHMM (UTC)" >&2; exit 2; }
            list=$(ssh_jetson "cd '$DET/frames/$DAY' && find . -name '*.jpg'" \
                | awk -v a="${BASH_REMATCH[1]}" -v b="${BASH_REMATCH[2]}" -F/ \
                    '{ t = substr($NF, 1, 4) } t >= a && t < b { print substr($0, 3) }')
            [[ -n "$list" ]] || { echo "no frames in $WINDOW UTC"; exit 0; }
            printf '%s\n' "$list" > "$DEST/.fetch_list"
            filter=(--files-from="$DEST/.fetch_list")
        fi
        if [[ -n "$WINDOW" ]]; then
            need_kb=$(ssh_jetson "cd '$DET/frames/$DAY' && xargs -d '\\n' du -ck | tail -1 | cut -f1" < "$DEST/.fetch_list")
        else
            need_kb=$(ssh_jetson "du -sck '$DET/frames/$DAY' '$DET/event_frames/$DAY' 2>/dev/null | tail -1 | cut -f1")
        fi
        free_kb=$(df -Pk "$DEST" | awk 'NR == 2 { print $4 }')
        if (( need_kb + 1048576 > free_kb )); then
            echo "Not enough local disk: need ~$((need_kb / 1024)) MB + 1 GB headroom, $((free_kb / 1024)) MB free" >&2; exit 1
        fi
        rsync -az -e "ssh ${SSH_OPTS[*]}" "$JETSON_HOST:$DET/$DAY.jsonl" "$DEST/"
        if ssh_jetson "test -d '$DET/frames/$DAY'"; then
            rsync -az -e "ssh ${SSH_OPTS[*]}" "${filter[@]}" "$JETSON_HOST:$DET/frames/$DAY/" "$DEST/frames/$DAY/"
        fi
        echo "Pulled frames $DAY -> $DEST/frames/$DAY ($(find "$DEST/frames/$DAY" -name '*.jpg' | wc -l) frames)"
        # event-triggered half-size frames (src/tracking/event_capture.py), all of the day
        if ssh_jetson "test -d '$DET/event_frames/$DAY'"; then
            mkdir -p "$DEST/event_frames/$DAY"
            rsync -az -e "ssh ${SSH_OPTS[*]}" "$JETSON_HOST:$DET/event_frames/$DAY/" "$DEST/event_frames/$DAY/"
            echo "Pulled event frames $DAY -> $DEST/event_frames/$DAY ($(find "$DEST/event_frames/$DAY" -name '*.jpg' | wc -l) frames)"
        fi
        ;;
    *)
        sed -n '2,11p' "$0"; exit 2 ;;
esac
