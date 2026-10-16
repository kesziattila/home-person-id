#!/usr/bin/env bash
# Fetch the latest annotated frame of each camera (zone polygons + tracks drawn) from the
# running app's API, for visual zone review. Lands in local/remote/frames/<UTC ts>/.
#
# Usage: scripts/remote/grab_frames.sh [camera_id ...]   (default: all cameras the API reports)
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

: "${REMOTE_API_PORT:=8001}"
API="http://localhost:$REMOTE_API_PORT/api/v1"
DEST="$PROJECT_ROOT/$LOCAL_PULL_DIR/frames/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$DEST"

cams=("$@")
if [[ ${#cams[@]} -eq 0 ]]; then
    mapfile -t cams < <(ssh_jetson "curl -sf $API/cameras" \
        | python3 -c 'import json,sys; print("\n".join(json.load(sys.stdin)["cameras"]))')
fi
for cam in "${cams[@]}"; do
    if ssh_jetson "curl -sf '$API/frame/$cam?zones=true'" > "$DEST/$cam.jpg"; then
        echo "  $DEST/$cam.jpg"
    else
        rm -f "$DEST/$cam.jpg"; echo "  $cam: no frame" >&2
    fi
done
