#!/usr/bin/env bash
# Pull the LIVE config from the Jetson for local reproduction of issues.
# The copy lands in local/remote/ (gitignored) — reference only, never commit it.
#
# Usage: scripts/remote/fetch_config.sh
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

mkdir -p "$PROJECT_ROOT/$LOCAL_PULL_DIR"
DEST="$PROJECT_ROOT/$LOCAL_PULL_DIR/config.jetson-live.yaml"

scp "${SSH_OPTS[@]}" -q "$JETSON_HOST:$REMOTE_PROJECT_DIR/config/config.yaml" "$DEST"
echo "Pulled live config -> $DEST"
