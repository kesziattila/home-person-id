#!/usr/bin/env bash
# Upload an edited live config to the Jetson (the ONLY sanctioned way to change it).
# 1. validates the file locally with src.config.load_config
# 2. backs up the current remote config to data/config-backups/config.yaml.<UTC ts>
# 3. uploads it as config/config.yaml (does not restart — use restart.sh / deploy.sh --restart)
#
# Usage: scripts/remote/push_config.sh [local-file]   (default: local/remote/config.jetson-live.yaml)
#        scripts/remote/push_config.sh --rollback      (restore the newest backup)
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

REMOTE_CFG="$REMOTE_PROJECT_DIR/config/config.yaml"
BACKUP_DIR="$REMOTE_PROJECT_DIR/data/config-backups"

if [[ "${1:-}" == "--rollback" ]]; then
    ssh_jetson "set -e; latest=\$(ls -1t $BACKUP_DIR/config.yaml.* | head -1); cp \"\$latest\" $REMOTE_CFG; echo \"Restored \$latest\""
    exit 0
fi

SRC="${1:-$PROJECT_ROOT/$LOCAL_PULL_DIR/config.jetson-live.yaml}"
[[ -f "$SRC" ]] || { echo "ERROR: $SRC not found" >&2; exit 1; }

(cd "$PROJECT_ROOT" && PYTHONPATH=. "$PROJECT_ROOT/venv/bin/python" -c \
    "import sys; from src.config import load_config; load_config(sys.argv[1])" "$SRC") \
    || { echo "ERROR: $SRC failed validation, not uploaded" >&2; exit 1; }

TS=$(date -u +%Y%m%dT%H%M%SZ)
ssh_jetson "mkdir -p $BACKUP_DIR && cp $REMOTE_CFG $BACKUP_DIR/config.yaml.$TS"
scp "${SSH_OPTS[@]}" -q "$SRC" "$JETSON_HOST:$REMOTE_CFG"
echo "Uploaded $SRC -> $REMOTE_CFG (previous saved as data/config-backups/config.yaml.$TS)"
