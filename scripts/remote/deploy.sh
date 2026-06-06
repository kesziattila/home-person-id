#!/usr/bin/env bash
# Deploy the local working tree to the Jetson — CODE ONLY.
# Never touches remote data/, venv/, models, config/config.yaml or other runtime state.
#
# Usage: scripts/remote/deploy.sh [--restart] [--dry-run]
#   --restart   restart the server on the Jetson after deploying (scripts/remote/restart.sh)
#   --dry-run   show what rsync would transfer/delete without doing it
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

RESTART=0
DRY=()
for arg in "$@"; do
    case "$arg" in
        --restart) RESTART=1 ;;
        --dry-run) DRY=(--dry-run -v) ;;
        *) echo "usage: deploy.sh [--restart] [--dry-run]" >&2; exit 2 ;;
    esac
done

# Only these top-level dirs/files are synced; --delete applies INSIDE them only,
# so remote-only state at the project root (data/, venv/, models/, trash dirs,
# config/config.yaml via the exclude below) is never touched.
SOURCES=(src tests tools docs config scripts
         requirements.txt requirements-jetson.txt constraints-jetson.txt
         README.md ARCHITECTURE.md CLAUDE.md
         Dockerfile docker-compose.yaml mosquitto.conf)

cd "$PROJECT_ROOT"
rsync -azR --delete "${DRY[@]}" \
    -e "ssh ${SSH_OPTS[*]}" \
    --exclude='__pycache__/' --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    --exclude='config/config.yaml' \
    --exclude='scripts/remote/remote.env' \
    --exclude='*.pt' --exclude='*.pth' --exclude='*.onnx' --exclude='*.engine' --exclude='*.trt' \
    --exclude='*.log' --exclude='*.db' --exclude='*.db-wal' --exclude='*.db-shm' \
    "${SOURCES[@]}" "$JETSON_HOST:$REMOTE_PROJECT_DIR/"

if [[ ${#DRY[@]} -gt 0 ]]; then
    echo "(dry run — nothing deployed)"
    exit 0
fi
echo "Deployed to $JETSON_HOST:$REMOTE_PROJECT_DIR"

if (( RESTART )); then
    "$_REMOTE_DIR/restart.sh" restart
fi
