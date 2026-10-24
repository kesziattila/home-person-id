#!/usr/bin/env bash
# Hourly health check (skill health-watch): pull what the checker reads, then run it.
#
# Usage: scripts/remote/health_check.sh [extra tools/health_check.py args]
# Pulls metrics (incl. naming/perf/age), today's audit samples and a consistent DB copy,
# then runs tools/health_check.py over the window since its previous run. Prints one
# metrics line + JSON of NEW alerts; exit 1 = new alerts (investigate), 0 = OK / nothing new.
# Writes only local/health/ (gitignored). Read-only on the Jetson.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
cd "$PROJECT_ROOT" || exit 2

"$_REMOTE_DIR/fetch_metrics.sh" >/dev/null || { echo "fetch_metrics failed (VPN / Jetson down?)"; exit 2; }
"$_REMOTE_DIR/fetch_audit.sh" >/dev/null || true
"$_REMOTE_DIR/sync_db.sh" >/dev/null || echo "sync_db failed: DB checks skipped"
[[ -f "$LOCAL_PULL_DIR/config.jetson-live.yaml" ]] || "$_REMOTE_DIR/fetch_config.sh" >/dev/null

source venv/bin/activate
PYTHONPATH=. python tools/health_check.py "$LOCAL_PULL_DIR/metrics" --db "$LOCAL_PULL_DIR/database.db" \
    --config "$LOCAL_PULL_DIR/config.jetson-live.yaml" "$@"
