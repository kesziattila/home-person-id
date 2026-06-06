#!/usr/bin/env bash
# One-shot health check of the live Jetson system (read-only).
# Reports: process state, sqlite3/python3 availability, DB + WAL + log sizes,
# disk usage, and whether a recent performance report exists in the log.
#
# Usage: scripts/remote/rstatus.sh
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

ssh_jetson "DIR='$REMOTE_PROJECT_DIR' DB='$REMOTE_DB' LOG='$REMOTE_LOG' OUTLOG='$REMOTE_STDOUT_LOG' bash -s" <<'REMOTE'
set -uo pipefail
cd "$DIR" 2>/dev/null || { echo "ERROR: remote project dir missing: $DIR"; exit 1; }

echo "== host =="
echo "$(hostname)  $(uptime -p 2>/dev/null || uptime)"

echo "== process =="
if [[ -f data/app.pid ]] && kill -0 "$(cat data/app.pid)" 2>/dev/null; then
    echo "RUNNING (pid $(cat data/app.pid), via pidfile)"
elif pgrep -af 'python.*src\.main' >/dev/null 2>&1; then
    echo "RUNNING (no/stale pidfile):"
    pgrep -af 'python.*src\.main'
else
    echo "NOT RUNNING"
fi

echo "== capabilities =="
if command -v sqlite3 >/dev/null 2>&1; then
    echo "sqlite3: $(command -v sqlite3)"
else
    echo "sqlite3: MISSING (rquery falls back to python3)"
fi
echo "python3: $(python3 --version 2>&1 || echo MISSING)"

echo "== database =="
if [[ -f "$DB" ]]; then
    for f in "$DB" "$DB-wal" "$DB-shm"; do
        [[ -f "$f" ]] && echo "$(basename "$f"): $(du -h "$f" | cut -f1)  (mtime $(date -r "$f" '+%Y-%m-%d %H:%M:%S'))"
    done
else
    echo "DB missing: $DIR/$DB"
fi

echo "== log =="
if [[ -f "$LOG" ]]; then
    echo "size: $(du -h "$LOG" | cut -f1)  (mtime $(date -r "$LOG" '+%Y-%m-%d %H:%M:%S'))"
    echo "last line: $(tail -n 1 "$LOG" | cut -c1-160)"
else
    echo "log missing: $DIR/$LOG"
fi
if [[ -f "$OUTLOG" ]]; then
    # grep -c prints the count itself (0 included); ignore its no-match exit code
    echo "stdout log: $(du -h "$OUTLOG" | cut -f1), perf reports: $(grep -ac -- '--- Performance Report' "$OUTLOG" || true)"
else
    echo "stdout log missing: $DIR/$OUTLOG (app probably started manually, not via restart.sh)"
fi

echo "== disk =="
df -h "$DIR" | tail -n 1 | awk '{print $4" free of "$2" ("$5" used) on "$6}'
# du may exit non-zero on unreadable subdirs while still printing the total
echo "data/ size: $({ du -sh data 2>/dev/null || true; } | cut -f1)"
REMOTE
