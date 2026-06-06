#!/usr/bin/env bash
# Pull a CONSISTENT snapshot of the live Jetson DB for heavy local analysis.
#
# Usage: scripts/remote/sync_db.sh [--dump]
#   default: remote `sqlite3 .backup` into /tmp (the only remote write; cleaned up
#            afterwards), then scp the single self-consistent file.
#   --dump:  stream `.dump` SQL over ssh into a local sqlite3 — zero remote disk
#            writes, but slower (needs sqlite3 both sides).
#
# NEVER copy the live .db + -wal + -shm files individually — that can produce a
# torn, inconsistent snapshot. `.backup` merges the WAL into one consistent file.
#
# Output: $LOCAL_PULL_DIR/database.db
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

mkdir -p "$PROJECT_ROOT/$LOCAL_PULL_DIR"
DEST="$PROJECT_ROOT/$LOCAL_PULL_DIR/database.db"
RDB="$REMOTE_PROJECT_DIR/$REMOTE_DB"

if [[ "${1:-}" == "--dump" ]]; then
    command -v sqlite3 >/dev/null || { echo "ERROR: local sqlite3 required for --dump" >&2; exit 2; }
    rm -f "$DEST"
    ssh_jetson "sqlite3 -readonly 'file:$RDB?mode=ro' .dump" | sqlite3 "$DEST"
else
    TMP="/tmp/hpid_db_pull_$$.db"
    cleanup() { ssh_jetson "rm -f '$TMP'" 2>/dev/null || true; }
    trap cleanup EXIT
    ssh_jetson "RDB='$RDB' TMP='$TMP' bash -s" <<'REMOTE'
set -euo pipefail
rm -f "$TMP"
if command -v sqlite3 >/dev/null 2>&1; then
    sqlite3 -readonly "file:$RDB?mode=ro" ".backup '$TMP'"
else
    export RDB TMP
    python3 - <<'PY'
import os, sqlite3
src = sqlite3.connect(f"file:{os.environ['RDB']}?mode=ro", uri=True)
dst = sqlite3.connect(os.environ["TMP"])
src.backup(dst)
dst.close()
src.close()
PY
fi
REMOTE
    scp "${SSH_OPTS[@]}" -q "$JETSON_HOST:$TMP" "$DEST"
fi

echo "Pulled DB -> $DEST"
DEST="$DEST" python3 - <<'PY'
import os, sqlite3
con = sqlite3.connect(f"file:{os.environ['DEST']}?mode=ro", uri=True)
print("integrity_check:", con.execute("PRAGMA integrity_check").fetchone()[0])
print("events rows:", con.execute("SELECT count(*) FROM events").fetchone()[0])
print("newest event:", con.execute("SELECT max(timestamp) FROM events").fetchone()[0], "(UTC)")
PY
