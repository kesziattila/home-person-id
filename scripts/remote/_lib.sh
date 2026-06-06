#!/usr/bin/env bash
# Shared helpers for scripts/remote/* — source this, don't execute it.
# Configuration: copy remote.env.example to remote.env and fill in real values.

set -euo pipefail

_REMOTE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$_REMOTE_DIR/../.." && pwd)"

if [[ ! -f "$_REMOTE_DIR/remote.env" ]]; then
    echo "ERROR: $_REMOTE_DIR/remote.env not found." >&2
    echo "Copy remote.env.example to remote.env and set JETSON_HOST / REMOTE_PROJECT_DIR." >&2
    exit 2
fi
# shellcheck disable=SC1091
source "$_REMOTE_DIR/remote.env"

: "${JETSON_HOST:?JETSON_HOST must be set in remote.env}"
: "${REMOTE_PROJECT_DIR:?REMOTE_PROJECT_DIR must be set in remote.env}"
: "${REMOTE_DB:=data/database.db}"
: "${REMOTE_LOG:=data/app.log}"
: "${REMOTE_STDOUT_LOG:=data/stdout.log}"
: "${LOCAL_PULL_DIR:=local/remote}"
: "${REMOTE_APP_ARGS:=}"

# Never hang on a password prompt; fail fast when the Jetson is unreachable.
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5)

ssh_jetson() {
    ssh "${SSH_OPTS[@]}" "$JETSON_HOST" "$@"
}

# Run a read-only SQL query against the live DB on the Jetson.
# Usage: run_remote_sql "<SQL>" [-json|-csv]
# - Always opens the DB with mode=ro: WAL-aware consistent reads, no locks taken.
#   (Do NOT use immutable=1 — it skips the WAL and returns stale data.)
# - SQL is base64-encoded across the ssh boundary to avoid quoting bugs.
# - Uses the sqlite3 CLI if present on the Jetson, otherwise python3 stdlib.
run_remote_sql() {
    local sql="$1" fmt="${2:--json}"
    local b64
    b64=$(printf '%s' "$sql" | base64 -w0)
    ssh_jetson "SQL_B64='$b64' RDB='$REMOTE_PROJECT_DIR/$REMOTE_DB' FMT='$fmt' bash -s" <<'REMOTE'
set -euo pipefail
sql=$(printf '%s' "$SQL_B64" | base64 -d)
if command -v sqlite3 >/dev/null 2>&1; then
    printf '%s' "$sql" | sqlite3 -readonly "$FMT" "file:$RDB?mode=ro"
else
    export SQL_B64 RDB FMT
    python3 - <<'PY'
import base64, csv, json, os, sqlite3, sys

sql = base64.b64decode(os.environ["SQL_B64"]).decode()
con = sqlite3.connect(f"file:{os.environ['RDB']}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
rows = [dict(r) for r in con.execute(sql).fetchall()]
if os.environ.get("FMT") == "-csv":
    w = csv.writer(sys.stdout)
    if rows:
        w.writerow(rows[0].keys())
        for r in rows:
            w.writerow(r.values())
else:
    print(json.dumps(rows, default=str))
PY
fi
REMOTE
}
