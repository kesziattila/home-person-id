#!/usr/bin/env bash
# Fetch snapshot JPEGs from the live Jetson so they can be viewed locally (Read tool).
#
# Usage:
#   scripts/remote/fetch_snapshots.sh --event 123[,124,...]          # by events.id
#   scripts/remote/fetch_snapshots.sh --query <name> [key=value ...] # any canned query returning snapshot_path
#   scripts/remote/fetch_snapshots.sh --path data/snapshots/face_matches/x.jpg [more...]
#   options: --limit N   (default 50 — guards against accidentally pulling thousands)
#
# snapshot_path in the DB is project-relative (already starts with data/...), so files
# land under $LOCAL_PULL_DIR/snapshots/data/... preserving that structure.
# Prints the local path of every fetched file.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

LIMIT=50
MODE=""
ARG=""
EXTRA=()
PATHS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --event) MODE=event; ARG="${2:?event id(s) required}"; shift 2 ;;
        --query) MODE=query; ARG="${2:?query name required}"; shift 2 ;;
        --path)  MODE=path; shift
                 while [[ $# -gt 0 && "$1" != --* ]]; do PATHS+=("$1"); shift; done ;;
        --limit) LIMIT="${2:?}"; shift 2 ;;
        *)       EXTRA+=("$1"); shift ;;
    esac
done
[[ -n "$MODE" ]] || { echo "usage: fetch_snapshots.sh --event <ids> | --query <name> [params] | --path <relpath...> [--limit N]" >&2; exit 2; }

case "$MODE" in
    event)
        IDS=$(tr -cd '0-9,' <<<"$ARG")
        [[ -n "$IDS" ]] || { echo "ERROR: no numeric event ids in '$ARG'" >&2; exit 2; }
        SQL="SELECT snapshot_path FROM events WHERE id IN ($IDS) AND snapshot_path IS NOT NULL"
        mapfile -t PATHS < <(run_remote_sql "$SQL" -json | python3 -c '
import json, sys
data = sys.stdin.read().strip()
for row in (json.loads(data) if data else []):
    if row.get("snapshot_path"):
        print(row["snapshot_path"])
')
        ;;
    query)
        # sqlite3 -json prints nothing (not "[]") for zero rows — handle empty input
        mapfile -t PATHS < <("$_REMOTE_DIR/rquery_file.sh" "$ARG" ${EXTRA[@]+"${EXTRA[@]}"} | python3 -c '
import json, sys
data = sys.stdin.read().strip()
for row in (json.loads(data) if data else []):
    if row.get("snapshot_path"):
        print(row["snapshot_path"])
')
        ;;
    path) ;;
esac

if [[ ${#PATHS[@]} -eq 0 ]]; then
    echo "No snapshot paths found — nothing to fetch."
    exit 0
fi

if (( ${#PATHS[@]} > LIMIT )); then
    echo "NOTE: capping ${#PATHS[@]} files to --limit $LIMIT"
    PATHS=("${PATHS[@]:0:LIMIT}")
fi

DEST="$PROJECT_ROOT/$LOCAL_PULL_DIR/snapshots"
mkdir -p "$DEST"

LIST=$(mktemp)
trap 'rm -f "$LIST"' EXIT
printf '%s\n' "${PATHS[@]}" > "$LIST"

# Single batched transfer (one ssh session), preserving relative structure.
rsync -a -e "ssh ${SSH_OPTS[*]}" --files-from="$LIST" \
    "$JETSON_HOST:$REMOTE_PROJECT_DIR/" "$DEST/"

echo "Fetched ${#PATHS[@]} file(s):"
for p in "${PATHS[@]}"; do
    echo "$DEST/$p"
done
