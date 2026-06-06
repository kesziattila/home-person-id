#!/usr/bin/env bash
# Run a canned diagnostic query from scripts/remote/queries/<name>.sql against the live DB.
#
# Usage:
#   scripts/remote/rquery_file.sh --list                      # list available queries
#   scripts/remote/rquery_file.sh <name> [key=value ...] [--csv]
#
# Parameters: .sql files declare defaults with header lines like "-- param: hours=24";
# every {{hours}} token in the SQL is replaced. Override on the command line:
#   scripts/remote/rquery_file.sh low_confidence_matches hours=48 threshold=0.7
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

QUERIES_DIR="$_REMOTE_DIR/queries"

if [[ "${1:-}" == "--list" || -z "${1:-}" ]]; then
    echo "Available queries (details: $QUERIES_DIR/index.md):"
    for f in "$QUERIES_DIR"/*.sql; do
        printf '  %-36s %s\n' "$(basename "${f%.sql}")" "$(sed -n 's/^-- purpose: //p' "$f")"
    done
    exit 0
fi

NAME="$1"; shift
FILE="$QUERIES_DIR/$NAME.sql"
[[ -f "$FILE" ]] || { echo "ERROR: no such query: $NAME (try --list)" >&2; exit 2; }

FMT=-json
declare -A P=()
while IFS='=' read -r k v; do
    [[ -n "$k" ]] && P[$k]="$v"
done < <(sed -n 's/^-- param: //p' "$FILE")
for arg in "$@"; do
    if [[ "$arg" == "--csv" ]]; then FMT=-csv; continue; fi
    [[ "$arg" == *=* ]] || { echo "ERROR: bad parameter '$arg' (expected key=value)" >&2; exit 2; }
    P[${arg%%=*}]="${arg#*=}"
done

SQL="$(cat "$FILE")"
for k in "${!P[@]}"; do
    SQL="${SQL//\{\{$k\}\}/${P[$k]}}"
done

if [[ "$SQL" == *'{{'* ]]; then
    echo "ERROR: unsubstituted parameter remains in $NAME.sql:" >&2
    grep -o '{{[a-z_]*}}' <<<"$SQL" | sort -u >&2
    exit 2
fi

run_remote_sql "$SQL" "$FMT"
