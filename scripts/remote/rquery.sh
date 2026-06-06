#!/usr/bin/env bash
# Read-only SQL query against the live Jetson database.
#
# Usage:
#   scripts/remote/rquery.sh 'SELECT count(*) FROM events'
#   scripts/remote/rquery.sh --csv 'SELECT ...'
#   echo 'SELECT ...' | scripts/remote/rquery.sh -      # SQL from stdin
#
# Output: JSON array (default) or CSV.
# Notes: timestamps in the DB are UTC. For multi-line/complex queries prefer
# rquery_file.sh with a canned query in scripts/remote/queries/ (no quoting issues).
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

FMT=-json
SQL=""
for arg in "$@"; do
    case "$arg" in
        --csv) FMT=-csv ;;
        -)     SQL="$(cat)" ;;
        *)     SQL="$arg" ;;
    esac
done
[[ -n "$SQL" ]] || { echo "usage: rquery.sh [--csv] '<SQL>' (or '-' for stdin)" >&2; exit 2; }

run_remote_sql "$SQL" "$FMT"
