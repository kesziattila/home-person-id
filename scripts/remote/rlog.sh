#!/usr/bin/env bash
# Bounded, context-safe access to the live Jetson log.
# $REMOTE_LOG contains app logging AND raw stdout (perf reports) — restart.sh
# redirects both there. Every mode is bounded by default: the log grows fast,
# never dump it whole.
#
# Usage:
#   scripts/remote/rlog.sh tail [N]              last N lines (default 100)
#   scripts/remote/rlog.sh grep <pattern> [N]    last N matching lines (default 50)
#   scripts/remote/rlog.sh around '<ts>' [C]     C context lines around the first line
#                                                matching the timestamp prefix (default 30)
#                                                NOTE: log timestamps are Jetson LOCAL time,
#                                                DB event timestamps are UTC.
#   scripts/remote/rlog.sh errors [N]            last N ERROR/WARNING lines (default 50)
#   scripts/remote/rlog.sh perf                  last "--- Performance Report ---" block
#                                                (reads $REMOTE_STDOUT_LOG — perf reports are
#                                                print()ed, not logged)
#   scripts/remote/rlog.sh stdout [N]            last N lines of the raw stdout log
#   scripts/remote/rlog.sh follow                live tail -f (for humans, not the agent loop)
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

LOG="$REMOTE_PROJECT_DIR/$REMOTE_LOG"
OUTLOG="$REMOTE_PROJECT_DIR/$REMOTE_STDOUT_LOG"
CMD="${1:-tail}"
shift || true

case "$CMD" in
    tail)
        N="${1:-100}"
        ssh_jetson "tail -n $N '$LOG'"
        ;;
    grep)
        PAT="${1:?pattern required}"
        N="${2:-50}"
        PB64=$(printf '%s' "$PAT" | base64 -w0)
        ssh_jetson "grep -a -- \"\$(printf '%s' $PB64 | base64 -d)\" '$LOG' | tail -n $N"
        ;;
    around)
        TS="${1:?timestamp prefix required, e.g. '2026-06-06 14:23'}"
        C="${2:-30}"
        TB64=$(printf '%s' "$TS" | base64 -w0)
        ssh_jetson "LOG='$LOG' C=$C TB64='$TB64' bash -s" <<'REMOTE'
set -uo pipefail
ts=$(printf '%s' "$TB64" | base64 -d)
line=$(grep -an -m1 -- "$ts" "$LOG" | cut -d: -f1)
if [[ -z "${line:-}" ]]; then
    echo "(timestamp not found in log: $ts)"
    exit 0
fi
start=$(( line - C )); (( start < 1 )) && start=1
sed -n "${start},$(( line + C ))p" "$LOG"
REMOTE
        ;;
    errors)
        N="${1:-50}"
        ssh_jetson "grep -aE ' - (ERROR|WARNING) - ' '$LOG' | tail -n $N"
        ;;
    stdout)
        N="${1:-100}"
        ssh_jetson "tail -n $N '$OUTLOG'"
        ;;
    perf)
        # Block starts at the last "--- Performance Report ---" header (a print(),
        # so unstamped) and ends at the short all-dash terminator line; the 75-dash
        # table separator inside the block is longer, so length<70 distinguishes them.
        ssh_jetson "LOG='$OUTLOG' bash -s" <<'REMOTE'
set -uo pipefail
line=$(grep -an -- '--- Performance Report' "$LOG" | tail -n 1 | cut -d: -f1)
if [[ -z "${line:-}" ]]; then
    echo "(no performance report found in log)"
    exit 0
fi
tail -n "+$line" "$LOG" | awk 'NR > 1 && /^-+$/ && length($0) < 70 { print; exit } { print } NR > 120 { exit }'
REMOTE
        ;;
    follow)
        ssh "${SSH_OPTS[@]}" -t "$JETSON_HOST" "tail -f '$LOG'"
        ;;
    *)
        echo "usage: rlog.sh tail|grep|around|errors|perf|stdout|follow (see header of $0)" >&2
        exit 2
        ;;
esac
