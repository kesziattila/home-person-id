#!/usr/bin/env bash
# Start/stop/restart the live system on the Jetson.
#
# Usage: scripts/remote/restart.sh [start|stop|restart|status]   (default: restart)
#
# start runs (on the Jetson, inside REMOTE_PROJECT_DIR):
#   nohup HPID_NO_CONSOLE_LOG=1 PYTHONPATH=. python -m src.main >> $REMOTE_STDOUT_LOG 2>&1
# App logging goes to the rotated $REMOTE_LOG via RotatingFileHandler (the console
# handler is suppressed by HPID_NO_CONSOLE_LOG to avoid duplicating it); raw stdout
# (perf-report prints, tracebacks) lands in $REMOTE_STDOUT_LOG. pid -> data/app.pid.
# Stop is graceful first (SIGTERM, waits for the DB checkpoint in
# PersonIDSystem.stop()), SIGKILL only after a 20s timeout.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

ACTION="${1:-restart}"
case "$ACTION" in start|stop|restart|status) ;; *)
    echo "usage: restart.sh [start|stop|restart|status]" >&2; exit 2 ;;
esac

ssh_jetson "DIR='$REMOTE_PROJECT_DIR' LOG='$REMOTE_LOG' OUTLOG='$REMOTE_STDOUT_LOG' ACTION='$ACTION' APP_ARGS='$REMOTE_APP_ARGS' bash -s" <<'REMOTE'
set -uo pipefail
cd "$DIR" || { echo "ERROR: remote project dir missing: $DIR"; exit 1; }
PIDFILE=data/app.pid

is_running() {
    [[ -f $PIDFILE ]] && kill -0 "$(cat $PIDFILE)" 2>/dev/null
}

do_stop() {
    local pid=""
    [[ -f $PIDFILE ]] && pid=$(cat $PIDFILE)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping pid $pid (graceful) ..."
        kill "$pid"
        for _ in $(seq 1 20); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "Graceful stop timed out, sending SIGKILL"
            kill -9 "$pid" 2>/dev/null || true
        fi
        echo "Stopped."
    elif pgrep -f 'python.*src\.main' >/dev/null 2>&1; then
        echo "No valid pidfile; stopping stray src.main process(es):"
        pgrep -af 'python.*src\.main'
        pkill -f 'python.*src\.main' || true
        sleep 3
        pkill -9 -f 'python.*src\.main' 2>/dev/null || true
    else
        echo "Not running."
    fi
    rm -f $PIDFILE
}

do_start() {
    if is_running; then
        echo "Already running (pid $(cat $PIDFILE))."
        return
    fi
    mkdir -p "$(dirname "$LOG")" "$(dirname "$OUTLOG")"
    echo "Starting python -m src.main $APP_ARGS ..."
    # shellcheck disable=SC1091
    source venv/bin/activate
    # Keep the (unrotated) stdout log from growing unbounded across restarts
    if [[ -f "$OUTLOG" ]] && (( $(stat -c%s "$OUTLOG") > 20000000 )); then
        tail -c 1000000 "$OUTLOG" > "$OUTLOG.tmp" && mv "$OUTLOG.tmp" "$OUTLOG"
    fi
    # shellcheck disable=SC2086  # APP_ARGS is intentionally word-split
    nohup env PYTHONPATH=. HPID_NO_CONSOLE_LOG=1 python -m src.main $APP_ARGS >> "$OUTLOG" 2>&1 < /dev/null &
    echo $! > $PIDFILE
    sleep 4
    if is_running; then
        echo "Started (pid $(cat $PIDFILE)). Last log lines:"
        tail -n 20 "$LOG"
    else
        echo "FAILED to start — last log lines:"
        tail -n 20 "$LOG" 2>/dev/null
        echo "--- stdout/stderr ($OUTLOG):"
        tail -n 40 "$OUTLOG" 2>/dev/null
        rm -f $PIDFILE
        exit 1
    fi
}

case "$ACTION" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; do_start ;;
    status)  if is_running; then echo "RUNNING (pid $(cat $PIDFILE))"; else echo "NOT RUNNING"; fi ;;
esac
REMOTE
