#!/usr/bin/env bash
# Install (or remove) the systemd unit that keeps the Jetson GPU at its maximum clock
# (scripts/jetson/gpu-min-freq.service; why: docs/JETSON.md "GPU clock").
#
# Usage: scripts/remote/install_gpu_clock.sh [--uninstall|--status]
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

UNIT=gpu-min-freq.service
case "${1:-install}" in
    install)
        scp -q "${SSH_OPTS[@]}" "$PROJECT_ROOT/scripts/jetson/$UNIT" "$JETSON_HOST:/etc/systemd/system/$UNIT"
        ssh_jetson "systemctl daemon-reload && systemctl enable --now $UNIT >/dev/null 2>&1 && systemctl is-active $UNIT"
        ;;
    --uninstall)
        ssh_jetson "systemctl disable --now $UNIT >/dev/null 2>&1; rm -f /etc/systemd/system/$UNIT; systemctl daemon-reload"
        ;;
    --status) ;;
    *) sed -n '2,5p' "$0"; exit 2 ;;
esac
ssh_jetson "systemctl is-enabled $UNIT 2>/dev/null || echo 'not installed'; for d in /sys/class/devfreq/*.gpu; do echo \"\$(basename \$d): min \$((\$(cat \$d/min_freq) / 1000000)) MHz, max \$((\$(cat \$d/max_freq) / 1000000)) MHz, now \$((\$(cat \$d/cur_freq) / 1000000)) MHz\"; done"
