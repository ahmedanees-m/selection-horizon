#!/usr/bin/env bash
# Supervisor for the Evo 2 scoring run.
#
# The run is roughly 76,000 requests over about a day, so it will outlive the
# session that started it and will meet at least one network problem. This
# script is the container's entrypoint and keeps the run going across all of
# them. Docker's restart policy brings the container back after a crash, an
# out-of-memory kill or a reboot; this script decides what to do when it
# returns.
#
# The scorer's exit code carries the meaning:
#
#     0    nothing left to score, the run is complete
#     1    genes were deferred by a network problem, worth another pass
#     2    the credential or the request is wrong, another pass cannot help
#
# Progress lives in the checkpoint, which is appended one gene at a time and
# flushed to disk, so a pass that dies loses only the genes in flight.

set -u

STATE_DIR="${SH_DATA_ROOT:-/work/data}/interim/evo2"
LOG="${STATE_DIR}/supervisor.log"
COMPLETE="${STATE_DIR}/COMPLETE"
FAILED="${STATE_DIR}/FAILED"

PAUSE_BETWEEN_PASSES=300

mkdir -p "${STATE_DIR}"

say() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "${LOG}"
}

# A finished or abandoned run must not start again when the container comes
# back after a reboot, so the container idles instead of exiting. Exiting would
# have the restart policy start it again immediately, in a loop.
hold() {
    say "$1"
    sleep infinity
}

[ -f "${COMPLETE}" ] && hold "already complete, holding"
[ -f "${FAILED}" ] && hold "previously stopped on a fatal error, holding"

say "supervisor started, pid $$"

pass=0
while true; do
    pass=$((pass + 1))
    say "pass ${pass} starting"

    python -m src.evo2.score --workers "${EVO2_WORKERS:-8}" >> "${LOG}" 2>&1
    status=$?

    case "${status}" in
        0)
            say "pass ${pass} finished, nothing left to score"
            date -u +%Y-%m-%dT%H:%M:%SZ > "${COMPLETE}"
            hold "run complete"
            ;;
        2)
            say "pass ${pass} stopped on a fatal error; not retrying"
            date -u +%Y-%m-%dT%H:%M:%SZ > "${FAILED}"
            hold "held after a fatal error"
            ;;
        *)
            say "pass ${pass} exited ${status}, retrying in ${PAUSE_BETWEEN_PASSES}s"
            sleep "${PAUSE_BETWEEN_PASSES}"
            ;;
    esac
done
