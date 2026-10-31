#!/usr/bin/env bash
# The rig with one command, from the repo root:
#   ./understudy.sh --sim [robotd args]    robotd on the simulator: no arms, no cameras
#   ./understudy.sh [robotd args]          the cameras, then robotd on the real arms (press a for torque)
#   ./understudy.sh cameras [--view]       the cameras only, checked, and shown with --view; Ctrl+C stops them
#   ./understudy.sh check                  preflight: is everything ready for a session (robotd running)?
#   ./understudy.sh view                   the three cameras, the grid, the grippers and the arms' state
# BRAIN="<command>" and UI="<command>" start next to robotd once it answers (e.g. BRAIN="python -m brain"), with
# their output in logs/brain.log and logs/ui.log and the settings in .env; they stop when robotd does.
# FAKE_CAMERAS=1 serves test patterns instead of the cameras. USE_CAMERAS=overhead,left_wrist runs a subset.
# First time on a Mac: cameras/setup.sh (the camera venv, and starting the cameras without the password).
set -eo pipefail
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot
source cameras/serve.sh

SIDECARS=()

start_sidecar() {  # name, command: runs once robotd answers, in the background
  mkdir -p logs
  (
    until python -c "from harness import robot; robot.status()" > /dev/null 2>&1; do sleep 0.5; done
    exec bash -c "$2"
  ) > "logs/$1.log" 2>&1 &
  SIDECARS+=("$!")
  echo "$1 starts once robotd answers (log: logs/$1.log)"
}

stop_sidecars() {
  for pid in "${SIDECARS[@]}"; do
    pkill -P "${pid}" 2> /dev/null || true
    kill "${pid}" 2> /dev/null || true
  done
}

start_sidecars() {
  if [ -f .env ]; then
    set -a
    source .env
    set +a
  fi
  [ -n "${BRAIN}" ] && start_sidecar brain "${BRAIN}"
  [ -n "${UI}" ] && start_sidecar ui "${UI}"
  trap 'stop_sidecars; stop_cameras' EXIT
}

case "${1:-}" in
  --sim)
    start_sidecars
    python -m harness.robotd "$@"
    ;;
  check)
    exec python -m harness.preflight
    ;;
  view)
    exec python -m harness.viewer
    ;;
  cameras)
    start_cameras
    if [ "${2:-}" = "--view" ]; then
      python -m harness.cameras --view --timeout 5
    else
      echo "Cameras up (log: cameras/server.log). Ctrl+C stops them."
      wait "${CAMERA_SERVER_PID}"
    fi
    ;;
  *)
    start_cameras
    start_sidecars
    python -m harness.robotd "$@"
    ;;
esac
