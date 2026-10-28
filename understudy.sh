#!/usr/bin/env bash
# The rig with one command, from the repo root:
#   ./understudy.sh --sim [robotd args]    robotd on the simulator: no arms, no cameras
#   ./understudy.sh [robotd args]          the cameras, then robotd on the real arms (press a for torque)
#   ./understudy.sh cameras [--view]       the cameras only, checked, and shown with --view; Ctrl+C stops them
# FAKE_CAMERAS=1 serves test patterns instead of the cameras. USE_CAMERAS=overhead,left_wrist runs a subset.
# First time on a Mac: cameras/setup.sh (the camera venv, and starting the cameras without the password).
set -eo pipefail
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot
source cameras/serve.sh

case "${1:-}" in
  --sim)
    exec python -m harness.robotd "$@"
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
    python -m harness.robotd "$@"
    ;;
esac
