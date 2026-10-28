# Sourced by ./understudy.sh (from the repo root, lerobot env active). start_cameras starts cameras/server.py as root
# through sudo and waits until every camera streams; the server exits by itself when the launcher does.
# Environment: USE_CAMERAS=overhead,right_wrist (a subset), CAMERA_PORT (the first port, default in cameras.json),
# FAKE_CAMERAS=1 (test patterns: no cameras, no root).

CAMERAS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

start_cameras() {
  local py="${CAMERAS_DIR}/.uvcenv/bin/python" server="${CAMERAS_DIR}/server.py" log="${CAMERAS_DIR}/server.log"
  local args=(--parent-pid $$)
  if pgrep -f "cameras/server.py" > /dev/null; then
    echo "A camera server is already running (pgrep -fl cameras/server.py). Stop it with: touch cameras/.stop"
    return 1
  fi
  [ -n "${USE_CAMERAS}" ] && args+=(--cameras "${USE_CAMERAS}")
  [ -n "${CAMERA_PORT}" ] && args+=(--port "${CAMERA_PORT}")
  if [ "${FAKE_CAMERAS}" = 1 ]; then
    echo "Starting the camera server with test patterns..."
    python "${server}" --fake "${args[@]}" > "${log}" &
  else
    if [ ! -x "${py}" ]; then
      echo "No camera venv in ${CAMERAS_DIR}/.uvcenv: run cameras/setup.sh once"
      return 1
    fi
    # Absolute paths, so the rule cameras/setup.sh installs lets it start without the password.
    if sudo -n -l "${py}" "${server}" "${args[@]}" > /dev/null 2>&1; then
      echo "Starting the camera server..."
    else
      echo "Starting the camera server (needs your Mac password; cameras/setup.sh makes it start without)..."
      sudo -v || return 1
    fi
    sudo -n "${py}" "${server}" "${args[@]}" > "${log}" &
  fi
  CAMERA_SERVER_PID=$!
  trap stop_cameras EXIT
  if ! python -m harness.cameras --timeout 60 --server-pid "${CAMERA_SERVER_PID}"; then
    echo "The cameras didn't all start. The end of cameras/server.log:"
    tail -n 20 "${log}"
    return 1
  fi
}

stop_cameras() {
  # A user can't signal the root server: the stop file reaches it (and it also exits with this shell).
  if [ -n "${CAMERA_SERVER_PID}" ] && ps -p "${CAMERA_SERVER_PID}" > /dev/null; then
    touch "${CAMERAS_DIR}/.stop"
  fi
  CAMERA_SERVER_PID=
}
