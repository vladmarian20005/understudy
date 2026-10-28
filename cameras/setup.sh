#!/usr/bin/env bash
# One-time setup of the camera server on this Mac. It asks for your password once.
# 1. cameras/.uvcenv: a venv on top of the lerobot env with pupil-labs-uvc (libuvc), unless it's there already.
# 2. A sudoers rule so ./understudy.sh starts cameras/server.py as root without the password:
#    /etc/sudoers.d/understudy-cameras. It names this checkout's absolute paths: run this again if the repo moves.
#    Remove it with: sudo rm /etc/sudoers.d/understudy-cameras
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
venv="${root}/cameras/.uvcenv"

if ! "${venv}/bin/python" -c "import uvc" 2> /dev/null; then
  echo "Creating ${venv}..."
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate lerobot
  python -m venv --system-site-packages "${venv}"
  "${venv}/bin/pip" install pupil-labs-uvc
fi
"${venv}/bin/python" -c "import uvc, cv2, zmq, numpy" || { echo "${venv} can't import uvc, cv2, zmq and numpy"; exit 1; }

rule="$(id -un) ALL=(root) NOPASSWD: ${venv}/bin/python ${root}/cameras/server.py *"
target=/etc/sudoers.d/understudy-cameras
staged="${target}.new"  # sudo ignores files in sudoers.d with a dot in the name, so a bad rule can't lock it out
tmp="$(mktemp)"
echo "${rule}" > "${tmp}"
sudo install -m 0440 -o root -g wheel "${tmp}" "${staged}"
rm -f "${tmp}"
if ! sudo visudo -cf "${staged}" > /dev/null; then
  sudo rm -f "${staged}"
  echo "The rule didn't parse; nothing installed."
  exit 1
fi
sudo mv "${staged}" "${target}"
echo "Installed ${target}:"
echo "  ${rule}"
