"""Record a clip from one camera stream for the Watch pipeline: e.g. Vlad passing the brush from hand to hand under the
overhead camera, for SAM 3.1 to turn into a score. Saves an MP4 and, next to it, a JSON with the camera model, so the
masks' pixels can be turned into table cm (or ask robotd: the to_table command).

    python -m harness.clip --seconds 8 --out clips/handpass.mp4 [--camera overhead]

It records every frame the camera server sends (the real cameras, or robotd --sim's), not a resampled copy.
"""
import argparse
import base64
import datetime
import json
import os
import time

import cv2
import numpy as np
import zmq

from .cameras import BASE_PORT, ORDER
from .scene import SCENE


def record(path, seconds, camera="overhead", base_port=BASE_PORT, scene_path=SCENE):
    """Write the clip and its JSON; returns the JSON's contents."""
    s = zmq.Context.instance().socket(zmq.SUB)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVTIMEO, 2000)
    s.setsockopt_string(zmq.SUBSCRIBE, "")
    s.connect(f"tcp://127.0.0.1:{base_port + ORDER.index(camera)}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    writer, stamps, started = None, [], None
    try:
        end = None
        while end is None or time.time() < end:
            try:
                msg = json.loads(s.recv_string())
            except zmq.Again:
                raise RuntimeError(f"no frames from {camera}: is the camera server (or robotd --sim) running?")
            if msg.get("repeat"):
                continue
            img = cv2.imdecode(np.frombuffer(base64.b64decode(msg["images"][camera]), np.uint8), cv2.IMREAD_COLOR)
            if writer is None:
                end = time.time() + seconds
                started = datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (img.shape[1], img.shape[0]))
            writer.write(img)
            stamps.append(msg["timestamps"][camera])
    finally:
        s.close()
        if writer is not None:
            writer.release()
    with open(scene_path) as f:
        cam = json.load(f)["camera"]
    span = stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0
    info = dict(video=os.path.basename(path), camera=camera, started=started, frames=len(stamps),
                fps=round((len(stamps) - 1) / span, 2) if span else None, timestamps=[round(t - stamps[0], 4) for t in stamps],
                camera_model=cam if camera == "overhead" else None,
                note="pixels to table cm: robotd's to_table command, or harness.scene.Camera.to_table")
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump(info, f, indent=1)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", default=os.path.join("clips", time.strftime("clip-%Y%m%d-%H%M%S.mp4")))
    ap.add_argument("--camera", default="overhead", choices=ORDER)
    a = ap.parse_args()
    print(f"recording {a.seconds:.0f} s from {a.camera}...", flush=True)
    info = record(a.out, a.seconds, a.camera)
    print(f"{a.out}: {info['frames']} frames at {info['fps']} fps")


if __name__ == "__main__":
    main()
