"""Overhead frames of a recorded episode, read from its LeRobot dataset, with the table grid and the target drawn on.

robotd records in lerobot's v3 layout: a camera's video file holds many episodes back to back, and
meta/episodes/*.parquet says which file each episode is in and where it starts. Decoding goes through lerobot's own
decoder, so AV1 (robotd's codec) works wherever training does.
"""
import glob
import json
import os

import cv2
import numpy as np

from harness import overlay
from harness.recorder import CHECK_CM

OVERHEAD = "observation.images.left_overhead"
TARGET = (0, 0, 255)  # BGR red


def episode_row(root, episode):
    """The episode's row in meta/episodes: which video file it's in and where it starts."""
    import pandas as pd

    for path in sorted(glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet"))):
        df = pd.read_parquet(path)
        hit = df[df["episode_index"] == int(episode)]
        if len(hit):
            return hit.iloc[0].to_dict()
    raise KeyError(f"episode {episode} isn't in {root}/meta/episodes")


def video_file(root, row, key=OVERHEAD):
    """(path of the video holding the episode, the episode's start in it in s, fps)."""
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    rel = info["video_path"].format(video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]),
                                    file_index=int(row[f"videos/{key}/file_index"]))
    return os.path.join(root, rel), float(row[f"videos/{key}/from_timestamp"]), int(info["fps"])


def read(root, episode, indices, key=OVERHEAD):
    """The episode's frames at these indices (0 = its first frame), as BGR uint8 images."""
    from lerobot.datasets.video_utils import decode_video_frames

    path, start, fps = video_file(root, episode_row(root, episode), key)
    ts = [start + i / fps for i in indices]
    tol = 1 / fps - 1e-4  # lerobot's own tolerance
    try:
        t = decode_video_frames(path, ts, tolerance_s=tol, return_uint8=True)
    except TypeError:  # older lerobot: float frames only
        t = decode_video_frames(path, ts, tolerance_s=tol)
    arr = t.numpy()
    if arr.dtype != np.uint8:
        arr = (arr * 255).round().clip(0, 255).astype(np.uint8)
    return [np.ascontiguousarray(a.transpose(1, 2, 0)[:, :, ::-1]) for a in arr]  # (N, C, H, W) RGB -> HWC BGR


def annotate(img, scene, target_cm=None, caption=None):
    """The table grid (as the brain sees it), the target with robotd's check radius around it, and a caption."""
    img = img.copy()
    overlay.grid(img, scene)
    if target_cm is not None:
        cam = scene.camera
        x, y = target_cm[0] / 100, target_cm[1] / 100
        ring = [[x + CHECK_CM / 100 * np.cos(a), y + CHECK_CM / 100 * np.sin(a), 0]
                for a in np.linspace(0, 2 * np.pi, 33)]
        overlay.polyline(img, cam, np.array(ring), TARGET, 2)
        c = cam.project([[x, y, 0]])[0]
        if overlay.visible(img, c):
            cv2.drawMarker(img, (int(c[0]), int(c[1])), TARGET, cv2.MARKER_CROSS, 10, 2)
            overlay.text(img, "target", (c[0] + 8, c[1] - 8), TARGET)
    if caption:
        overlay.text(img, caption, (8, 18), scale=0.5)
    return img


def jpeg_b64(img, quality=85):
    import base64

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("couldn't encode the frame")
    return base64.b64encode(buf.tobytes()).decode()
