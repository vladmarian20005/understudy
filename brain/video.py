"""A video of a person doing the task, for the plan: frames from it, what they do, and how they held what they held."""
import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MIN_MASK_PX = 300  # a mask smaller than this says nothing about how a thing leans
MIN_FRAMES = 10
LIFTED = 0.05  # off the table: its lowest pixel this share of the image's height above where it stood


def frames(path, n=6, width=480):
    """n frames spread over a video, as JPEG base64, with their times."""
    cap = cv2.VideoCapture(path)
    count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30
    out = []
    for k in range(n):
        i = int((k + 0.5) * count / n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if not ok:
            continue
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.append((round(i / fps, 1), base64.b64encode(buf.tobytes()).decode()))
    return out


def outputs(path):
    """Where watch keeps what it read from this video: next to it (its video mode's clip), or under its stem."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return [os.path.dirname(os.path.abspath(path)), os.path.join(ROOT, "watch", "sam_out", "videos", stem),
            os.path.join(ROOT, "sam_out", "videos", stem)]


def reading(path):
    """What the person in a video does: (summary, objects), from watch's cached Muse reading, or read now."""
    for folder in outputs(path):
        cache = os.path.join(folder, "muse.json")
        if os.path.exists(cache):
            d = json.load(open(cache))
            return d.get("summary", ""), d.get("objects", [])
    from watch.sam_pipeline.credentials import api_key
    from watch.sam_pipeline.muse_client import PROMPT, MuseClient
    c = MuseClient(api_key(), "https://api.meta.ai/v1")
    r = c.watch(c.upload(Path(path)), PROMPT)
    return r.summary, r.objects


def held(path):
    """What SAM's masks of the video (watch's video mode: masks_object.json, the object a hand held) say about how it was
    held: when it was off the table and how it leaned. Muse's words can leave that out. '' when watch hasn't segmented
    the video."""
    folder = next((f for f in outputs(path) if os.path.exists(os.path.join(f, "masks_object.json"))), None)
    if folder is None:
        return ""
    from watch.sam_pipeline.masks import rle_to_mask
    masks = json.load(open(os.path.join(folder, "masks_object.json")))
    score_path = os.path.join(folder, "score.json")
    score = json.load(open(score_path)) if os.path.exists(score_path) else {}
    fps = float(score.get("fps") or 30.0)
    name = (score.get("object") or {}).get("name") or "object"
    rows = []  # (frame, lean deg: + its top toward the image's right, lowest pixel)
    for f, dets in sorted(masks["frames"].items(), key=lambda kv: int(kv[0])):
        if not dets:
            continue
        ys, xs = np.nonzero(rle_to_mask(dets[0]["rle"]))
        if len(xs) < MIN_MASK_PX:
            continue
        w, v = np.linalg.eigh(np.cov(np.vstack([xs, ys])))
        ax = v[:, 1] if v[1, 1] < 0 else -v[:, 1]  # the long axis, pointing up the image
        rows.append((int(f), float(np.degrees(np.arctan2(ax[0], -ax[1]))), int(ys.max())))
    if len(rows) < MIN_FRAMES:
        return ""
    table = max(r[2] for r in rows)  # its lowest point standing on the table
    up = [r for r in rows if r[2] < table - LIFTED * masks["size_px"][1]]
    if not up:
        return ""
    lean = float(np.median([r[1] for r in up]))
    most = max((r[1] for r in up), key=abs)
    toward = "the image's right: the person's left hand, the camera facing them" if lean > 0 else \
        "the image's left: the person's right hand, the camera facing them"
    return (f"SAM's masks of the {name}: off the table from {up[0][0] / fps:.1f} s to {up[-1][0] / fps:.1f} s, "
            f"leaning about {abs(lean):.0f} deg from upright in the image (up to {abs(most):.0f}), its top toward "
            f"{toward}; the image shows the lean flattened, so it was tilted more than that.")
