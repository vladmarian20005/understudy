"""Preview video: masks and boxes drawn over the clip, to judge SAM by eye."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from sam_pipeline.analysis import Analysis
from sam_pipeline.detections import Det
from sam_pipeline.masks import rle_to_mask

COLORS = {"tool": (230, 40, 40), "other_tool": (255, 160, 0), "left": (40, 120, 255),
          "right": (40, 200, 80), "other_hand": (160, 160, 160)}


def draw_box(img: np.ndarray, box, color, thick: int = 2) -> None:
    hgt, wid = img.shape[:2]
    l, t, r, b = (int(round(v)) for v in box)
    l, r = max(0, l), min(wid - 1, r - 1)
    t, b = max(0, t), min(hgt - 1, b - 1)
    if r <= l or b <= t:
        return
    img[t:t + thick, l:r] = color
    img[b - thick + 1:b + 1, l:r] = color
    img[t:b, l:l + thick] = color
    img[t:b, r - thick + 1:r + 1] = color


def render_preview(clip: Path, out: Path, results, a: Analysis, w: int, h: int, fps: float) -> None:
    """Tool red (look-alikes orange), left hand/gripper blue, right green, extra hands grey.
    Bottom bar: left half lights blue while the left side holds the tool, right half green;
    white while both hold it (the pass)."""
    by_frame: dict[int, list[tuple[Det, tuple]]] = {}
    track_ids = {d.obj_id for d in a.track}
    for d in results["object"][0]:
        by_frame.setdefault(d.frame, []).append((d, COLORS["tool"] if d.obj_id in track_ids else COLORS["other_tool"]))
    for d in results["holder"][0]:
        sides = a.holders.get(d.frame, {})
        side = next((s for s, x in sides.items() if x is d), None)
        by_frame.setdefault(d.frame, []).append((d, COLORS[side] if side else COLORS["other_hand"]))

    bar = 10
    reader = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(clip), "-f", "rawvideo",
                               "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    writer = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{w}x{h + bar}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                               "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(out)],
                              stdin=subprocess.PIPE)
    size = w * h * 3
    f = 0
    while True:
        buf = reader.stdout.read(size)
        if len(buf) < size:
            break
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).astype(np.float32)
        for d, color in by_frame.get(f, []):
            if d.rle is not None:
                m = rle_to_mask(d.rle)
                img[m] = img[m] * 0.5 + np.array(color) * 0.5
        img = img.astype(np.uint8)
        for d, color in by_frame.get(f, []):
            draw_box(img, d.box, color)
        strip = np.zeros((bar, w, 3), dtype=np.uint8)
        held = a.held[f] if f < len(a.held) else frozenset()
        if len(held) == 2:
            strip[:] = 255
        else:
            if "left" in held:
                strip[:, : w // 2] = COLORS["left"]
            if "right" in held:
                strip[:, w // 2:] = COLORS["right"]
        writer.stdin.write(np.vstack([img, strip]).tobytes())
        f += 1
    writer.stdin.close()
    reader.wait()
    writer.wait()
