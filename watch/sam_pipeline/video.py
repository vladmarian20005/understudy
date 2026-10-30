"""ffmpeg / ffprobe wrappers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def cut_clip(src: Path, dst: Path, start: float, duration: float, width: int, fps: float) -> None:
    """Re-encode to H.264 (Ego4D full-scale is VP9, which SAM doesn't accept)."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src),
           "-t", f"{duration:.3f}", "-vf", f"scale={width}:-2,fps={fps}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-an", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)


def probe(path: Path) -> tuple[int, int, float, int]:
    """Return (width, height, fps, frame count) for a video file."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den), int(s.get("nb_read_frames") or 0)
