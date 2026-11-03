"""The watch pipeline's command line: a video in; Muse Spark's reading of it, SAM 3.1's masks, a preview and a score
out, in <out>/videos/<name>/.

Hard budget: every SAM request is charged against a ledger file (default 60 minutes of video in total, across all
runs). A request that would go over is not sent. A request is charged for its whole clip, at least 50 frames (Meta's
billing minimum), and each prompt is its own request, so a 6 s clip with two prompts costs 12 s.

Setup: ffmpeg on PATH, and MODEL_API_KEY (the key Meta gave you) in the environment or .env.

    python -m watch --video watch/test_video_balls.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from sam_pipeline.budget import Budget
from sam_pipeline.muse_mode import run_video


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m watch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True, help="the video to read")
    ap.add_argument("--out", type=Path, default=Path("sam_out"))
    ap.add_argument("--budget-file", type=Path, default=Path.home() / ".sam31_budget.json",
                    help="ledger shared by every run and output folder (default: ~/.sam31_budget.json)")
    ap.add_argument("--max-minutes", type=float, default=60.0, help="hard cap on video sent to SAM, all runs")
    ap.add_argument("--base-url", default="https://api.meta.ai/v1")
    ap.add_argument("--no-rle", action="store_true", help="skip full-frame COCO RLE masks (raw SAM output is kept)")
    ap.add_argument("--no-preview", action="store_true", help="skip the preview.mp4 with masks drawn on")
    ap.add_argument("--object-attempts", type=int, default=3, metavar="N",
                    help="SAM requests to spend on the tool before giving up, trying a fallback "
                         "concept each time (default 3, 1 = no fallbacks). Each attempt is billed.")
    ap.add_argument("--retry-empty", action="store_true",
                    help="re-send concepts that returned nothing on an earlier run (costs again)")
    ap.add_argument("--object", help="skip Muse's choice and segment this concept instead")
    ap.add_argument("--max-objects", type=int, default=2, metavar="N",
                    help="how many of Muse's objects to segment (default 2). Each is a billed SAM request")
    ap.add_argument("--holder-prompt", default="hand")
    ap.add_argument("--muse-model", default=os.environ.get("MUSE_MODEL", "muse-spark-1.3"))
    ap.add_argument("--muse-prompt", help="override the question put to Muse")
    ap.add_argument("--width", type=int, default=1280,
                    help="width to re-encode at (default 1280); SAM is billed per frame, not per pixel")
    ap.add_argument("--remuse", action="store_true", help="re-ask Muse even if muse.json is cached (costs again)")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.max_minutes > 60:
        sys.exit("--max-minutes above 60 is not allowed (1 hour hard limit)")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH")
    args.out.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.budget_file, args.max_minutes * 60)
    print(f"SAM budget: {budget.used / 60:.1f} of {budget.limit_s / 60:.1f} min used ({args.budget_file})")
    run_video(args, budget)
