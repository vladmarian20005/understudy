#!/usr/bin/env python3
"""
SAM 3.1 pipeline for "Watch, Delegate, Distill" (Andrei's side of the Understudy contracts).

Two modes:

  ego4d    Find crafting clips where someone passes a tool from one hand to the other,
           cut them, segment hands + the tool with SAM 3.1 through Meta Model API,
           and write one *score* per clip (what the human video says should happen).

  episode  Segment a recorded robot episode's overhead video and write the label's
           "vision" section (contract: datasets/<dataset>/sidecar/<episode>.json),
           optionally handing it to robotd through harness.robot.episode_label.

Hard budget: every SAM request is charged against a ledger file (default 60 minutes of
video in total, across all runs and both modes). A request that would go over is not sent.
A request is charged for its whole clip, at least 50 frames (Meta's billing minimum), and
each prompt is its own request, so a 6 s clip with two prompts costs 12 s.

Setup:
    pip install requests numpy meta-sam-parser
    ffmpeg on PATH
    export MODEL_API_KEY=...        # the key Meta gave you

Examples, from the repo root:
    # 1. see what would be processed, no API calls, no cost
    python -m watch.sam_ego4d ego4d --metadata ego4d.json --narrations narration.json \
        --videos ~/ego4d/v2/full_scale --dry-run

    # 2. process the 10 best clips
    python -m watch.sam_ego4d ego4d --metadata ego4d.json --narrations narration.json \
        --videos ~/ego4d/v2/full_scale --max-clips 10

    # 3. a robot episode (--send needs robotd running on the same dataset)
    python -m watch.sam_ego4d episode --dataset understudy --episode 37 \
        --video ep37_overhead.mp4 --object brush --send
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from sam_pipeline.budget import Budget
from sam_pipeline.ego4d import run_ego4d
from sam_pipeline.episode import run_episode
from sam_pipeline.muse_mode import run_video


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    sub = ap.add_subparsers(dest="mode", required=True)

    e = sub.add_parser("ego4d")
    e.add_argument("--metadata", type=Path, required=True, help="ego4d.json")
    e.add_argument("--narrations", type=Path, required=True, help="annotations/narration.json")
    e.add_argument("--videos", type=Path,
                   help="folder with <video_uid>.mp4 full_scale videos and/or <clip_uid>.mp4 "
                        "clips (searched recursively)")
    e.add_argument("--clip-annotations", type=Path, metavar="DIR",
                   help="annotation folder used to locate <clip_uid>.mp4 clips inside their "
                        "parent video (default: the folder --narrations is in)")
    e.add_argument("--max-clips", type=int, default=20)
    e.add_argument("--before", type=float, default=2.5, help="seconds before the narration")
    e.add_argument("--after", type=float, default=3.5, help="seconds after the narration")
    e.add_argument("--fps", type=float, default=30.0)
    e.add_argument("--width", type=int, default=1280,
                   help="width to re-encode clips at (default 1280). SAM is billed per frame, "
                        "not per pixel, so resolution is free -- at 640 a held brush shrinks to "
                        "a few dozen pixels and SAM returns nothing for it")
    e.add_argument("--any-object", action="store_true", help="also keep objects not on the good list")
    e.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("episode")
    p.add_argument("--video", type=Path, required=True, help="the episode's overhead video")
    p.add_argument("--dataset", default="understudy",
                   help="the dataset the episode belongs to (datasets/<name>); results are kept per dataset")
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--object", required=True, help='SAM prompt, e.g. "brush"')
    p.add_argument("--holder-prompt", default="robot gripper")
    p.add_argument("--send", action="store_true", help="store it via harness.robot.episode_label")

    v = sub.add_parser("video", help="any video: Muse Spark names the actions and objects, SAM segments them")
    v.add_argument("--video", type=Path, required=True, help="the video to read")
    v.add_argument("--object", help="skip Muse's choice and segment this concept instead")
    v.add_argument("--max-objects", type=int, default=2, metavar="N",
                   help="how many of Muse's objects to segment (default 2). Each is a billed SAM request")
    v.add_argument("--holder-prompt", default="hand")
    v.add_argument("--muse-model", default=os.environ.get("MUSE_MODEL", "muse-spark-1.3"))
    v.add_argument("--muse-prompt", help="override the question put to Muse")
    v.add_argument("--width", type=int, default=1280,
                   help="width to re-encode at (default 1280); SAM is billed per frame, not per pixel")
    v.add_argument("--remuse", action="store_true", help="re-ask Muse even if muse.json is cached (costs again)")

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
    if args.mode == "video":
        run_video(args, budget)
    elif args.mode == "ego4d":
        if not args.dry_run and not args.videos:
            sys.exit("--videos is required unless --dry-run")
        run_ego4d(args, budget)
    else:
        run_episode(args, budget)


if __name__ == "__main__":
    main()
