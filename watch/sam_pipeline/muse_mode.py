"""Mode: video (any video -> Muse Spark reads it -> SAM segments what it named -> score.json).

The ego4d mode gets its SAM prompt from the narration. A plain video has none, so Muse
Spark supplies both halves: an egocentric run of the actions, and the objects the hands
touched. Those objects become SAM's concept prompts, and the result is written in the same
score format (watch/SCORE.md) as every other source.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sam_pipeline.analysis import analyse, object_summary, pick_object_track, steps_from_events
from sam_pipeline.budget import Budget
from sam_pipeline.config import MODEL, object_fallbacks
from sam_pipeline.credentials import api_key
from sam_pipeline.ego4d import handover_summary
from sam_pipeline.muse_client import PROMPT, MuseClient, MuseReading, parse_reading
from sam_pipeline.pipeline import segment_video
from sam_pipeline.preview import render_preview
from sam_pipeline.sam_client import SamClient
from sam_pipeline.video import cut_clip, probe


def display_width(path: Path, coded_w: int, coded_h: int) -> int:
    """Width after ffmpeg applies the file's rotation, which `probe` does not see.

    A phone clip is often stored landscape with a 90-degree rotate matrix; ffmpeg rotates
    on decode, so scaling to the coded width would upscale a portrait frame.
    """
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream_side_data=rotation:stream_tags=rotate",
                          "-of", "json", str(path)], capture_output=True, text=True).stdout
    try:
        d = json.loads(out or "{}")
    except ValueError:
        return coded_w
    rot = 0
    for sd in d.get("side_data_list") or d.get("streams", [{}])[0].get("side_data_list", []) or []:
        rot = int(float(sd.get("rotation", 0) or 0))
    return coded_h if rot % 180 else coded_w


def read_with_muse(args, budget: Budget, video: Path, outdir: Path, n: int, fps: float) -> MuseReading | None:
    """Ask Muse what happens in the clip. Cached: the reply is the expensive part, not the parse."""
    cache = outdir / "muse.json"
    if cache.exists() and not args.remuse:
        d = json.loads(cache.read_text())
        r = parse_reading(d["raw"]) if d.get("raw") else MuseReading(d.get("summary", ""), d.get("objects", []))
        print(f"  Muse (cached): {len(r.objects)} object(s)")
        return r
    cost = Budget.cost(n, fps)          # Muse watches the same seconds of video SAM would
    if not budget.can_spend(cost):
        print(f"  stop: Muse needs {cost:.0f}s, only {budget.left:.0f}s of budget left")
        return None
    client = MuseClient(api_key(), args.base_url, args.muse_model)
    print(f"  Muse '{args.muse_model}' on {video.name} "
          f"({cost:.1f}s, budget used {budget.used:.0f}/{budget.limit_s:.0f}s)")
    reading = client.watch(client.upload(video), args.muse_prompt or PROMPT)
    budget.spend(cost, f"{video.stem}:muse")
    print(f"    done in {reading.seconds:.1f}s")
    cache.write_text(json.dumps({"model": args.muse_model, "summary": reading.summary,
                                 "objects": reading.objects, "raw": reading.raw}, indent=2))
    return reading


def run_video(args, budget: Budget) -> None:
    video = args.video
    w, h, fps, n = probe(video)
    outdir = args.out / "videos" / video.stem
    outdir.mkdir(parents=True, exist_ok=True)

    # SAM wants H.264, and a small frame loses small tools (watch/NOTES.md), so re-encode once
    clip = outdir / "clip.mp4"
    if not clip.exists():
        cut_clip(video, clip, 0, n / fps, min(args.width, display_width(video, w, h)), fps)
    w, h, fps, n = probe(clip)
    print(f"{video.name}: {n} frames, {fps:.1f} fps, {w}x{h}")

    reading = read_with_muse(args, budget, clip, outdir, n, fps)
    if reading is None:
        sys.exit(1)
    print(f"\n  summary: {reading.summary}\n  objects: {', '.join(reading.objects) or '(none)'}\n")
    if not reading.objects:
        sys.exit("Muse named no objects to segment; re-run with --remuse or give --object yourself")

    wanted = [args.object] if args.object else reading.objects[: args.max_objects]
    prompts = {"holder": args.holder_prompt}
    roles = {}
    for i, name in enumerate(wanted):
        role = "object" if i == 0 else f"object{i + 1}"
        prompts[role] = name
        roles[role] = name

    client = SamClient(api_key(), args.base_url)
    results = segment_video(client, budget, clip, prompts, outdir, fps, n, w, h, not args.no_rle, video.stem,
                            fallbacks={r: object_fallbacks(nm) for r, nm in roles.items()},
                            max_attempts=args.object_attempts, retry_empty=args.retry_empty)
    if results is None:
        sys.exit(1)

    a = analyse(results, None, n, fps, w, h, holder_name="hand")
    events, notes = a.events, a.notes
    passes = [e for e in events if e["kind"] == "pass"]
    giver = passes[0]["from"] if passes else None
    taker = passes[0]["to"] if passes else None

    # a.objects is [primary, left hand, right hand]; the other objects Muse named go between
    extra = []
    for role, name in roles.items():
        if role == "object":
            continue
        dets = results[role][0]
        track = pick_object_track(dets, a.holders)
        extra.append(object_summary(results[role][2] or name, track, n))
    objects = [a.objects[0]] + extra + a.objects[1:]

    primary = roles.get("object", "")
    task = (f"Pass the {a.object_prompt or primary} to the other arm" if passes
            else (reading.summary[:200] or f"Handle the {primary}"))
    score = {
        "version": 1,
        "kind": "score",
        "task": task,
        "object": {"name": primary, "prompt": a.object_prompt or primary},
        "source": {"kind": "video", "file": video.name,
                   "muse": {"model": args.muse_model, "summary": reading.summary,
                            "objects": reading.objects, "segmented": wanted}},
        "frame": "ego",      # a hand-held or head-mounted camera: pixels don't map to the table
        "fps": fps, "frames": n, "size_px": [w, h],
        "giver": giver, "taker": taker,
        "handover": handover_summary(passes[0] if passes else None, a.track, fps),
        "target_cm": None,
        "vision": {"model": MODEL, "objects": objects, "events": events},
        "steps": steps_from_events(events, giver, taker),
        "quality": {
            "pass_seen": bool(passes),
            "agrees_with_narration": None,        # no narration: Muse's summary is not ground truth
            "object_visible": objects[0]["visible"],
            "object_prompt": a.object_prompt,
            "object_prompt_fallback": a.object_prompt != primary,
            "segmentation": a.stats,
            "notes": notes[:20],
        },
        "held": "".join({0: "-", 1: "?"}.get(len(s), "B") if len(s) != 1 else ("L" if "left" in s else "R")
                        for s in a.held),
        "files": {"clip": "clip.mp4", "muse": "muse.json",
                  "sam_raw": [f"sam_{r}.txt" for r in prompts],
                  "masks": [] if args.no_rle else [f"masks_{r}.json" for r in prompts]},
    }
    (outdir / "score.json").write_text(json.dumps(score, indent=2))
    if not args.no_preview:
        render_preview(clip, outdir / "preview.mp4", results, a, w, h, fps)
        print(f"  preview: {outdir / 'preview.mp4'}")

    s = a.stats
    print(f"\n  pass seen: {bool(passes)}  {giver or '?'}->{taker or '?'}")
    for o in objects:
        print(f"    {o['name']:<18} visible {o['visible']:.0%}")
    print(f"  SAM: tool in {s['tool_frames']:.0%} of frames (conf {s['tool_confidence']}, "
          f"{s['tool_instances']} instance(s)), both hands in {s['two_hands_frames']:.0%}, "
          f"{s['hand_ids']} hand id(s)")
    print(f"  wrote {outdir / 'score.json'}")
    print(f"Budget used: {budget.used / 60:.1f} of {budget.limit_s / 60:.1f} min")
