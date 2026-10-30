"""Mode: ego4d (crafting clips -> one score.json per clip)."""

from __future__ import annotations

import json

from sam_pipeline.analysis import analyse, steps_from_events
from sam_pipeline.budget import Budget
from sam_pipeline.clip_index import build_clip_index
from sam_pipeline.clips import select_clips
from sam_pipeline.config import MODEL, object_fallbacks
from sam_pipeline.credentials import api_key
from sam_pipeline.pipeline import segment_video
from sam_pipeline.preview import render_preview
from sam_pipeline.sam_client import SamClient
from sam_pipeline.sources import VideoLibrary
from sam_pipeline.video import cut_clip, probe


def handover_summary(p: dict | None, track, fps: float) -> dict | None:
    """The score's handover block (watch/SCORE.md): when both start holding, when the giver lets go, how long both
    hold, and where the object was meanwhile (its mask centroid nearest the middle of the pass)."""
    if p is None:
        return None
    mid = round((p["t"] + p["t_end"]) / 2 * fps)
    near = min(track, key=lambda d: abs(d.frame - mid), default=None)
    return {"t": p["t"], "t_end": p["t_end"], "both_s": round(p["t_end"] - p["t"], 2),
            "at_px": [round(near.centroid[0]), round(near.centroid[1])] if near is not None else None,
            "at_cm": None}


def run_ego4d(args, budget: Budget) -> None:
    clips = select_clips(args.metadata, args.narrations, args.before, args.after, args.any_object)
    print(f"{len(clips)} hand-to-hand passes found in crafting videos")

    # clips/ files are named by clip_uid and cover one window of their parent video, so
    # they only become addressable once we know which window that is
    clip_index = {}
    if args.videos:
        ann = args.clip_annotations or args.narrations.parent
        clip_index = build_clip_index(ann, cache=args.out / "clip_index.json")
    library = VideoLibrary(args.videos, clip_index)
    if args.videos:
        full, cl = library.counts()
        print(f"video files: {cl} clip(s) located inside their parent video, "
              f"{full} unrecognised (addressed as whole videos)")

    picked: list = []
    sources = {}
    for c in clips:
        if len(picked) >= args.max_clips:
            break
        if args.videos:
            src = library.find(c.video_uid, c.start, c.end)
            if src is None:
                continue
            sources[c.clip_id] = src
        picked.append(c)
    planned = sum(Budget.cost(int(c.duration * args.fps), args.fps) * 2 for c in picked)
    print(f"{len(picked)} clips with a local video file, ~{planned / 60:.1f} min of SAM video "
          f"(budget left {budget.left / 60:.1f} of {budget.limit_s / 60:.1f} min)\n")
    for c in picked:
        where = ""
        src = sources.get(c.clip_id)
        if src is not None and src.offset:
            where = f"  [{src.path.stem[:8]} +{src.local(c.start):.1f}s]"
        print(f"  {c.clip_id}  {c.duration:4.1f}s  {c.prompt:<11} {c.giver or '?'}->{c.taker or '?'}  {c.narration[:70]}{where}")
    if args.dry_run:
        uids = sorted({c.video_uid for c in picked})
        (args.out / "needed_video_uids.txt").write_text("\n".join(uids) + "\n")
        print(f"\nDry run: nothing cut or sent. The {len(uids)} videos these clips need are listed in "
              f"{args.out / 'needed_video_uids.txt'}; download just those with\n"
              f"  ego4d --output_directory ~/ego4d --datasets full_scale --video_uid_file "
              f"{args.out / 'needed_video_uids.txt'}")
        return

    client = SamClient(api_key(), args.base_url)
    out = args.out
    index = []
    for c in picked:
        cdir = out / "scores" / c.clip_id
        cdir.mkdir(parents=True, exist_ok=True)
        clip_path = cdir / "clip.mp4"
        src = sources[c.clip_id]
        if not clip_path.exists():
            # src.local() puts the narration timestamp on the source file's own clock,
            # which differs from video time whenever the source is a clip
            cut_clip(src.path, clip_path, src.local(c.start), c.duration, args.width, args.fps)
        w, h, fps, n = probe(clip_path)
        print(f"\n{c.clip_id}: {c.narration}")
        results = segment_video(client, budget, clip_path, {"holder": "hand", "object": c.prompt},
                                cdir, fps, n, w, h, not args.no_rle, c.clip_id,
                                fallbacks={"object": object_fallbacks(c.prompt)},
                                max_attempts=args.object_attempts, retry_empty=args.retry_empty)
        if results is None:
            break
        a = analyse(results, None, n, fps, w, h, holder_name="hand")
        objects, events, notes = a.objects, a.events, a.notes
        if not args.no_preview:
            render_preview(clip_path, cdir / "preview.mp4", results, a, w, h, fps)
        passes = [e for e in events if e["kind"] == "pass"]
        giver = passes[0]["from"] if passes else c.giver
        taker = passes[0]["to"] if passes else c.taker
        score = {
            "version": 1,
            "kind": "score",
            "task": f"Pass the {c.prompt} to the other arm",
            "object": {"name": c.prompt, "prompt": a.object_prompt or c.prompt},
            "source": {"kind": "ego4d", "video_uid": c.video_uid, "start_s": c.start, "end_s": c.end,
                       "narration": c.narration, "narration_t": round(c.narration_t - c.start, 2),
                       "file": src.path.name, "file_offset_s": round(src.local(c.start), 2)},
            "frame": "ego",  # a head-mounted camera: pixels don't map to the table (watch/SCORE.md)
            "fps": fps, "frames": n, "size_px": [w, h],
            "giver": giver, "taker": taker,
            "handover": handover_summary(passes[0] if passes else None, a.track, fps),
            "target_cm": None,
            "vision": {"model": MODEL, "objects": objects, "events": events},
            "steps": steps_from_events(events, giver, taker),
            "quality": {
                "pass_seen": bool(passes),
                "agrees_with_narration": bool(passes) and (c.giver is None or c.giver == giver),
                "object_visible": objects[0]["visible"],
                "object_prompt": a.object_prompt,          # concept that actually found the tool
                "object_prompt_fallback": a.object_prompt != c.prompt,
                "segmentation": a.stats,
                "notes": notes[:20],
            },
            "held": "".join({0: "-", 1: "?"}.get(len(s), "B") if len(s) != 1 else ("L" if "left" in s else "R")
                            for s in a.held),
            "files": {"clip": "clip.mp4", "sam_raw": ["sam_holder.txt", "sam_object.txt"],
                      "masks": [] if args.no_rle else ["masks_holder.json", "masks_object.json"]},
        }
        (cdir / "score.json").write_text(json.dumps(score, indent=2))
        index.append({"clip_id": c.clip_id, "pass_seen": bool(passes), "object": c.prompt,
                      "giver": giver, "taker": taker, "visible": objects[0]["visible"], **a.stats})
        s = a.stats
        print(f"  pass seen: {bool(passes)}  {giver}->{taker}")
        print(f"  SAM: tool in {s['tool_frames']:.0%} of frames (conf {s['tool_confidence']}, "
              f"{s['tool_instances']} instance(s), jitter {s['tool_jitter_px']} px), "
              f"both hands in {s['two_hands_frames']:.0%}, {s['hand_ids']} hand id(s), "
              f"extra hands in {s['extra_hand_frames']:.0%}")
        if not args.no_preview:
            print(f"  preview: {cdir / 'preview.mp4'}")

    (out / "index.json").write_text(json.dumps(sorted(index, key=lambda i: (not i["pass_seen"], -i["visible"])), indent=2))
    print(f"\nWrote {len(index)} scores to {out / 'scores'}; best first in {out / 'index.json'}")
    print(f"Budget used: {budget.used / 60:.1f} of {budget.limit_s / 60:.1f} min")
