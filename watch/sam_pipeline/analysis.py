"""From detections to the contract's objects, events and robot steps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sam_pipeline.config import (
    HOLD_BOX_PAD,
    HOLD_OVERLAP,
    HOLD_SMOOTH_FRAMES,
    PASS_MAX_UNHELD_FRAMES,
)
from sam_pipeline.detections import Det


def overlap(a, b, pad: float = HOLD_BOX_PAD) -> float:
    """Share of box a covered by box b grown by pad."""
    bw, bh = b[2] - b[0], b[3] - b[1]
    b = (b[0] - pad * bw, b[1] - pad * bh, b[2] + pad * bw, b[3] + pad * bh)
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1.0)
    return iw * ih / area


def assign_sides(dets: list[Det], frame_w: int, frame_h: int) -> dict[int, dict[str, Det]]:
    """Per frame, label up to two holders as "left" / "right".

    Ego4D: the wearer's own hands are the biggest ones in the lower part of the image.
    Robot: the two grippers. Left/right by horizontal position; a lone holder goes to
    whichever side it was nearest to last time.
    """
    by_frame: dict[int, list[Det]] = {}
    for d in dets:
        by_frame.setdefault(d.frame, []).append(d)
    last = {"left": (frame_w * 0.3, frame_h * 0.75), "right": (frame_w * 0.7, frame_h * 0.75)}
    out: dict[int, dict[str, Det]] = {}
    for f in sorted(by_frame):
        cands = [d for d in by_frame[f] if d.box[3] > frame_h * 0.35] or by_frame[f]
        cands = sorted(cands, key=lambda d: -d.area)[:2]
        sides: dict[str, Det] = {}
        if len(cands) == 2:
            l, r = sorted(cands, key=lambda d: d.centroid[0])
            sides = {"left": l, "right": r}
        elif cands:
            d = cands[0]
            dist = {s: (d.centroid[0] - p[0]) ** 2 + (d.centroid[1] - p[1]) ** 2 for s, p in last.items()}
            sides = {min(dist, key=dist.get): d}
        for s, d in sides.items():
            last[s] = d.centroid
        out[f] = sides
    return out


def pick_object_track(obj_dets: list[Det], holders: dict[int, dict[str, Det]]) -> list[Det]:
    """If SAM found several instances (three brushes on the table), keep the one that gets held."""
    tracks: dict[str, list[Det]] = {}
    for d in obj_dets:
        tracks.setdefault(d.obj_id, []).append(d)
    if not tracks:
        return []

    def held_frames(track):
        return sum(any(overlap(d.box, h.box) >= 0.15 for h in holders.get(d.frame, {}).values()) for d in track)

    best = max(tracks.values(), key=lambda tr: (held_frames(tr), len(tr)))
    return sorted(best, key=lambda d: d.frame)


def smooth(states: list[frozenset], k: int = HOLD_SMOOTH_FRAMES) -> list[frozenset]:
    out = []
    for i in range(len(states)):
        window = states[max(0, i - k // 2): i + k // 2 + 1]
        out.append(frozenset(s for s in ("left", "right") if sum(s in w for w in window) * 2 > len(window)))
    return out


def detect_events(obj_track: list[Det], holders: dict[int, dict[str, Det]], n_frames: int,
                  fps: float, frame_h: int) -> tuple[list[dict], list[frozenset]]:
    """grasp / release / pass / drop, in the contract's vision.events shape, plus who holds per frame."""
    pos = {d.frame: d for d in obj_track}
    raw = []
    for f in range(n_frames):
        d = pos.get(f)
        hs = holders.get(f, {})
        raw.append(frozenset(s for s, h in hs.items() if d and overlap(d.box, h.box) >= HOLD_OVERLAP))
    held = smooth(raw)
    t = lambda f: round(f / fps, 2)
    events: list[dict] = []
    for f in range(1, n_frames):
        prev, cur = held[f - 1], held[f]
        for s in cur - prev:
            events.append({"t": t(f), "kind": "grasp", "arm": s})
        for s in prev - cur:
            events.append({"t": t(f), "kind": "release", "arm": s})

    # pass: only A holds -> handover -> only B. The handover stretch is both hands on the
    # tool, nobody on it (it is occluded by the two hands meeting), or a mix of the two.
    passes = find_passes(held, n_frames)
    for start, end, giver, taker in passes:
        events.append({"t": t(start), "t_end": t(end), "kind": "pass", "from": giver, "to": taker})

    # drop: was held, then nobody holds it and it falls (or vanishes) within half a second.
    # An unheld stretch that is part of a handover is occlusion, not a drop.
    in_pass = set()
    for start, end, _, _ in passes:
        in_pass.update(range(start, end + 1))
    for f in range(1, n_frames):
        if held[f - 1] and not held[f] and f not in in_pass:
            before = pos.get(f - 1)
            later = [pos[x] for x in range(f, min(n_frames, f + int(fps / 2))) if x in pos]
            if before and (not later or later[-1].centroid[1] - before.centroid[1] > 0.1 * frame_h):
                events.append({"t": t(f), "kind": "drop", "arm": next(iter(held[f - 1]))})
    return sorted(events, key=lambda e: e["t"]), held


def find_passes(held: list[frozenset], n_frames: int,
                max_unheld: int = PASS_MAX_UNHELD_FRAMES) -> list[tuple[int, int, str, str]]:
    """Every (start, end, giver, taker) handover in a per-frame holder sequence.

    A handover is one hand alone on the tool, then a stretch where the tool is shared or
    briefly unseen, then the other hand alone on it. `max_unheld` caps how many frames of
    that stretch may credit nobody -- beyond it the tool was put down, not passed.
    """
    out: list[tuple[int, int, str, str]] = []
    f = 0
    while f < n_frames:
        if len(held[f]) != 1:
            f += 1
            continue
        giver = next(iter(held[f]))
        taker = "right" if giver == "left" else "left"
        g = f
        while g < n_frames and held[g] == frozenset({giver}):
            g += 1
        start = g
        while g < n_frames and (held[g] == frozenset({giver, taker}) or not held[g]):
            g += 1
        unheld = sum(1 for x in range(start, g) if not held[x])
        if g > start and g < n_frames and held[g] == frozenset({taker}) and unheld <= max_unheld:
            out.append((start, g, giver, taker))
            f = g
            continue
        f = max(g, f + 1)
    return out


def object_summary(name: str, track: list[Det], n_frames: int) -> dict:
    if not track:
        return {"name": name, "visible": 0.0, "start_px": None, "end_px": None}
    r = lambda p: [round(p[0]), round(p[1])]
    return {"name": name, "visible": round(len({d.frame for d in track}) / max(n_frames, 1), 2),
            "start_px": r(track[0].centroid), "end_px": r(track[-1].centroid)}


def steps_from_events(events: list[dict], giver: str | None, taker: str | None) -> list[dict]:
    """The primitive sequence the robot should run to reproduce this clip (contract op names).

    The robot always starts empty-handed, so all five steps are listed. "t" is when the
    video shows that step, or null when the clip doesn't show it (e.g. the person already
    held the tool when the clip starts).
    """
    passes = [e for e in events if e["kind"] == "pass"]
    if passes:
        giver, taker = passes[0]["from"], passes[0]["to"]
    if not (giver and taker):
        return []
    p = passes[0] if passes else None
    g = next((e for e in events if e["kind"] == "grasp" and e.get("arm") == giver
              and (not p or e["t"] <= p["t"])), None)
    rel = next((e for e in events if e["kind"] == "release" and e.get("arm") == taker
                and (not p or e["t"] >= p["t_end"])), None)
    t_grasp = g["t"] if g else None
    t_pass, t_pass_end = (p["t"], p["t_end"]) if p else (None, None)
    t_rel = rel["t"] if rel else None
    return [
        {"op": "reach", "arm": giver, "t": [None, t_grasp] if t_grasp is not None else None},
        {"op": "grasp", "arm": giver, "t": [t_grasp, t_pass] if t_grasp is not None else None},
        {"op": "handover", "giver": giver, "taker": taker, "mode": "air",
         "t": [t_pass, t_pass_end] if p else None},
        {"op": "move_to", "arm": taker, "t": [t_pass_end, t_rel] if p and rel else None},
        {"op": "release", "arm": taker, "t": [t_rel, t_rel] if rel else None},
    ]


@dataclass
class Analysis:
    objects: list[dict]
    events: list[dict]
    notes: list[str]
    held: list[frozenset]
    track: list[Det]
    holders: dict[int, dict[str, Det]]
    stats: dict
    object_prompt: str = ""   # the concept that actually found the tool


def segmentation_stats(obj_dets, track, holder_dets, holders, n_frames) -> dict:
    """Numbers that say how well SAM did on this clip, independent of the pass logic."""
    n = max(n_frames, 1)
    per_frame_hands: dict[int, int] = {}
    for d in holder_dets:
        per_frame_hands[d.frame] = per_frame_hands.get(d.frame, 0) + 1
    confs = [d.conf for d in track if d.conf is not None]
    jumps = [((b.centroid[0] - a.centroid[0]) ** 2 + (b.centroid[1] - a.centroid[1]) ** 2) ** 0.5
             for a, b in zip(track, track[1:]) if b.frame == a.frame + 1]
    return {
        "tool_frames": round(len({d.frame for d in track}) / n, 2),          # tool found, share of frames
        "tool_instances": len({d.obj_id for d in obj_dets}),                  # >1: other look-alikes or id switches
        "tool_confidence": round(float(np.mean(confs)), 2) if confs else None,
        "tool_jitter_px": round(float(np.median(jumps)), 1) if jumps else None,  # median move per frame
        "two_hands_frames": round(sum(len(s) == 2 for s in holders.values()) / n, 2),
        "hand_ids": len({d.obj_id for d in holder_dets}),                     # 2 is ideal
        "extra_hand_frames": round(sum(c > 2 for c in per_frame_hands.values()) / n, 2),
    }


def analyse(results, object_name: str | None, n_frames: int, fps: float, w: int, h: int,
            holder_name: str = "holder") -> Analysis:
    """object_name defaults to whichever concept actually found the tool."""
    holder_dets, holder_notes = results["holder"][0], results["holder"][1]
    obj_dets, obj_notes = results["object"][0], results["object"][1]
    used = results["object"][2] if len(results["object"]) > 2 else ""
    object_name = object_name or used
    holders = assign_sides(holder_dets, w, h)
    track = pick_object_track(obj_dets, holders)
    events, held = detect_events(track, holders, n_frames, fps, h)
    objects = [object_summary(object_name, track, n_frames)]
    for side in ("left", "right"):
        side_track = [hs[side] for f, hs in sorted(holders.items()) if side in hs]
        objects.append(object_summary(f"{side} {holder_name}", side_track, n_frames))
    stats = segmentation_stats(obj_dets, track, holder_dets, holders, n_frames)
    return Analysis(objects, events, holder_notes + obj_notes, held, track, holders, stats, used)
