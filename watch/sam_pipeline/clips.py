"""Selecting clips from Ego4D metadata + narrations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from sam_pipeline.config import ARTICLES, BAD_OBJECTS, GOOD_OBJECTS, HAND_TO_HAND, SCENARIOS


@dataclass
class Clip:
    video_uid: str
    narration: str
    narration_t: float
    start: float
    end: float
    obj_phrase: str
    prompt: str          # SAM concept prompt for the tool
    giver: str | None    # "left" / "right" / None if the narration doesn't say
    taker: str | None
    score: float         # ranking

    @property
    def clip_id(self) -> str:
        return f"{self.video_uid[:8]}_{self.narration_t:08.2f}".replace(".", "_")

    @property
    def duration(self) -> float:
        return self.end - self.start


def iter_narrations(entry):
    """Yield (t, text) from one video's narration entry, whatever pass layout it has."""
    if isinstance(entry, dict):
        for key, value in entry.items():
            if key == "narrations" and isinstance(value, list):
                for n in value:
                    t = n.get("timestamp_sec")
                    text = n.get("narration_text") or n.get("text")
                    if t is not None and text:
                        yield float(t), text
            else:
                yield from iter_narrations(value)
    elif isinstance(entry, list):
        for value in entry:
            yield from iter_narrations(value)


def clean_object(phrase: str) -> str:
    phrase = re.sub(r"#\w+|\bC\b", " ", phrase)
    phrase = ARTICLES.sub(" ", phrase)
    phrase = re.sub(r"\b(?:with|in|on|from|to|into|and|both|hands?|left|right|other)\b.*$", "", phrase.strip(), flags=re.I)
    return re.sub(r"\s+", " ", phrase).strip(" ,.").lower()


def match_pass(text: str):
    for pattern in HAND_TO_HAND:
        m = pattern.search(text)
        if m:
            g = m.groupdict()
            obj = clean_object(g.get("obj") or "")
            src, dst = (g.get("src") or "").lower() or None, (g.get("dst") or "").lower() or None
            if dst == "other":
                dst = None
            if src is None and dst in ("left", "right"):
                src = "right" if dst == "left" else "left"
            if dst is None and src in ("left", "right"):
                dst = "right" if src == "left" else "left"
            return obj, src, dst
    return None


def object_prompt(obj: str) -> tuple[str | None, float]:
    """SAM prompt for the object and a fitness score for two small grippers."""
    if not obj:
        return None, 0.0
    if any(re.search(rf"\b{b}", obj) for b in BAD_OBJECTS):
        return None, 0.0
    for key in sorted(GOOD_OBJECTS, key=len, reverse=True):
        if re.search(rf"\b{key}\b", obj):
            return GOOD_OBJECTS[key], 1.0
    words = obj.split()
    return (" ".join(words[-2:]) if words else None), 0.3   # unknown object: keep, rank low


def select_clips(metadata: Path, narrations: Path, before: float, after: float,
                 allow_unknown: bool) -> list[Clip]:
    videos = {v["video_uid"]: v for v in json.loads(metadata.read_text())["videos"]}
    wanted = {uid: v for uid, v in videos.items() if set(v.get("scenarios") or []) & set(SCENARIOS)}
    narr = json.loads(narrations.read_text())
    if isinstance(narr, dict) and "videos" in narr and isinstance(narr["videos"], list):
        narr = {v.get("video_uid"): v for v in narr["videos"]}

    clips: list[Clip] = []
    for uid, meta in wanted.items():
        duration = float(meta.get("duration_sec") or 0)
        last_t = -1e9
        for t, text in sorted(iter_narrations(narr.get(uid, {}))):
            m = match_pass(text)
            if not m:
                continue
            obj, src, dst = m
            prompt, fit = object_prompt(obj)
            if prompt is None or (fit < 1.0 and not allow_unknown):
                continue
            if t - last_t < before + after:      # overlaps the previous clip
                continue
            start, end = max(0.0, t - before), min(duration or t + after, t + after)
            if end - start < 2.0:
                continue
            last_t = t
            score = fit + (0.3 if src and dst else 0) + 0.2 * bool(meta.get("split_fho"))
            clips.append(Clip(uid, text.strip(), t, round(start, 2), round(end, 2), obj, prompt,
                              src, dst, round(score, 2)))
    clips.sort(key=lambda c: (-c.score, c.video_uid, c.narration_t))
    return clips


def find_video(video_dir: Path, uid: str, cache: dict) -> Path | None:
    if not cache:
        for p in video_dir.rglob("*.mp4"):
            cache[p.stem] = p
    return cache.get(uid)
