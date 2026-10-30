"""Where each Ego4D clip sits inside its parent video.

Ego4D ships video two ways. `full_scale` files are named `<video_uid>.mp4` and hold the
whole video, so a narration timestamp addresses them directly. `clips` files are named
`<clip_uid>.mp4` and hold one window of a parent video, so the window's start has to be
subtracted from every timestamp before cutting. Only the annotation files know which
window a clip covers, and they spell it several ways; this module reads them all once and
caches the answer.
"""

from __future__ import annotations

import json
from pathlib import Path

# Clip-level window fields, in preference order. Deliberately excludes the action-level
# `parent_start_sec`/`parent_end_sec`, which describe one annotated action inside the
# clip rather than the clip itself, and would yield a window a few seconds wide.
WINDOW_FIELDS = [("video_start_sec", "video_end_sec"),
                 ("clip_parent_start_sec", "clip_parent_end_sec")]

# A real clip is minutes long; anything shorter came from a field we misread.
MIN_CLIP_SPAN_S = 30.0

# Parsed for clip windows. The narration files carry no clip windows and are huge.
SKIP_FILES = {"narration.json", "all_narrations_redacted.json", "fho_main.json",
              "fho_lta_taxonomy.json", "fho_main_taxonomy.json"}

ClipWindow = tuple[str, float, float]   # video_uid, start_s, end_s


def _harvest(node, out: dict[str, ClipWindow]) -> None:
    if isinstance(node, dict):
        cu, vu = node.get("clip_uid"), node.get("video_uid")
        if cu and vu and cu not in out:
            for s_key, e_key in WINDOW_FIELDS:
                s, e = node.get(s_key), node.get(e_key)
                if s is not None and e is not None and float(e) - float(s) >= MIN_CLIP_SPAN_S:
                    out[cu] = (vu, float(s), float(e))
                    break
        for v in node.values():
            _harvest(v, out)
    elif isinstance(node, list):
        for v in node:
            _harvest(v, out)


def build_clip_index(annotations_dir: Path, cache: Path | None = None,
                     verbose: bool = True) -> dict[str, ClipWindow]:
    """clip_uid -> (video_uid, start_s, end_s), read from `cache` when it exists."""
    if cache is not None and cache.exists():
        try:
            data = json.loads(cache.read_text())
            index = {k: (v[0], float(v[1]), float(v[2])) for k, v in data.items()}
            if verbose:
                print(f"clip index: {len(index)} clips (cached, {cache})")
            return index
        except (ValueError, OSError, IndexError, TypeError):
            pass   # unreadable cache is not worth a crash; rebuild it

    index: dict[str, ClipWindow] = {}
    if not annotations_dir.is_dir():
        return index
    for p in sorted(annotations_dir.glob("*.json")):
        if p.name in SKIP_FILES:
            continue
        try:
            _harvest(json.loads(p.read_text(encoding="utf-8")), index)
        except (ValueError, OSError, RecursionError):
            continue
    if verbose:
        print(f"clip index: {len(index)} clips read from {annotations_dir}")
    if cache is not None and index:
        cache.write_text(json.dumps({k: list(v) for k, v in index.items()}))
    return index
