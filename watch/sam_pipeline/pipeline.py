"""Running one video through SAM (shared by both modes)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from sam_pipeline.budget import Budget
from sam_pipeline.config import MAX_FRAMES_PER_REQUEST
from sam_pipeline.detections import Det, dump_masks, parse_sam
from sam_pipeline.sam_client import SamClient

# One role's SAM result: the detections, the parser's diagnostics, and the concept that
# actually produced them (which is not always the prompt we started from).
RoleResult = tuple[list[Det], list[str], str]


def _raw_path(outdir: Path, role: str) -> Path:
    return outdir / f"sam_{role}.txt"


def _used_path(outdir: Path, role: str) -> Path:
    return outdir / f"sam_{role}_prompt.txt"


def _empty_path(outdir: Path, role: str) -> Path:
    return outdir / f"sam_{role}_empty.json"


def _load_empty(outdir: Path, role: str) -> list[str]:
    """Concepts already paid for on an earlier run that SAM had nothing to say about."""
    p = _empty_path(outdir, role)
    if not p.exists():
        return []
    try:
        return [str(x) for x in json.loads(p.read_text())]
    except (ValueError, OSError):
        return []


def _record_empty(outdir: Path, role: str, prompt: str) -> None:
    seen = _load_empty(outdir, role)
    if prompt not in seen:
        seen.append(prompt)
        _empty_path(outdir, role).write_text(json.dumps(seen, indent=1))


def _cached(outdir: Path, role: str, default_prompt: str) -> tuple[str, str] | None:
    """The raw text of an earlier successful run, with the concept that produced it.

    A zero-length file is not a cache hit: SAM returns an empty stream when a concept
    finds nothing, and treating that as "already done" would pin the clip to a dead
    prompt forever.
    """
    raw = _raw_path(outdir, role)
    if not raw.exists() or raw.stat().st_size == 0:
        return None
    used = default_prompt
    if _used_path(outdir, role).exists():
        used = _used_path(outdir, role).read_text().strip() or default_prompt
    return raw.read_text(), used


def segment_video(client: SamClient | None, budget: Budget, video: Path, prompts: dict[str, str],
                  outdir: Path, fps: float, n_frames: int, w: int, h: int, want_rle: bool,
                  tag: str, fallbacks: dict[str, list[str]] | None = None,
                  max_attempts: int = 1, retry_empty: bool = False) -> dict[str, RoleResult] | None:
    """prompts: role -> SAM concept, e.g. {"holder": "hand", "object": "paintbrush"}.

    fallbacks maps a role to further concepts to try when the one before it comes back
    with no detections; at most `max_attempts` requests are made per role. Every attempt
    is a separate billed request, so the budget is re-checked before each one.
    """
    one = Budget.cost(n_frames, fps)
    if n_frames > MAX_FRAMES_PER_REQUEST:
        print(f"  skip {tag}: {n_frames} frames is over SAM's per-request limit")
        return None
    if not budget.can_spend(one * len(prompts)):
        print(f"  stop: {tag} needs {one * len(prompts):.0f}s, only {budget.left:.0f}s of budget left")
        return None

    results: dict[str, RoleResult] = {}
    file_id = None
    for role, prompt in prompts.items():
        chain = [prompt] + [p for p in (fallbacks or {}).get(role, []) if p != prompt]
        chain = chain[: max(1, max_attempts)]
        skip = set() if retry_empty else set(_load_empty(outdir, role))

        hit = _cached(outdir, role, prompt)
        if hit is not None:
            text, used = hit
            dets, notes = parse_sam(text, w, h, want_rle)
            results[role] = (dets, notes, used)
            if want_rle:
                dump_masks(outdir / f"masks_{role}.json", used, dets, w, h)
            continue

        dets: list[Det] = []
        notes: list[str] = []
        used = prompt
        for attempt, concept in enumerate(chain):
            if concept in skip:
                print(f"  skip '{concept}' on {tag}: already returned nothing on an earlier run")
                continue
            if not budget.can_spend(one):
                print(f"  stop: '{concept}' on {tag} needs {one:.0f}s, only {budget.left:.0f}s left")
                break
            if file_id is None:
                file_id = client.upload(video)
            label = "retry " if attempt else ""
            print(f"  SAM {label}'{concept}' on {tag} "
                  f"({one:.1f}s, budget used {budget.used:.0f}/{budget.limit_s:.0f}s)")
            t0 = time.time()
            text = client.segment(file_id, concept)
            budget.spend(one, f"{tag}:{concept}")
            print(f"    done in {time.time() - t0:.1f}s")
            if not text.strip():
                print(f"    '{concept}' found nothing")
                _record_empty(outdir, role, concept)
                continue
            dets, notes = parse_sam(text, w, h, want_rle)
            if not dets:
                print(f"    '{concept}' returned no usable detections")
                _record_empty(outdir, role, concept)
                continue
            used = concept
            _raw_path(outdir, role).write_text(text)
            _used_path(outdir, role).write_text(concept)
            if attempt:
                print(f"    '{concept}' worked ({len(dets)} detections)")
            break

        results[role] = (dets, notes, used)
        if want_rle:
            dump_masks(outdir / f"masks_{role}.json", used, dets, w, h)
    return results
