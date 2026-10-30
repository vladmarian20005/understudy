"""Finding the file on disk that holds a given stretch of a given video."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from sam_pipeline.clip_index import ClipWindow


@dataclass
class Source:
    """A video file, and where its first frame sits on the parent video's clock."""
    path: Path
    offset: float    # video time of this file's frame 0; 0.0 for a full_scale video
    start: float     # window of the parent video this file covers
    end: float

    @property
    def is_clip(self) -> bool:
        return self.offset > 0.0 or math.isfinite(self.end)

    def local(self, video_t: float) -> float:
        """Translate a parent-video timestamp into this file's own timeline."""
        return video_t - self.offset


class VideoLibrary:
    """The mp4 files under one directory, addressable by (video_uid, time window).

    Handles both Ego4D layouts: `<video_uid>.mp4` full_scale videos, which cover
    everything at offset 0, and `<clip_uid>.mp4` clips, which cover one window of a
    parent video and need that window's start subtracted from every cut.
    """

    def __init__(self, video_dir: Path | None, clip_index: dict[str, ClipWindow] | None = None):
        self.video_dir = video_dir
        self.clip_index = clip_index or {}
        self._by_video: dict[str, list[Source]] | None = None

    def _index(self) -> dict[str, list[Source]]:
        if self._by_video is not None:
            return self._by_video
        by_video: dict[str, list[Source]] = {}
        if self.video_dir and self.video_dir.is_dir():
            for p in self.video_dir.rglob("*.mp4"):
                window = self.clip_index.get(p.stem)
                if window:
                    vu, s, e = window
                    by_video.setdefault(vu, []).append(Source(p, s, s, e))
                else:
                    # not a known clip, so treat the stem as a full_scale video_uid
                    by_video.setdefault(p.stem, []).append(Source(p, 0.0, 0.0, math.inf))
        # full_scale first: it can serve any window, and needs no offset
        for sources in by_video.values():
            sources.sort(key=lambda s: (s.offset, s.path.name))
        self._by_video = by_video
        return by_video

    def find(self, video_uid: str, start: float, end: float) -> Source | None:
        """The file wholly covering [start, end] of `video_uid`, if we have one."""
        for src in self._index().get(video_uid, []):
            if src.start <= start and end <= src.end:
                return src
        return None

    def counts(self) -> tuple[int, int]:
        """(full_scale videos, clips) found on disk."""
        full = clips = 0
        for sources in self._index().values():
            for s in sources:
                if math.isinf(s.end):
                    full += 1
                else:
                    clips += 1
        return full, clips
