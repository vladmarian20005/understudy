"""The watch pipeline: a video of a person's hands through Muse Spark and SAM 3.1 into a score.

    config       tunables
    budget       the shared request ledger
    video        ffmpeg/ffprobe wrappers
    muse_client  Muse Spark through Meta's Model API: what the hands did, and what they touched
    sam_client   SAM 3.1 through Meta's Model API
    masks        COCO RLE encode/decode and the mask sidecar dump
    detections   SAM's text output -> per-frame detections
    analysis     detections -> the score's objects, events and handover
    pipeline     one video through SAM
    preview      the annotated preview.mp4
    muse_mode    a video -> score.json
    cli          argument parsing and entry point
"""

from __future__ import annotations

__all__ = ["main"]



def main() -> None:
    """Run the command line interface (lazy import to keep `-c "import sam_pipeline"` cheap)."""
    from sam_pipeline.cli import main as _main

    _main()
