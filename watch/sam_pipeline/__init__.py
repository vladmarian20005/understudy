"""SAM 3.1 segmentation pipeline for "Watch, Delegate, Distill".

The package is split by concern:

    config       tunables and the clip-selection vocabulary
    budget       the shared request ledger
    clips        picking hand-to-hand passes out of Ego4D narrations
    video        ffmpeg/ffprobe wrappers
    sam_client   Meta Model API transport
    masks        COCO RLE encode/decode and the mask sidecar dump
    parsing      SAM's text output -> per-frame detections
    analysis     detections -> contract objects, events and steps
    pipeline     one video through SAM (shared by both modes)
    preview      the annotated preview.mp4
    ego4d        mode: Ego4D clips -> score.json
    episode      mode: robot episode -> vision.json
    cli          argument parsing and entry point
"""

from __future__ import annotations

__all__ = ["main"]



def main() -> None:
    """Run the command line interface (lazy import to keep `-c "import sam_pipeline"` cheap)."""
    from sam_pipeline.cli import main as _main

    _main()
