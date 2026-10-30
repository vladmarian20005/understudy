"""Parsing SAM output into per-frame detections, and writing them out."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from meta_sam_parser import (
    CompletedOutcome,
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    video_segmentation_format,
)

from sam_pipeline.masks import cached_decode, clear_raster_cache, coco_rle


@dataclass
class Det:
    frame: int
    obj_id: str
    box: tuple[float, float, float, float]   # left, top, right, bottom (exclusive), px
    conf: float | None
    centroid: tuple[float, float]
    area: float
    rle: dict | None = None


def parse_sam(text: str, frame_w: int, frame_h: int, want_rle: bool) -> tuple[list[Det], list[str]]:
    clear_raster_cache()
    parser = video_segmentation_format().create_parser()
    parser.push(text, emit=False)
    result = parser.finish(CompletedOutcome()).result
    notes = [f"{d.severity}: {d.code} {d.message}" for d in result.diagnostics]
    boxes = {(r.frame.frame_index, r.object_id): r for r in result.records
             if isinstance(r, SegmentationBoxRecord) and r.frame}
    dets = []
    for r in result.records:
        if not isinstance(r, SegmentationMaskRecord) or r.frame is None:
            continue
        b = r.bounds
        raster = np.frombuffer(cached_decode(r.mask), dtype=np.uint8).reshape(r.mask.height, r.mask.width)
        bw, bh = int(b.right - b.left), int(b.bottom - b.top)
        # the mask raster can be smaller than its box; stretch it to the box (nearest)
        ys = (np.arange(bh) * r.mask.height // max(bh, 1)).clip(0, r.mask.height - 1)
        xs = (np.arange(bw) * r.mask.width // max(bw, 1)).clip(0, r.mask.width - 1)
        local = raster[np.ix_(ys, xs)]
        py, px = np.nonzero(local)
        if len(px):
            centroid = (float(b.left + px.mean()), float(b.top + py.mean()))
        else:
            centroid = ((b.left + b.right) / 2, (b.top + b.bottom) / 2)
        box_rec = boxes.get((r.frame.frame_index, r.object_id))
        conf = r.confidence if r.confidence is not None else (box_rec.confidence if box_rec else None)
        det = Det(r.frame.frame_index, r.object_id, (b.left, b.top, b.right, b.bottom), conf,
                  centroid, float(len(px)))
        if want_rle:
            full = np.zeros((frame_h, frame_w), dtype=np.uint8)
            t, l = int(b.top), int(b.left)
            full[t:t + bh, l:l + bw] = local[: frame_h - t, : frame_w - l]
            det.rle = coco_rle(full)
        dets.append(det)
    clear_raster_cache()
    return dets, notes


def dump_masks(path: Path, prompt: str, dets: list[Det], w: int, h: int) -> None:
    frames: dict[int, list] = {}
    for d in dets:
        frames.setdefault(d.frame, []).append({
            "id": d.obj_id, "box_px": [round(v, 1) for v in d.box],
            "confidence": d.conf, "centroid_px": [round(d.centroid[0], 1), round(d.centroid[1], 1)],
            "area_px": int(d.area), "rle": d.rle})
    path.write_text(json.dumps({"prompt": prompt, "size_px": [w, h], "mask_format": "coco_rle_uncompressed",
                                "frames": {str(k): v for k, v in sorted(frames.items())}}))
