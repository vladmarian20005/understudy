"""Mask codecs and the decode cache.

Importing this module installs a memoising wrapper around the parser's mask decoder: the
parser already decodes (and re-checks) every mask while it validates the stream, so
remembering those rasters keeps `parse_sam` from paying for the slow pure-Python decode
twice.
"""

from __future__ import annotations

import numpy as np
from meta_sam_parser import decode_mask_to_raster

import meta_sam_parser._segmentation as _sam_seg

_RASTERS: dict[tuple, bytes] = {}


def cached_decode(mask):
    key = (mask.encoding, mask.width, mask.height, mask.payload)
    if key not in _RASTERS:
        _RASTERS[key] = decode_mask_to_raster(mask)
    return _RASTERS[key]


def clear_raster_cache() -> None:
    _RASTERS.clear()


_sam_seg.decode_mask_to_raster = cached_decode


def coco_rle(mask: np.ndarray) -> dict:
    """Uncompressed COCO RLE (column-major, starts with a run of zeros)."""
    flat = mask.flatten(order="F")
    changes = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.concatenate([[0], changes, [flat.size]])).tolist()
    if flat.size and flat[0] == 1:
        runs = [0] + runs
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": runs}


def rle_to_mask(rle: dict) -> np.ndarray:
    """Inverse of `coco_rle`, as a boolean raster."""
    h, w = rle["size"]
    counts = rle["counts"]
    values = np.zeros(len(counts), dtype=np.uint8)
    values[1::2] = 1
    flat = np.repeat(values, counts)
    return flat.reshape((w, h)).T.astype(bool)
