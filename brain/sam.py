"""SAM 3.1 on the overhead image: the objects a task names, found by name, placed on the table in cm.

robotd's colour detector only sees strongly coloured blobs, and not the arms' red and orange; SAM finds things by name
(a tennis ball, a cylinder, a cup) whatever their colour, including ones an arm half covers. Each name is one request
and all of them go at once: about 1.5 s for any number of names (31 Oct, inline JPEG, no upload). A mask becomes an
object shaped like robotd's detections: its outline projected onto the table through the calibrated camera, and its
height, which one camera can't see directly: a round thing is as tall as it is wide, and an upright one is as tall as
its top leans out from its base in the image (the camera looks down from one point, so the higher a point, the farther
from the camera's foot it lands when projected onto the table).
"""
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
from meta_sam_parser import CompletedOutcome, SegmentationMaskRecord, decode_mask_to_raster, image_segmentation_format

from harness.perception import TablePerception

URL = "https://api.meta.ai/v1/responses"
MODEL = "sam-3.1"
MIN_CONF = 0.5
ROUND = 0.8  # a mask filling this much of its enclosing circle is a ball
UPRIGHT_CM = 4.0  # an outline that leans out this much is a standing object, not a flat one
FLAT_H = 0.025  # m: the height of anything neither round nor standing, a guess
REACH = 0.50  # m: farther than this from both bases is off the table for us
FRONT_Y = 0.05  # m: nearer the bases' edge than this is the bases and their cables
ARM_SHARE = 0.9  # a mask this much inside the band round an arm is the arm (the band is 10 cm wide)
SAME_IOU = 0.5  # two names' masks overlapping this much are one object: the first name wins


def api_key():
    from .llm import load_env
    load_env()
    key = os.environ.get("MODEL_API_KEY") or os.environ.get("MUSE_API_KEY")
    if not key:
        raise SystemExit("no MODEL_API_KEY: put it in .env")
    return key


def ask(img_url, name, key, timeout=30):
    """One name on one image: [(confidence, full-frame mask)]."""
    body = {"model": MODEL, "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": name}, {"type": "input_image", "image_url": img_url}]}],
        "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"}, "stream": True}
    r = requests.post(URL, json=body, stream=True, timeout=timeout,
                      headers={"Authorization": f"Bearer {key}", "Accept": "text/event-stream"})
    if r.status_code >= 400:
        raise RuntimeError(f"SAM: HTTP {r.status_code}: {r.text[:200]}")
    parts = []
    for raw in r.iter_lines(decode_unicode=True):
        if raw and raw.startswith("data:"):
            d = raw[5:].strip()
            if d == "[DONE]":
                break
            e = json.loads(d)
            if e.get("type") == "response.output_text.delta":
                parts.append(e.get("delta", ""))
            elif e.get("type") in ("response.failed", "error"):
                raise RuntimeError(f"SAM failed: {d[:200]}")
    text = "".join(parts)
    if not text:
        return []
    p = image_segmentation_format().create_parser()
    p.push(text, emit=False)
    out = []
    for rec in p.finish(CompletedOutcome()).result.records:
        if not isinstance(rec, SegmentationMaskRecord) or (rec.confidence or 0) < MIN_CONF:
            continue
        b, m = rec.bounds, rec.mask
        raster = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width)
        bw, bh = int(b.right - b.left), int(b.bottom - b.top)
        if bw <= 0 or bh <= 0:
            continue
        # the raster can be smaller than its box: stretch it to the box
        ys = (np.arange(bh) * m.height // bh).clip(0, m.height - 1)
        xs = (np.arange(bw) * m.width // bw).clip(0, m.width - 1)
        out.append((float(rec.confidence or 0), (int(b.left), int(b.top)), raster[np.ix_(ys, xs)]))
    return out


def segment(img, names):
    """Every name on the image at once: {name: [(confidence, mask)]}, masks full-frame uint8."""
    names = list(dict.fromkeys(n.strip() for n in names if n and n.strip()))
    if not names:
        return {}
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    key = api_key()
    h, w = img.shape[:2]

    def one(name):
        found = []
        for conf, (x0, y0), local in ask(url, name, key):
            full = np.zeros((h, w), np.uint8)
            lh, lw = min(local.shape[0], h - y0), min(local.shape[1], w - x0)
            full[y0:y0 + lh, x0:x0 + lw] = local[:lh, :lw]
            found.append((conf, full))
        return name, found

    with ThreadPoolExecutor(len(names)) as ex:
        return dict(ex.map(one, names))


def shape(scene, contour, touches_edge):
    """A mask outline as an object on the table: (centre m, z of its middle m, [L, W, H] m, yaw deg, kind)."""
    tp = TablePerception(scene)
    _, r_px = cv2.minEnclosingCircle(contour)
    if cv2.contourArea(contour) > ROUND * np.pi * r_px ** 2:  # a ball: as tall as it is wide, seen at its widest
        centre, L, W, _ = tp.footprint(contour, 0.03)
        h = (L + W) / 2
        centre, L, W, _ = tp.footprint(contour, h / 2)
        return centre, h / 2, [L, W, h], 0.0, "round"
    cam = scene.camera.cam_to_world[:3, 3]
    foot, cz = cam[:2], cam[2]
    pts = scene.camera.to_table(contour.reshape(-1, 2).astype(float), z=0.0)[:, :2]
    u = pts.mean(axis=0) - foot
    u /= np.linalg.norm(u)
    v = np.array([-u[1], u[0]])
    s, t = (pts - foot) @ u, (pts - foot) @ v
    near_s, far_s = float(s.min()), float(s.max())
    near = s < near_s + 0.3 * (far_s - near_s)
    width = float(t[near].max() - t[near].min())
    h = cz * (1 - (near_s + width) / far_s)
    if h * 100 > UPRIGHT_CM:  # standing: the base is the near end, as wide as the outline is there
        centre = foot + u * (near_s + width / 2) + v * float(t[near].mean())
        return centre, h / 2, [width, width, h], 0.0, "upright, at least this tall" if touches_edge else "upright"
    centre, L, W, long_edge = tp.footprint(contour, FLAT_H / 2)
    yaw = float(np.degrees(np.arctan2(long_edge[1], long_edge[0]))) % 180
    return centre, FLAT_H / 2, [L, W, FLAT_H], yaw, "flat"


def objects(scene, img, names, q, held_px=()):
    """The named objects on the image, shaped like robotd's detections plus found_as (the name SAM was given) and
    shape. q: both arms' joints (masks mostly on an arm are dropped). held_px: pixels of objects a gripper holds
    (those aren't on the table)."""
    arm = TablePerception(scene).arm_mask(img.shape, q) > 0
    kept = []
    for name, found in segment(img, names).items():
        for conf, mask in found:
            m = mask > 0
            if not m.any() or (m & arm).sum() > ARM_SHARE * m.sum():
                continue
            if any(m[min(max(int(py), 0), m.shape[0] - 1), min(max(int(px), 0), m.shape[1] - 1)]
                   for px, py in held_px):
                continue
            if any((m & k["mask"]).sum() > SAME_IOU * min(m.sum(), k["mask"].sum()) for k in kept):
                continue
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < 60:
                continue
            x, y, w, h = cv2.boundingRect(c)
            edge = x <= 1 or y <= 1 or x + w >= img.shape[1] - 1 or y + h >= img.shape[0] - 1
            centre, z, size, yaw, kind = shape(scene, c, edge)
            if centre[1] < FRONT_Y or min(np.linalg.norm(centre - scene.base_xy(s)) for s in scene.arms) > REACH:
                continue
            mo = cv2.moments(c)
            kept.append(dict(found_as=name, conf=round(conf, 2), mask=m, center_cm=[round(float(v) * 100, 1)
                                                                                     for v in centre],
                             z_cm=round(z * 100, 1), size_cm=[round(float(v) * 100, 1) for v in size],
                             yaw=round(yaw, 1), shape=kind, px=[int(mo["m10"] / mo["m00"]), int(mo["m01"] / mo["m00"])]))
    for k in kept:
        del k["mask"]
    return kept
