"""Muse Spark through Meta's Model API: watch a video, say what the hands did.

SAM needs to be told what to segment. For Ego4D the narration supplies that word; for a
video with no narration -- a rig recording, a phone clip -- nothing does. Muse Spark
watches the clip and answers both halves at once: the egocentric run of actions, and the
objects those actions touched, which become SAM's concept prompts.

Same transport as `SamClient`: upload to /files, then stream /responses.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

DEFAULT_MODEL = os.environ.get("MUSE_MODEL", "muse-spark-1.3")

# Two labelled sections, so the reply parses without a JSON mode we may not have.
PROMPT = """You are watching a short first-person (egocentric) video. The camera is on the person's head, so their own hands enter from the bottom of the frame.

Answer in exactly two sections, nothing else:

SUMMARY: one paragraph describing, in order, every action the hands perform. Name which hand does what, in the present tense, as a run-on list. For example: "right hand picks up the ball box, left hand picks up ball 1, puts it in the ball box, picks up ball 2, puts it in the ball box, right hand sets the box down".

OBJECTS: the physical objects the hands interact with, tab-separated, most important first. Use the plain everyday noun for each, singular, lowercase, no articles and no counts, because these are fed to a segmentation model as concept prompts. Do not list hands, body parts, the background, or anything untouched. For example: "ball box<TAB>tube<TAB>ball"
"""


@dataclass
class MuseReading:
    summary: str
    objects: list[str]
    raw: str = ""
    seconds: float = 0.0

    @property
    def primary(self) -> str | None:
        return self.objects[0] if self.objects else None


class MuseClient:
    def __init__(self, api_key: str, base_url: str, model: str = DEFAULT_MODEL, timeout: float = 600):
        self.base = base_url.rstrip("/")
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout

    def upload(self, path: Path) -> str:
        with path.open("rb") as f:
            r = requests.post(f"{self.base}/files", headers=self.headers,
                              data={"purpose": "user_data"},
                              files={"file": (path.name, f, "video/mp4")}, timeout=self.timeout)
        self._raise(r)
        return r.json()["id"]

    def watch(self, file_id: str, prompt: str = PROMPT) -> MuseReading:
        body = {
            "model": self.model,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_video", "file_id": file_id},
            ]}],
            "stream": True,
        }
        t0 = time.time()
        for attempt in range(4):
            r = requests.post(f"{self.base}/responses", json=body, stream=True, timeout=self.timeout,
                              headers={**self.headers, "Accept": "text/event-stream"})
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                wait = float(r.headers.get("retry-after") or 2 ** (attempt + 1))
                print(f"    Muse {r.status_code}, retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            self._raise(r)
            text = self._read_stream(r)
            reading = parse_reading(text)
            reading.seconds = time.time() - t0
            return reading
        raise RuntimeError("Muse request kept failing")

    @staticmethod
    def _read_stream(r) -> str:
        parts, status = [], None
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            kind = event.get("type", "")
            if kind == "response.output_text.delta":
                parts.append(event.get("delta", ""))
            elif kind in ("response.completed", "response.incomplete"):
                status = kind
                if not parts:   # some servers only send the full text at the end
                    for item in event.get("response", {}).get("output", []):
                        for c in item.get("content", []):
                            if c.get("type") == "output_text":
                                parts.append(c.get("text", ""))
            elif kind in ("response.failed", "error"):
                raise RuntimeError(f"Muse failed: {json.dumps(event)[:500]}")
        if status == "response.incomplete":
            print("    warning: Muse response was incomplete")
        return "".join(parts)

    @staticmethod
    def _raise(r) -> None:
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} from Meta API: {r.text[:500]}")


# A concept prompt is a thing, not a phrase: drop anything that reads like prose.
_JUNK = re.compile(r"^(?:the|a|an|and|or|hand|hands|left|right|person|camera|background|none|n/?a)$", re.I)


def clean_object(name: str) -> str:
    name = re.sub(r"^[\s\-*•\d.)]+", "", name).strip().strip('"\'.,;')
    name = re.sub(r"\b(?:the|a|an|his|her|their|some)\b", " ", name, flags=re.I)
    return re.sub(r"\s+", " ", name).strip().lower()


def parse_reading(text: str) -> MuseReading:
    """Pull the SUMMARY paragraph and the OBJECTS list out of Muse's reply."""
    summary, objects = "", []
    m = re.search(r"SUMMARY\s*:?\s*(.+?)(?=\n\s*OBJECTS\s*:|\Z)", text, re.I | re.S)
    if m:
        summary = re.sub(r"\s+", " ", m.group(1)).strip()
    m = re.search(r"OBJECTS\s*:?\s*(.+?)(?:\n\s*\n|\Z)", text, re.I | re.S)
    if m:
        blob = m.group(1).strip()
        # tab-separated as asked, but accept newlines or commas if the model drifts
        parts = re.split(r"\t|\n|,|;|(?<!\w)\|(?!\w)", blob) if blob else []
        seen = set()
        for p in parts:
            o = clean_object(p)
            if o and not _JUNK.match(o) and len(o) <= 40 and o not in seen:
                seen.add(o)
                objects.append(o)
    if not summary and not objects:
        summary = re.sub(r"\s+", " ", text).strip()
    return MuseReading(summary, objects, text)
