"""Meta Model API transport."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from sam_pipeline.config import MODEL


class SamClient:
    def __init__(self, api_key: str, base_url: str, timeout: float = 900):
        self.base = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout

    def upload(self, path: Path) -> str:
        with path.open("rb") as f:
            r = requests.post(f"{self.base}/files", headers=self.headers,
                              data={"purpose": "user_data"},
                              files={"file": (path.name, f, "video/mp4")}, timeout=self.timeout)
        self._raise(r)
        return r.json()["id"]

    def segment(self, file_id: str, prompt: str) -> str:
        """One concept prompt on one video. Returns SAM's raw text output (all frames)."""
        body = {
            "model": MODEL,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_video", "file_id": file_id},
            ]}],
            "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
            "stream": True,
        }
        for attempt in range(5):
            r = requests.post(f"{self.base}/responses", json=body, stream=True, timeout=self.timeout,
                              headers={**self.headers, "Accept": "text/event-stream"})
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                wait = float(r.headers.get("retry-after") or 2 ** (attempt + 1))
                print(f"    API {r.status_code}, retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            self._raise(r)
            return self._read_stream(r)
        raise RuntimeError("SAM request kept failing")

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
                raise RuntimeError(f"SAM failed: {json.dumps(event)[:500]}")
        if status == "response.incomplete":
            print("    warning: SAM response was incomplete")
        return "".join(parts)

    @staticmethod
    def _raise(r) -> None:
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} from Meta API: {r.text[:500]}")
