"""The hard budget ledger.

Every SAM request is charged against a ledger file (default 60 minutes of video in total,
across all runs and both modes). A request that would go over is not sent. A request is
charged for its whole clip, at least 50 frames (Meta's billing minimum), and each prompt
is its own request, so a 6 s clip with two prompts costs 12 s.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sam_pipeline.config import BILLING_MIN_FRAMES


class Budget:
    def __init__(self, path: Path, limit_s: float):
        self.path = path
        self.limit_s = limit_s
        self.data = {"limit_s": limit_s, "used_s": 0.0, "requests": []}
        if path.exists():
            self.data = json.loads(path.read_text())
        # the limit can only be tightened from the command line, never raised by an old ledger
        self.data["limit_s"] = limit_s

    @property
    def used(self) -> float:
        return self.data["used_s"]

    @property
    def left(self) -> float:
        return self.limit_s - self.used

    @staticmethod
    def cost(frames: int, fps: float) -> float:
        return max(frames, BILLING_MIN_FRAMES) / fps

    def can_spend(self, seconds: float) -> bool:
        return self.used + seconds <= self.limit_s + 1e-9

    def spend(self, seconds: float, what: str) -> None:
        self.data["used_s"] = round(self.used + seconds, 3)
        self.data["requests"].append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "s": round(seconds, 3), "what": what})
        self.path.write_text(json.dumps(self.data, indent=1))
