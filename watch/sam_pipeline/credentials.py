"""The Meta Model API key.

MODEL_API_KEY in the environment wins; FALLBACK_KEY below is used when it is unset.
Keeping a live key in source is convenient but it travels with the repo -- prefer the
environment variable and leave FALLBACK_KEY empty on anything you push.
"""

from __future__ import annotations

import os
import sys

ENV_VAR = "MODEL_API_KEY"
FALLBACK_KEY = ""


def api_key() -> str:
    key = os.environ.get(ENV_VAR) or FALLBACK_KEY
    if not key:
        sys.exit("Set MODEL_API_KEY to the key Meta gave you (export MODEL_API_KEY=...)")
    return key
