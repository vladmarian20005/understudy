"""The Meta Model API key.

Resolution order, first hit wins:
  1. MODEL_API_KEY in the environment
  2. MODEL_API_KEY (or MUSE_API_KEY) in the project's .env, the same file brain/llm.py reads
No key is kept in source: .env stays out of git.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_VAR = "MODEL_API_KEY"
ALT_VARS = ("MUSE_API_KEY",)


def load_env(start: Path | None = None) -> None:
    """KEY=value lines from the nearest .env above this file, without overriding the environment."""
    here = (start or Path(__file__).resolve()).parent
    for folder in [here, *here.parents][:5]:
        path = folder / ".env"
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip("'\"")
                if v:
                    os.environ.setdefault(k.strip(), v)
        return


def api_key() -> str:
    load_env()
    for var in (ENV_VAR, *ALT_VARS):
        key = os.environ.get(var)
        if key:
            return key
    sys.exit(f"No {ENV_VAR}. Put it in understudy/.env as {ENV_VAR}=... or export it.")
