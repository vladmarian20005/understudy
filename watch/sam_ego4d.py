#!/usr/bin/env python3
"""Entry point for the SAM 3.1 "Watch, Delegate, Distill" pipeline.

The implementation lives in the `sam_pipeline` package next to this file. From the repo root:

    python -m watch.sam_ego4d --help        (or: python watch/sam_ego4d.py --help)
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# sam_pipeline lives next to this file; harness (for --send) lives in the repo root. Both have to be importable
# however this is started: as a module from the root, or as a script from anywhere.
for path in (HERE, HERE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sam_pipeline.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
