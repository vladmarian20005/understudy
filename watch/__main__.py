"""Read a video of a person's hands: Muse Spark says what they do and what they touch, SAM 3.1 segments those things
and the hands, and a score says who held what when (watch/SCORE.md). The code is in sam_pipeline/.

    python -m watch --video clip.mp4        from the repo root; --help for the options
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# sam_pipeline lives next to this file: importable however this is started
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sam_pipeline.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
