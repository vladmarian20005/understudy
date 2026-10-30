"""The score format: SCORE.md's example is a score delegate accepts, and the pipeline's handover block."""
import json
import os
import re
import sys

WATCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WATCH)  # sam_pipeline lives in watch/

from sam_pipeline.detections import Det  # noqa: E402
from sam_pipeline.ego4d import handover_summary  # noqa: E402
from watch import delegate  # noqa: E402


def test_the_example_in_score_md_is_a_score_delegate_accepts():
    text = open(os.path.join(WATCH, "SCORE.md")).read()
    example = json.loads(re.search(r"```json\n(.*?)```", text, re.S).group(1).replace('"RRRRBBBBLLLL..."', '""'))
    delegate.check(example)
    assert example["version"] == 1 and example["frame"] in ("ego", "overhead")
    first_cells = [line.split("|")[1] for line in text.splitlines() if line.startswith("| `")]
    documented = {name for cell in first_cells for name in re.findall(r"`([a-z_]+)", cell)}
    assert set(example) <= documented  # every field in the example is in the table


def test_the_handover_block_is_the_pass_and_where_the_object_was():
    track = [Det(f, "1", (0, 0, 10, 10), 0.9, (100.0 + f, 50.0), 80.0) for f in range(0, 90, 3)]
    h = handover_summary({"t": 1.0, "t_end": 1.6, "kind": "pass", "from": "right", "to": "left"}, track, 30.0)
    assert h == {"t": 1.0, "t_end": 1.6, "both_s": 0.6, "at_px": [139, 50], "at_cm": None}
    assert handover_summary(None, track, 30.0) is None
    assert handover_summary({"t": 1.0, "t_end": 1.6}, [], 30.0)["at_px"] is None
