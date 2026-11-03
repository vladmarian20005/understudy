# The score format (version 1)

A **score** is what a video of a person says happened; a **label** is what a robot episode actually did. The watch
pipeline (`python -m watch --video clip.mp4`) writes one per video, as `sam_out/videos/<name>/score.json`: Muse
Spark reads the video and names what the hands touch, and SAM 3.1 segments those things and both hands in every
frame.

Left and right are the person's own: their left hand is the left arm. The camera is on their head or in front of them,
so pixel positions don't map to the table (`frame` is `"ego"`, and `target_cm` and `handover.at_cm` are `null`).

## Example

```json
{
  "version": 1,
  "kind": "score",
  "task": "Pass the tube to the other arm",
  "object": {"name": "tube", "prompt": "tube"},
  "source": {"kind": "video", "file": "tube.mp4",
             "muse": {"model": "muse-spark-1.3", "summary": "right hand picks up the tube, passes it to the left hand",
                      "objects": ["tube", "ball"], "segmented": ["tube", "ball"]}},
  "frame": "ego",
  "fps": 30.0, "frames": 264, "size_px": [576, 1024],
  "giver": "right", "taker": "left",
  "handover": {"t": 1.93, "t_end": 2.27, "both_s": 0.34, "at_px": [255, 590], "at_cm": null},
  "target_cm": null,
  "vision": {
    "model": "sam-3.1",
    "objects": [{"name": "tube", "visible": 0.93, "instances": 1, "start_px": [234, 634], "end_px": [194, 744]},
                {"name": "ball", "visible": 0.39, "instances": 1, "start_px": [466, 971], "end_px": [320, 562]},
                {"name": "left hand", "visible": 1.0, "start_px": [239, 786], "end_px": [372, 812]},
                {"name": "right hand", "visible": 0.98, "start_px": [545, 1011], "end_px": [477, 841]}],
    "events": [{"t": 0.9, "kind": "grasp", "arm": "right"},
               {"t": 1.93, "t_end": 2.27, "kind": "pass", "from": "right", "to": "left"},
               {"t": 2.27, "kind": "release", "arm": "right"}]
  },
  "quality": {"pass_seen": true, "object_visible": 0.93, "object_prompt": "tube", "object_prompt_fallback": false,
              "segmentation": {"tool_frames": 0.93, "tool_instances": 1, "tool_confidence": 0.87,
                               "tool_jitter_px": 8.4, "two_hands_frames": 0.98, "hand_ids": 2,
                               "extra_hand_frames": 0.0},
              "notes": []},
  "held": "RRRRBBBBLLLL...",
  "files": {"clip": "clip.mp4", "muse": "muse.json", "sam_raw": ["sam_holder.txt", "sam_object.txt", "sam_object2.txt"],
            "masks": ["masks_holder.json", "masks_object.json", "masks_object2.json"]}
}
```

## Fields

| field | type | meaning |
|---|---|---|
| `version`, `kind` | 1, `"score"` | this format |
| `task` | string | the task in words, for people and logs |
| `object.name` | string | the first thing Muse named: the one the score follows |
| `object.prompt` | string | the SAM 3.1 concept that found it (can be a fallback, see `quality.object_prompt_fallback`) |
| `source` | object | the video's file, and Muse's reading of it: the model, its summary of what the hands do, the things it named and the ones SAM segmented |
| `frame` | `"ego"` | which camera the pixel positions are in |
| `fps`, `frames`, `size_px` | | the clip SAM saw. Times are seconds since its first frame; frame = round(t × fps) |
| `giver`, `taker` | `"left"` / `"right"` or `null` | who passed the object to whom, when a pass was seen |
| `handover` | object or `null` | the pass: when both start holding (`t`), when the giver lets go (`t_end`), how long both hold (`both_s`), where the object was (`at_px`; `at_cm` is for a calibrated camera). `null` when no pass was seen |
| `target_cm` | `[x, y]` or `null` | where the object ended, on the table, for a calibrated camera |
| `vision` | object | the same shape as a label's `vision` section, so one reader handles both. `objects[0]` is the followed object, then the other things Muse named, then `left hand`, `right hand`. Each may also have `instances` (how many of it SAM tracked: 4 balls), `color` (its colour in preview.mp4) and `muse_name` (what Muse called it, when SAM found it under a fallback concept). `events` are `grasp`, `release`, `pass`, `drop`, with `arm`, or `from`/`to` for a pass |
| `quality` | object | whether to trust it: `pass_seen` (a pass was detected), how much of the clip the object is visible, SAM's segmentation numbers, parser notes |
| `held` | string | debug: who held the object, one letter per frame (`L`, `R`, `B` both, `-` nobody) |
| `files` | object | the clip, Muse's reading, SAM's raw output and the masks, next to `score.json` |

## Rules

- `giver`/`taker` in a score map one to one onto the arms. The robot's objects are whatever is on its table, not
  necessarily what the person held.
- `null` means "not seen in this clip", never zero.
- New fields may be added; existing ones keep their meaning within version 1.
