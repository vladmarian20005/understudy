# The score format (version 1)

A **score** is what a human video says should happen; a **label** is what a robot episode actually did
(see the contracts doc). The watch pipeline (`python -m watch.sam_ego4d ego4d ...`) writes one score per clip, as
`sam_out/scores/<clip_id>/score.json`.

## Where a score comes from

| `source.kind` | camera | `frame` | what carries over to the arms |
|---|---|---|---|
| `ego4d` | a head-mounted camera that moves | `"ego"` | who gives, who takes, the order and the timing. Pixel positions are in a moving camera, so they don't map to the table: `target_cm` is `null`. |
| `clip` (planned) | the rig's calibrated overhead camera (`harness/clip.py`) | `"overhead"` | all of the above, and positions in table cm (`target_cm`, `handover.at_cm`). |

Left and right are the same for a person and the rig: the wearer's left hand is the left arm. In the overhead camera,
the left arm is on the left of the image.

## Example (an Ego4D clip)

```json
{
  "version": 1,
  "kind": "score",
  "task": "Pass the paintbrush to the other arm",
  "object": {"name": "paintbrush", "prompt": "paintbrush"},
  "source": {"kind": "ego4d", "video_uid": "...", "start_s": 125.88, "end_s": 131.88,
             "narration": "#C C transfers the paintbrush from his right hand to his left hand",
             "narration_t": 2.5, "file": "<clip_uid>.mp4", "file_offset_s": 128.38},
  "frame": "ego",
  "fps": 30.0, "frames": 180, "size_px": [1280, 960],
  "giver": "right", "taker": "left",
  "handover": {"t": 1.93, "t_end": 2.27, "both_s": 0.34, "at_px": [655, 590], "at_cm": null},
  "target_cm": null,
  "vision": {
    "model": "sam-3.1",
    "objects": [{"name": "paintbrush", "visible": 0.99, "start_px": [863, 496], "end_px": [448, 848]},
                {"name": "left hand", "visible": 1.0, "start_px": [353, 587], "end_px": [478, 678]},
                {"name": "right hand", "visible": 1.0, "start_px": [748, 497], "end_px": [705, 629]}],
    "events": [{"t": 1.93, "t_end": 2.27, "kind": "pass", "from": "right", "to": "left"},
               {"t": 2.27, "kind": "release", "arm": "right"}]
  },
  "steps": [{"op": "reach", "arm": "right", "t": null},
            {"op": "grasp", "arm": "right", "t": null},
            {"op": "handover", "giver": "right", "taker": "left", "mode": "air", "t": [1.93, 2.27]},
            {"op": "move_to", "arm": "left", "t": null},
            {"op": "release", "arm": "left", "t": null}],
  "quality": {"pass_seen": true, "agrees_with_narration": true, "object_visible": 0.99,
              "object_prompt": "paintbrush", "object_prompt_fallback": false,
              "segmentation": {"tool_frames": 0.99, "tool_instances": 1, "tool_confidence": 0.87,
                               "tool_jitter_px": 8.4, "two_hands_frames": 1.0, "hand_ids": 2,
                               "extra_hand_frames": 0.0},
              "notes": []},
  "held": "RRRRBBBBLLLL...",
  "files": {"clip": "clip.mp4", "sam_raw": ["sam_holder.txt", "sam_object.txt"],
            "masks": ["masks_holder.json", "masks_object.json"]}
}
```

## Fields

| field | type | meaning |
|---|---|---|
| `version`, `kind` | 1, `"score"` | this format |
| `task` | string | the task in words, for people and logs |
| `object.name` | string | what the person passed, as the narration names it |
| `object.prompt` | string | the SAM 3.1 concept that found it (can be a fallback, see `quality.object_prompt_fallback`) |
| `source` | object | where the clip came from: for `ego4d`, the video, the window in it (`start_s`, `end_s`), the narration and its time in the clip, and the file and offset it was cut from |
| `frame` | `"ego"` or `"overhead"` | which camera the pixel positions are in |
| `fps`, `frames`, `size_px` | | the clip SAM saw. Times are seconds since its first frame; frame = round(t × fps) |
| `giver`, `taker` | `"left"` / `"right"` | who passes to whom: the detected pass when there is one, else what the narration says |
| `handover` | object or `null` | the pass, as the robot needs it: when both start holding (`t`), when the giver lets go (`t_end`), how long both hold (`both_s`), where the object was (`at_px`) and, overhead clips only, in table cm (`at_cm`). `null` when no pass was seen |
| `target_cm` | `[x, y]` or `null` | overhead clips only: where the object ended, on the table. The robot's target |
| `vision` | object | the same shape as a label's `vision` section, so one reader handles both. `objects[0]` is the passed object, then (video mode) the other objects Muse named, then `left hand`, `right hand`. Each may also have `instances` (how many of it SAM tracked: 4 balls), `color` (its colour in preview.mp4) and `muse_name` (what Muse called it, when SAM found it under a fallback concept). `events` are `grasp`, `release`, `pass`, `drop`, with `arm`, or `from`/`to` for a pass |
| `steps` | list | the five primitives the robot would run, with the contract's op names, and `t` = when the clip shows that step (`[start, end]`), or `null` when it doesn't |
| `quality` | object | whether to trust it: `pass_seen` (a pass was detected), `agrees_with_narration`, how much of the clip the object is visible, SAM's segmentation numbers, parser notes |
| `held` | string | debug: who held the object, one letter per frame (`L`, `R`, `B` both, `-` nobody) |
| `files` | object | the clip, SAM's raw output and the masks, next to `score.json` |

## Rules

- `giver`/`taker` in a score map one to one onto the arms. The robot's object is whatever is on the table
  (the simulator's `brush`), not necessarily what the person held.
- `null` means "not seen in this clip", never zero.
- New fields may be added; existing ones keep their meaning within version 1.
