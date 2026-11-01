"""Tunables and the vocabulary that decides which Ego4D clips are relevant."""

from __future__ import annotations

import re

# --------------------------------------------------------------------------------------
# Meta Model API
# --------------------------------------------------------------------------------------
MODEL = "sam-3.1"
BILLING_MIN_FRAMES = 50
MAX_FRAMES_PER_REQUEST = 15000

# --------------------------------------------------------------------------------------
# What counts as a relevant clip
# --------------------------------------------------------------------------------------
SCENARIOS = ["Crafting/knitting/sewing/drawing/painting"]

# Narration verbs that describe moving something between hands.
VERBS = r"(?:transfer|pass|switch|change|move|shift|swap|hand)(?:s|es|ed|ing)?"
HAND_TO_HAND = [
    re.compile(rf"\b{VERBS}\b(?P<obj>.*?)\bfrom\s+(?:his|her|their|the|c'?s)?\s*(?P<src>left|right)\s+hand\s+(?:to|into|in)\s+(?:his|her|their|the|c'?s)?\s*(?P<dst>left|right|other)\s+hand", re.I),
    re.compile(rf"\b{VERBS}\b(?P<obj>.*?)\b(?:to|into|in)\s+(?:his|her|their|the|c'?s)?\s*(?P<dst>left|right|other)\s+hand", re.I),
    re.compile(r"\b(?:switch|change|swap)(?:s|es|ed|ing)?\b(?P<obj>.*?)\bhands\b", re.I),
]

# Rigid, light, elongated things two small grippers can pass mid-air. First word wins as SAM prompt.
GOOD_OBJECTS = {
    "paintbrush": "paintbrush", "paint brush": "paintbrush", "brush": "paintbrush",
    "pen": "pen", "pencil": "pencil", "marker": "marker", "crayon": "crayon",
    "glue stick": "glue stick", "glue": "glue stick", "ruler": "ruler",
    "stick": "stick", "chalk": "chalk", "highlighter": "marker", "spool": "spool",
    "tube": "paint tube", "eraser": "eraser", "scissors": "scissors",
}
# Things the arms should not try (bendy, tiny, sharp).
BAD_OBJECTS = ["thread", "yarn", "wool", "fabric", "cloth", "paper", "needle", "knife",
               "cutter", "blade", "bead", "string", "tape"]

ARTICLES = re.compile(r"\b(?:the|a|an|his|her|their|some|piece of|c'?s|c)\b", re.I)

# --------------------------------------------------------------------------------------
# Deciding who is holding the tool
# --------------------------------------------------------------------------------------
# Tuned against the 21 narrated hand-to-hand passes in the Ego4D crafting set, scoring
# against the narration's own giver/taker. The shipped defaults (0.15 / 0.1 / no gap)
# found 5 of 21; these find 13, with no direction ever flipped by the loosening.
#
# A hand grips a brush near its end, so only a sliver of the tool's box sits inside the
# hand's, and an egocentric camera foreshortens both -- hence a low share over a
# generously grown hand box.
HOLD_OVERLAP = 0.10        # share of the tool's box that must fall inside a holder's
HOLD_BOX_PAD = 0.30        # holder boxes are grown by this fraction first
HOLD_SMOOTH_FRAMES = 5     # median-ish window over the per-frame holder sets

# Mid-handover the tool is routinely occluded by both hands for a few frames, so a pass
# is still a pass if nobody is credited with holding it for up to this long.
PASS_MAX_UNHELD_FRAMES = 25   # 0.83 s at 30 fps

# --------------------------------------------------------------------------------------
# Prompt fallbacks
# --------------------------------------------------------------------------------------
# SAM returns an empty stream when a concept does not fire at all -- "paintbrush" finds
# nothing in a house-painting clip, where the object is a wide flat wall brush. Each entry
# is tried in order until one comes back with detections; the first hit wins and is what
# the score records. Keep them concrete: vague prompts match the whole frame or nothing.
OBJECT_FALLBACKS = {
    "paintbrush": ["paint brush", "brush", "brush held in hand"],
    "pen": ["ballpoint pen", "pen held in hand"],
    "pencil": ["colored pencil", "pencil held in hand"],
    "marker": ["marker pen", "felt tip pen"],
    "crayon": ["crayon stick", "colored crayon"],
    "glue stick": ["glue bottle", "glue container"],
    "ruler": ["straight ruler", "measuring ruler"],
    "stick": ["wooden stick"],
    "chalk": ["chalk stick", "piece of chalk"],
    "spool": ["thread spool", "spool of thread"],
    "paint tube": ["tube of paint", "paint container"],
    "eraser": ["rubber eraser"],
    "scissors": ["pair of scissors"],
    # Muse calls a tennis-ball can "tube"; SAM finds nothing for "tube" there (30 Oct, test_video_balls)
    "tube": ["tennis ball can", "can", "cylinder"],
    "can": ["tennis ball can", "cylinder"],
    "ball": ["tennis ball", "round ball"],
    "cup": ["mug", "paper cup"],
    "box": ["cardboard box", "small box"],
}

# Tried after the object-specific list is exhausted, for any object.
GENERIC_OBJECT_FALLBACKS = ["tool held in hand", "object held in hand"]


def object_fallbacks(prompt: str) -> list[str]:
    """Alternative SAM concepts to try when `prompt` comes back with nothing."""
    chain = list(OBJECT_FALLBACKS.get(prompt, []))
    chain += [p for p in GENERIC_OBJECT_FALLBACKS if p not in chain]
    return [p for p in chain if p != prompt]
