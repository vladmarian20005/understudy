"""Gravity sag. The SO-101's servos hold a position with a steady error under load, so an extended arm sits below where
it was told to go (1-1.5 cm at the fingertips on the real arms). The error on each joint grows with the gravity
torque on it, which the joint's own encoder sees: goal minus present position.

torque(arm, q) is that torque, from point masses along the arm: how the arm's potential energy changes per degree of
each joint. A joint's sag is about compliance x torque. SagModel learns each joint's compliance from the arm's own
readings while it holds still, and offset() is what to add to a goal so the arm ends where it was asked to. It starts
from PRIOR (about the 1-1.5 cm the real arms showed) or from what it learned last time (load/save), and refines that
within a second of the first hold.
"""
import json
import os

import numpy as np

# Point masses (kg) at the middle of each segment of Arm.points: shoulder bracket, upper arm, forearm, wrist, gripper
# with its camera. Only their ratios matter: the learned compliance takes the scale.
MASSES = np.array([0.05, 0.12, 0.12, 0.08, 0.25])
STEP_DEG = 0.5
GRAVITY_JOINTS = (1, 2, 3)  # shoulder lift, elbow, wrist flex; the pan and the wrist roll turn about near-vertical axes
LEARN_RATE = 0.1
MIN_TORQUE = 0.0003  # learn a joint's compliance only where gravity pulls on it this much: in a raised pose a few
# counts of stiction swamp the sag
MAX_COMPLIANCE = 8000.0  # deg per unit torque: the softest joint measured needs about 5500
MAX_OFFSET_DEG = 8.0
PRIOR = (0.0, 1000.0, 1100.0, 700.0, 0.0)


def potential(arm, q):
    pts = arm.points(q)
    mids = (pts[:-1, 2] + pts[1:, 2]) / 2
    return float(MASSES @ mids)


def torque(arm, q):
    """dU/dq (per degree) for the 5 joints. Gravity pushes each joint toward -torque."""
    q = np.asarray(q, dtype=float)[:5]
    out = np.zeros(5)
    for i in GRAVITY_JOINTS:
        up, down = q.copy(), q.copy()
        up[i] += STEP_DEG
        down[i] -= STEP_DEG
        out[i] = (potential(arm, up) - potential(arm, down)) / (2 * STEP_DEG)
    return out


class SagModel:
    """One arm's compliance per joint (deg per unit torque), learned from holds: present = commanded - C * torque."""

    def __init__(self, compliance=PRIOR):
        self.c = np.asarray(compliance, dtype=float).copy()
        self.samples = 0

    def offset(self, arm, q):
        """Degrees to add to a goal at q so the arm settles at q."""
        return np.clip(self.c * torque(arm, q), -MAX_OFFSET_DEG, MAX_OFFSET_DEG)

    def learn(self, arm, commanded, present):
        """One settled reading: the arm was commanded `commanded` and sits at `present`."""
        tau = torque(arm, present)
        err = np.asarray(commanded, dtype=float)[:5] - np.asarray(present, dtype=float)[:5]
        # signed: the servos stick either way of where they sag to, and clipping the readings at 0 drags the average down
        for i in GRAVITY_JOINTS:
            if abs(tau[i]) >= MIN_TORQUE:
                seen = float(np.clip(err[i] / tau[i], -MAX_COMPLIANCE, MAX_COMPLIANCE))
                self.c[i] = float(np.clip(self.c[i] + LEARN_RATE * (seen - self.c[i]), 0.0, MAX_COMPLIANCE))
        self.samples += 1


def load(path, sides):
    """SagModels for the arms, from a file saved earlier (prior values for arms it doesn't have)."""
    try:
        with open(path) as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    return {s: SagModel(saved[s]) if s in saved else SagModel() for s in sides}


def save(path, models):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({s: [round(float(v), 1) for v in m.c] for s, m in models.items()}, f)
