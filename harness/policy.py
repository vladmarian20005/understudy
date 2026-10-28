"""The policy runner: a trained LeRobot policy (ACT) drives both arms through robotd, with no model API in the loop.

policy_start loads a checkpoint and runs it at the dataset's 30 fps. Each step reads what the recorder stores for a
frame (the 12 joint positions and the three cameras, RGB), asks the policy for the 12 commanded positions, and streams
them to the controller, which moves toward them at a capped rate. Every streamed pose passes the same checks as a
planned trajectory: joints a little past their limits are clamped to them (the scripted grasps work right at the wrist
limit, so a trained policy overshoots it by a few degrees), and a pose far past a limit, or into the table, out of the
workspace or too close to the other arm stops the policy and both arms hold. The controller's reflexes stay on, and torque still only comes on with the a key.

In the simulator the policy steps in lockstep with simulated time (from the controller's tick), so --speed 0 runs it
flat out and the result doesn't depend on how fast the computer is. On the real arms it runs on its own thread.

A checkpoint is a `pretrained_model` directory, or a training output directory (its checkpoints/last is used).
"""
import os
import threading
import time

import numpy as np

from .cameras import CameraDown
from .kin import JOINTS
from .motion import ARM_FLOOR, ARM_GAP, TIP_FLOOR, Z_MAX, arms_gap
from .recorder import CAM_KEYS, FPS, frame_images, vector
from .scene import SIDES

TASK = "Pass the brush to the other arm"
REFUSE_DEG = 15.0  # this far past a joint limit is a broken policy, not an overshoot
FLOOR_SLACK = 0.002  # m below the planner's fingertip floor
GAP_FRACTION = 0.5  # of the planner's arm gap


def resolve(path):
    """The pretrained_model directory for a checkpoint or a training run."""
    path = os.path.abspath(os.path.expanduser(path))
    for cand in (path, os.path.join(path, "pretrained_model"), os.path.join(path, "checkpoints", "last", "pretrained_model")):
        if os.path.exists(os.path.join(cand, "config.json")):
            return cand
    raise FileNotFoundError(f"no policy checkpoint (config.json) in {path}")


def load(path, device=None):
    """(policy, preprocessor, postprocessor, device) for a checkpoint."""
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post, torch.device(device)


def check_pose(scene, targets):
    """None if both target poses are safe, else why not. Same checks as a planned trajectory, slightly looser.
    Clamps joints that are a little past their limits, in place."""
    pts = {}
    for s, q in targets.items():
        arm = scene.arms[s]
        if not arm.within_limits(q, slack=REFUSE_DEG):
            j = int(np.argmax(np.maximum(arm.limits[:, 0] - q[:5], q[:5] - arm.limits[:, 1])))
            return (f"{s} {JOINTS[j]} would go to {q[j]:.0f} deg, past its limits "
                    f"[{arm.limits[j, 0]:.0f}, {arm.limits[j, 1]:.0f}]")
        q[:5] = np.clip(q[:5], arm.limits[:, 0], arm.limits[:, 1])
        pts[s] = arm.points(q)
        tip = pts[s][-1]
        if tip[2] < TIP_FLOOR - FLOOR_SLACK:
            return f"{s} fingertips would go {tip[2] * 100:.1f} cm from the table"
        if np.min(pts[s][1:-1, 2]) < ARM_FLOOR - FLOOR_SLACK:
            return f"{s} arm would come within 2 cm of the table"
        if not scene.on_table(tip[:2]) or tip[2] > Z_MAX:
            return f"{s} fingertips would leave the workspace at ({tip[0] * 100:.0f}, {tip[1] * 100:.0f}) cm"
    if len(pts) == 2:
        gap = arms_gap(pts["left"], pts["right"])
        if gap < ARM_GAP * GAP_FRACTION:
            return f"the arms would come within {max(gap, 0) * 100:.1f} cm of each other"
    return None


class PolicyRunner:
    def __init__(self, core):
        self.core = core
        self.cache = {}
        self.run = None  # the active run
        self.last = None  # the last run's summary
        self.lock = threading.Lock()
        core.ctl.listeners.append(self._tick)

    def active(self):
        return self.run is not None

    def status(self):
        r = self.run
        if r is None:
            return dict(running=False, last=self.last)
        return dict(running=True, name=r["name"], t=round(self.core.ctl.now() - r["t0"], 2), steps=r["steps"],
                    infer_ms=round(float(np.mean(r["ms"][-30:])), 1) if r["ms"] else None)

    def start(self, path, max_s=None, task=TASK):
        core = self.core
        if self.run is not None:
            return dict(ok=False, errors=[f"a policy is already running ({self.run['name']})"], fail="busy")
        if core.busy():
            return dict(ok=False, errors=[f"{core.skill['name']} is still running"], fail="busy")
        if not core.ctl.armed():
            return dict(ok=False, errors=["the arms are off: Vlad switches torque on (key a)"], fail="torque_off")
        try:
            path = resolve(path)
            if path not in self.cache:
                self.cache[path] = load(path)
        except Exception as e:
            return dict(ok=False, errors=[f"can't load the policy: {e}"])
        policy = self.cache[path][0]
        policy.reset()
        name = os.path.relpath(path)
        with self.lock:
            self.run = dict(name=name, path=path, task=task, max_s=max_s, t0=core.ctl.now(), next_t=core.ctl.now(),
                            steps=0, ms=[])
        core.event(f"policy {name} started" + (f", {max_s:.0f} s at most" if max_s else ""))
        if core.world is None:  # the real arms: wall-clock 30 Hz on a thread of its own
            threading.Thread(target=self._loop, daemon=True, name="policy").start()
        return dict(ok=True, name=name)

    def stop(self, reason="policy stopped"):
        with self.lock:
            r, self.run = self.run, None
        if r is None:
            return dict(ok=False, errors=["no policy is running"])
        self.core.ctl.stop(reason)
        self.last = dict(name=r["name"], steps=r["steps"], duration_s=round(self.core.ctl.now() - r["t0"], 2),
                         why=reason)
        self.core.event(f"policy {r['name']} stopped after {r['steps']} steps: {reason}")
        return dict(ok=True, **self.last)

    # ---- stepping -------------------------------------------------------------------------------------------------
    def _tick(self, t, q, goal):
        r = self.run
        if r is None or self.core.world is None or t + 1e-9 < r["next_t"]:
            return
        r["next_t"] += 1.0 / FPS
        self._step(r, t, self.core.world.snapshot())

    def _loop(self):
        next_t = time.monotonic()
        while self.run is not None:
            r = self.run
            self._step(r, self.core.ctl.now(), None)
            next_t += 1.0 / FPS
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()

    def _step(self, r, t, snap):
        from lerobot.utils.control_utils import predict_action

        core = self.core
        if r["max_s"] and t - r["t0"] > r["max_s"]:
            self.stop(f"{r['max_s']:.0f} s are up")
            return
        policy, pre, post, device = self.cache[r["path"]]
        q = core.ctl.snapshot()["q"]
        try:
            images = core.cameras.grab(list(CAM_KEYS), snap) if snap is not None else core.cameras.grab(list(CAM_KEYS))
        except CameraDown as e:
            self.stop(f"camera down: {e}")
            return
        obs = {"observation.state": vector(q, core.scene.stroke_mm), **frame_images(images)}
        t_in = time.perf_counter()
        action = predict_action(obs, policy, device, pre, post, use_amp=False, task=r["task"],
                                robot_type="bimanual_so101")
        r["ms"].append((time.perf_counter() - t_in) * 1000)
        if snap is None and len(r["ms"]) == 30 and np.mean(r["ms"]) > 1000 / FPS:
            core.event(f"policy {r['name']} takes {np.mean(r['ms']):.0f} ms a step: slower than {FPS} fps", "warn")
        a = np.asarray(action.cpu(), dtype=float).reshape(-1)
        targets = {s: np.append(a[i * 6:i * 6 + 5], a[i * 6 + 5] / 100 * core.scene.stroke_mm)
                   for i, s in enumerate(SIDES)}
        why = check_pose(core.scene, targets)
        if why is not None:
            self.stop(f"refused a pose: {why}")
            return
        if not core.ctl.stream(targets):
            self.stop("the controller won't stream (torque off or a plan running)")
            return
        r["steps"] += 1
