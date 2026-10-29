"""The episode recorder: one LeRobot dataset episode per robot episode, and its label file next to it.

episode_begin starts sampling on the controller's clock at 30 fps: the 12 joint positions as the state, the 12 goals
as the action (what robotd commanded at that moment), and the three cameras. episode_end writes the episode, closes
the dataset so it's complete on disk, and writes the label's robot section to datasets/<name>/sidecar/<episode>.json.
episode_label stores the vision and judge sections, and once the judge has spoken, the final training decision.

Units in the dataset: joints in lerobot degrees, the gripper 0-100 (0 = closed), like lerobot's SO-101 recordings.
"""
import datetime
import json
import os
import queue
import shutil
import threading

import cv2
import numpy as np

from .cameras import CameraDown
from .kin import JOINTS
from .scene import SIDES

FPS = 30
CAM_KEYS = {"overhead": "left_overhead", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}
# Every camera is stored at this size, whatever its sensor (the left wrist is 1024x768): the same features in the
# simulator and on the arms, and the policy runner feeds the policy exactly this.
RECORD_SIZE = (640, 480)
NAMES = [f"{s}_{j}.pos" for s in SIDES for j in JOINTS + ["gripper"]]
CHECK_CM = 3.0  # robotd's own check: the object ends this close to the target, in no gripper
REVIEW_CONFIDENCE = 0.7
HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(os.path.dirname(HERE), "datasets")


def vector(q, stroke):
    return np.concatenate([np.append(q[s][:5], q[s][5] / stroke * 100) for s in SIDES]).astype(np.float32)


def frame_images(images):
    """The dataset's view of camera images (BGR, any size): keyed by feature, RGB, RECORD_SIZE."""
    out = {}
    for cam, img in images.items():
        if (img.shape[1], img.shape[0]) != RECORD_SIZE:
            img = cv2.resize(img, RECORD_SIZE, interpolation=cv2.INTER_AREA)
        out[f"observation.images.{CAM_KEYS[cam]}"] = img[:, :, ::-1].copy()
    return out


class Recorder:
    def __init__(self, core, name="understudy_pass", root=DATASETS, vcodec="libsvtav1"):
        self.core = core
        self.name = name
        self.root = os.path.join(root, name)
        self.vcodec = vcodec
        self.lock = threading.Lock()
        self.ds = None
        self.ep = None  # the episode being recorded
        self.queue = None
        self.worker = None
        self.t0 = 0.0
        core.ctl.listeners.append(self._tick)

    def active(self):
        return self.ep is not None and self.ep["recording"]

    def status(self):
        if self.ep is None:
            return dict(recording=False)
        return dict(recording=self.ep["recording"], episode=self.ep["index"], frames=self.ep["frames"],
                    t=round(self.core.ctl.now() - self.t0, 2), task=self.ep["task"])

    # ---- the dataset ----------------------------------------------------------------------------------------------
    def _features(self, images):
        f = {"observation.state": dict(dtype="float32", shape=(12,), names=NAMES),
             "action": dict(dtype="float32", shape=(12,), names=NAMES)}
        for key, img in frame_images(images).items():
            f[key] = dict(dtype="video", shape=img.shape, names=["height", "width", "channels"])
        return f

    def _open(self, images):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        kw = dict(image_writer_threads=4, vcodec=self.vcodec)
        if os.path.exists(os.path.join(self.root, "meta", "info.json")):
            ds = LeRobotDataset.resume(f"local/{self.name}", root=self.root, **kw)
            want = self._features(images)
            have = {k: tuple(v["shape"]) for k, v in ds.meta.features.items() if k in want}
            if have != {k: tuple(v["shape"]) for k, v in want.items()}:
                raise ValueError(f"{self.root} was recorded with other cameras or shapes: {have}")
            return ds
        if os.path.exists(self.root):
            shutil.rmtree(self.root)  # a dataset that never got its first episode
        return LeRobotDataset.create(f"local/{self.name}", FPS, self._features(images), root=self.root,
                                     robot_type="bimanual_so101", metadata_buffer_size=1, **kw)

    # ---- commands -------------------------------------------------------------------------------------------------
    def begin(self, task, source=None, giver=None, taker=None, target_cm=None):
        with self.lock:
            if self.active():
                return dict(ok=False, errors=[f"episode {self.ep['index']} is still recording"])
            images = self.core.cameras.grab(list(CAM_KEYS))
            self.ds = self._open(images)
            objects = self.core.perception.detect() if self.core.perception else []
            self.ep = dict(index=self.ds.meta.total_episodes, task=task, source=source or {"kind": "pingpong"},
                           giver=giver, taker=taker, target_cm=target_cm, frames=0, camera_gaps=0, recording=True,
                           steps=[], bus0=self.core.ctl.bus_faults, takeovers0=self.core.ctl.takeovers,
                           ev0=self.core.ev_id, start_objects=objects,
                           started=datetime.datetime.now().astimezone().isoformat(timespec="milliseconds"))
            self.queue = queue.Queue()
            self.worker = threading.Thread(target=self._work, daemon=True, name="recorder")
            self.worker.start()
            self.t0 = self.core.ctl.now()
            self.next_t = self.t0
        self.core.event(f"episode {self.ep['index']} recording: {task}")
        return dict(ok=True, episode=self.ep["index"])

    def end(self, valid=True, note=None):
        with self.lock:
            if not self.active():
                return dict(ok=False, errors=["no episode is recording"])
            self.ep["recording"] = False
            duration = self.core.ctl.now() - self.t0
            self.queue.put(None)
            self.worker.join()
            ep = self.ep
            if ep["frames"] == 0:
                self.ds.clear_episode_buffer()
                self.ds.finalize()
                self.ds, self.ep = None, None
                return dict(ok=False, errors=["the episode has no frames"])
            # the videos are encoded here, one camera after another: lerobot's process pool for them is slower (each
            # worker starts and imports lerobot) and a worker dying now and then lost the episode
            self.ds.save_episode(parallel_encoding=False)
            self.ds.finalize()
            self.ds = None
            if ep["camera_gaps"]:
                valid = False
                note = "; ".join(filter(None, [note, f"camera drop-out: {ep['camera_gaps']} frames without new images"]))
            ep["bus_faults"] = self.core.ctl.bus_faults - ep["bus0"]
            if ep["bus_faults"]:
                valid = False
                note = "; ".join(filter(None, [note, f"bus drop-out: {ep['bus_faults']} motion(s) stopped"]))
            label = self._robot_label(ep, duration, valid, note)
            path = self._label_path(ep["index"])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(label, open(path, "w"), indent=1)
            self.ep = None
        self.core.event(f"episode {ep['index']} saved: {ep['frames']} frames, {duration:.1f} s"
                        + ("" if valid else " (marked invalid)"))
        return dict(ok=True, episode=ep["index"], frames=ep["frames"], duration_s=round(duration, 2), sidecar=path)

    def label(self, episode, section, data):
        if section not in ("vision", "judge"):
            return dict(ok=False, errors=["section must be vision or judge"])
        path = self._label_path(int(episode))
        if not os.path.exists(path):
            return dict(ok=False, errors=[f"no episode {episode} in {self.root}"])
        with self.lock:
            label = json.load(open(path))
            label[section] = data
            if "judge" in label:
                label["final"] = final(label)
            json.dump(label, open(path, "w"), indent=1)
        return dict(ok=True, episode=int(episode), final=label.get("final"))

    # ---- sampling -------------------------------------------------------------------------------------------------
    def _tick(self, t, q, goal):
        if not self.active() or t + 1e-9 < self.next_t:
            return
        self.next_t += 1.0 / FPS
        snap = self.core.world.snapshot() if self.core.world is not None else None
        self.queue.put((vector(q, self.core.scene.stroke_mm), vector(goal, self.core.scene.stroke_mm), snap))

    def _work(self):
        last = None
        while True:
            item = self.queue.get()
            if item is None:
                return
            state, action, snap = item
            try:
                images = last = self.core.cameras.grab(list(CAM_KEYS), snap) if snap is not None else \
                    self.core.cameras.grab(list(CAM_KEYS))
            except CameraDown as e:
                # Keep the frame timing with the last images; the episode can't be used for training.
                if self.ep["camera_gaps"] == 0:
                    self.core.event(f"episode {self.ep['index']}: {e}; it will be saved as invalid", "warn")
                self.ep["camera_gaps"] += 1
                if last is None:
                    continue
                images = last
            frame = {"observation.state": state, "action": action, "task": self.ep["task"], **frame_images(images)}
            self.ds.add_frame(frame)
            self.ep["frames"] += 1

    def on_step(self, result):
        if self.active():
            self.ep["steps"].append({k: result.get(k) for k in ("op", "arm", "giver", "taker", "t", "outcome", "fail",
                                                                "why", "width_mm") if result.get(k) is not None})

    # ---- the label ------------------------------------------------------------------------------------------------
    def _label_path(self, episode):
        return os.path.join(self.root, "sidecar", f"{episode:06d}.json")

    def _robot_label(self, ep, duration, valid, note):
        events = []
        for s in ep["steps"]:
            if s["outcome"] != "done":
                continue
            t = s["t"][1]
            if s["op"] in ("grasp", "pick"):
                events.append(dict(t=t, kind="grasp", arm=s["arm"], width_mm=s.get("width_mm")))
            elif s["op"] in ("release", "place"):
                events.append(dict(t=t, kind="release", arm=s["arm"]))
            elif s["op"] == "handover":
                events.append(dict(t=t, kind="pass", **{"from": s["giver"], "to": s["taker"]}))
        objects = self.core.perception.detect() if self.core.perception else []
        obj = None
        if objects:
            start = ep["start_objects"][0] if ep["start_objects"] else None
            end = objects[0] if start is None else min(
                objects, key=lambda o: (o["name"] != start["name"],
                                        np.hypot(*np.subtract(o["center_cm"], start["center_cm"]))))
            obj = dict(name=end["name"], start_cm=start["center_cm"] if start else None, end_cm=end["center_cm"],
                       held_by=end["held_by"])
        check = None
        if obj is not None and ep["target_cm"] is not None:
            dist = float(np.hypot(*np.subtract(obj["end_cm"], ep["target_cm"])))
            check = dict(passed=dist <= CHECK_CM and obj["held_by"] is None, distance_cm=round(dist, 1))
        return {"version": 1, "dataset": self.name, "episode": ep["index"], "task": ep["task"], "fps": FPS,
                "frames": ep["frames"], "source": ep["source"],
                "robot": dict(started=ep["started"], duration_s=round(duration, 2), giver=ep["giver"],
                              taker=ep["taker"], target_cm=ep["target_cm"], valid=bool(valid), note=note,
                              camera_gaps=ep["camera_gaps"], bus_faults=ep.get("bus_faults", 0),
                              interventions=self.core.ctl.takeovers - ep["takeovers0"], steps=ep["steps"], events=events, object=obj, check=check,
                              log=[dict(t=e["t"], kind=e["kind"], text=e["text"])
                                   for e in self.core.events_since(ep["ev0"])])}


def final(label):
    """The training decision: valid, the judge says success, and robotd's own check agrees."""
    robot, judge = label.get("robot", {}), label["judge"]
    check = (robot.get("check") or {}).get("passed")
    success = bool(judge.get("success"))
    conf = float(judge.get("confidence", 0))
    if not robot.get("valid", False):
        return dict(success=False, train=False, review=False, why="invalid episode")
    if check is not None and check != success:
        return dict(success=False, train=False, review=True, why="the judge and robotd's check disagree")
    if conf < REVIEW_CONFIDENCE:
        return dict(success=False, train=False, review=True, why=f"judge confidence {conf:.2f}")
    return dict(success=success, train=success, review=False,
                why="valid; judge and check agree" if success else "the judge says it failed")
