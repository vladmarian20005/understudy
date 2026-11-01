"""The simulator the dashboard runs itself (python -m ui --sim): robotd on the HD sim, restarted on the scene each
video's objects make (watch/simscene.py), so the arms rehearse on what the video shows. Its output goes to
<runs>/robotd.log."""
import json
import os
import subprocess
import sys
import time

from harness import robot

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START_S = 120  # the HD sim builds its model and settles the objects first


class SimRig:
    def __init__(self, runs, dataset="understudy_ui", extra=()):
        self.runs = runs
        self.dataset = dataset
        self.extra = list(extra)
        self.proc = None
        self.scene = None

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, scene_path=None, log=lambda s: None):
        """robotd on scene_path (an empty table when None), once the last one has stopped; returns when it answers
        and is armed."""
        self.stop()
        try:  # someone else's robotd on the port: this one couldn't bind
            robot.call("status", timeout_s=0.5)
            raise RuntimeError("a robotd is already running: stop it, `python -m ui --sim` starts its own")
        except robot.RobotdDown:
            pass
        os.makedirs(self.runs, exist_ok=True)
        if scene_path is None:
            scene_path = os.path.join(self.runs, "empty_scene.json")
            json.dump(dict(objects=[]), open(scene_path, "w"))
        cmd = [sys.executable, "-m", "harness.robotd", "--sim", "--hd", "--objects", os.path.abspath(scene_path),
               "--dataset", self.dataset] + self.extra
        out = open(os.path.join(self.runs, "robotd.log"), "a")
        self.proc = subprocess.Popen(cmd, cwd=HERE, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
        self.scene = scene_path
        log(f"simulator starting on {os.path.basename(scene_path)}")
        deadline = time.time() + START_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the simulator stopped (exit {self.proc.returncode}): see {out.name}")
            try:
                if robot.call("status", timeout_s=0.5).get("armed"):
                    return
            except robot.RobotdDown:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"the simulator didn't answer in {START_S} s: see {out.name}")

    def stop(self):
        if self.running():
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None
