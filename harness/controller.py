"""The 50 Hz control loop for both arms, in one thread so phases start together on both.

Every tick it reads both arms, takes the next row of the playing trajectory as the goal (or keeps holding the last
goal), writes the goals, and calls the tick listeners (the recorder). In the simulator it also steps the world.
A plan is done once the joints have settled on its last goal (the servos lag a little behind).
Torque is only switched on by arm(), which first sets every goal to where the motors are now.

Reflexes: a joint more than TRACK_LIMIT degrees behind its goal for TRACK_TICKS ticks stops everything (something
blocks the arm); a servo hotter than HOT_C stops everything too.
"""
import threading
import time

import numpy as np

from .motion import TICK

TRACK_LIMIT = 25.0  # deg
TRACK_TICKS = 25
START_TOLERANCE = 4.0  # deg: a trajectory must start where the arm's goal is
SETTLE_DEG = 2.0  # a plan is done when every joint is this close to its last goal
SETTLE_TICKS = 50  # or after this long
TEMP_EVERY = 100  # ticks
WARM_C, HOT_C = 55, 62


class Controller:
    def __init__(self, ios, world=None, speed=1.0, on_event=None):
        self.ios = ios
        self.world = world
        self.speed = speed  # simulator only: 1 = real time, 0 = as fast as it computes
        self.on_event = on_event or (lambda text, kind="info": None)
        self.listeners = []
        self.lock = threading.RLock()
        self.q = {s: io.read() for s, io in ios.items()}
        self.goal = {s: q.copy() for s, q in self.q.items()}
        self.mode = {s: "off" for s in ios}
        self.queue = []
        self.k = 0
        self.settle = 0
        self.idle = threading.Event()
        self.idle.set()
        self.reason = None
        self.ticks = 0
        self.t = 0.0
        self.lag = {s: 0 for s in ios}
        self.temps = {}
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name="control")

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.running = False
        self.thread.join(timeout=1)

    def now(self):
        return self.t

    def armed(self):
        return all(m != "off" for m in self.mode.values())

    def snapshot(self):
        with self.lock:
            return dict(t=self.t, q={s: v.copy() for s, v in self.q.items()},
                        goal={s: v.copy() for s, v in self.goal.items()}, mode=dict(self.mode),
                        temps=dict(self.temps))

    # ---- torque ---------------------------------------------------------------------------------------------------
    def arm(self):
        """Torque on, holding where the arms are."""
        with self.lock:
            for s, io in self.ios.items():
                q = io.read()
                self.q[s] = q
                self.goal[s] = q.copy()
                io.torque(True)
                self.mode[s] = "hold"
        self.on_event("armed: torque on, holding position")

    def disarm(self):
        with self.lock:
            self._abort("torque off")
            for s, io in self.ios.items():
                io.torque(False)
                self.mode[s] = "off"
        self.on_event("torque off", "warn")

    # ---- motion ---------------------------------------------------------------------------------------------------
    def play(self, trajs, cancel=None):
        """Play the trajectories in order and wait. Returns (True, None) when done, (False, why) when stopped."""
        with self.lock:
            if not self.armed():
                return False, "torque off"
            if self.queue:
                return False, "busy"
            first = trajs[0]
            for s, rows in first.rows.items():
                off = float(np.max(np.abs(rows[0, :5] - self.goal[s][:5])))
                if off > START_TOLERANCE:
                    return False, f"the plan starts {off:.0f} deg from where {s} is"
            self.queue = list(trajs)
            self.k = self.settle = 0
            self.reason = None
            self.idle.clear()
            for s in first.rows:
                self.mode[s] = "moving"
        while not self.idle.wait(0.05):
            if cancel is not None and cancel.is_set():
                self.stop("stop requested")
        return self.reason is None, self.reason

    def stop(self, reason="stop requested"):
        """Both arms hold where they are now."""
        with self.lock:
            if self.queue:
                self.on_event(f"stopped: {reason}", "warn")
            self._abort(reason)
            for s in self.ios:
                if self.mode[s] != "off":
                    self.goal[s] = self.q[s].copy()
                    self.mode[s] = "hold"

    def _abort(self, reason):
        if self.queue:
            self.reason = reason
        self.queue = []
        self.idle.set()

    # ---- the loop -------------------------------------------------------------------------------------------------
    def _loop(self):
        next_t = time.monotonic()
        while self.running:
            try:
                self._tick()
            except Exception as e:  # a bus error: hold everything, report, keep the loop alive
                self.on_event(f"control error: {e!r}", "error")
                with self.lock:
                    self._abort(f"control error: {e!r}")
            if self.world is not None and self.speed == 0:
                continue
            next_t += TICK / (self.speed if self.world is not None else 1.0)
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()

    def _tick(self):
        with self.lock:
            for s, io in self.ios.items():
                self.q[s] = io.read()
            if self.queue:
                traj = self.queue[0]
                n = len(next(iter(traj.rows.values())))
                if self.k < n:
                    for s, rows in traj.rows.items():
                        self.goal[s] = rows[self.k].copy()
                    self.k += 1
                else:
                    self.settle += 1
                    far = max(float(np.max(np.abs(self.goal[s][:5] - self.q[s][:5]))) for s in traj.rows)
                    if far < SETTLE_DEG or self.settle >= SETTLE_TICKS or len(self.queue) > 1:
                        self.queue.pop(0)
                        self.k = self.settle = 0
                        if not self.queue:
                            for s in self.mode:
                                if self.mode[s] == "moving":
                                    self.mode[s] = "hold"
                            self.idle.set()
            for s, io in self.ios.items():
                if self.mode[s] != "off":
                    io.write(self.goal[s])
            self._reflexes()
            if self.world is not None:
                self.world.step(TICK)
                self.t += TICK
            else:
                self.t = time.monotonic()
            self.ticks += 1
            q = {s: v.copy() for s, v in self.q.items()}
            goal = {s: v.copy() for s, v in self.goal.items()}
            t = self.t
        for fn in self.listeners:
            fn(t, q, goal)

    def _reflexes(self):
        for s in self.ios:
            if self.mode[s] == "off":
                continue
            err = float(np.max(np.abs(self.goal[s][:5] - self.q[s][:5])))
            self.lag[s] = self.lag[s] + 1 if err > TRACK_LIMIT else 0
            if self.lag[s] >= TRACK_TICKS:
                self.lag[s] = 0
                self._abort(f"blocked: {s} is {err:.0f} deg behind its goal")
                self.on_event(f"reflex: {s} is {err:.0f} deg behind its goal, holding", "warn")
                for t in self.ios:
                    if self.mode[t] != "off":
                        self.goal[t] = self.q[t].copy()
                        self.mode[t] = "hold"
        if self.ticks % TEMP_EVERY == 0:
            for s, io in self.ios.items():
                temps = io.temperatures()
                self.temps[s] = temps
                hot = max(temps)
                if hot >= HOT_C and self.queue:
                    self._abort(f"hot: {s} servo at {hot:.0f} C")
                    self.on_event(f"{s} servo at {hot:.0f} C: stopped, let it cool", "warn")
                elif hot >= WARM_C and self.ticks % (TEMP_EVERY * 15) == 0:
                    self.on_event(f"{s} servo at {hot:.0f} C", "warn")
