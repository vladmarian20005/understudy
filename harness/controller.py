"""The 50 Hz control loop for both arms, in one thread so phases start together on both.

Every tick it reads both arms, takes the next row of the playing trajectory as the goal (or keeps holding the last
goal), writes the goals, and calls the tick listeners (the recorder). In the simulator it also steps the world.
A plan is done once the joints have settled on its last goal (the servos lag a little behind).
Torque is only switched on by arm(), which first sets every goal to where the motors are now.

Gravity: every goal is written with a feed-forward offset against the arm's sag (gravity.py), and while an arm holds
still the controller learns its compliance from the encoders, so the arms end where they're told to. Not while its
gripper squeezes something: the object's weight would read as a softer arm.

Contact: in a trajectory marked contact (a touch-down), a moving arm whose measured fingertips rise CONTACT_M above the
commanded ones, compared with where they were when the trajectory started (so a sag estimate that's still off doesn't
count), for CONTACT_TICKS ticks is resting on something: the trajectory ends with reason "contact: <arm>" and that arm
holds where it is.

Streaming (a policy driving the arms): stream() sets a pose target per arm, and every tick the goal moves toward it by
at most STREAM_DEG_S / STREAM_MM_S, so a jumpy policy can't jerk the arms. Trajectories and streaming exclude each
other.

Leader takeover: moving a leader arm more than TAKEOVER_DEG from where it rested takes its follower over at once. Any
motion or policy stops, the follower trails the leader (rate-limited, inside its joint limits) in mode "manual", and
nothing else can move until the leader has rested for RELEASE_S, when the arm holds where it was left. takeovers
counts them: they are the interventions.

Bus drop-outs: a read or write that fails (after the driver's retries) is skipped, the arm keeps its last reading and
its servos hold their last goal. BUS_STOP_TICKS failures in a row stop the motion (reason "blocked: bus ..."), and
bus_faults counts those stops for the recorder; BUS_LOST_TICKS in a row is reported as a lost bus.

Reflexes: a joint more than TRACK_LIMIT degrees behind its goal for TRACK_TICKS ticks stops everything (something
blocks the arm); a servo hotter than HOT_C stops everything too.
"""
import threading
import time

import numpy as np

from . import gravity
from .gravity import SagModel
from .motion import TICK

TRACK_LIMIT = 25.0  # deg
TRACK_TICKS = 25
START_TOLERANCE = 4.0  # deg: a trajectory must start where the arm's goal is
SETTLE_DEG = 2.0  # a plan is done when every joint is this close to its last goal
SETTLE_TICKS = 50  # or after this long
SETTLE_FAIL_DEG = 5.0  # still this far off by then: the arm is stuck (blocked), not slow
TEMP_EVERY = 100  # ticks
WARM_C, HOT_C = 55, 62
TEMP_MAX_C = 100  # a reading above this is a garbled packet, not heat: the servo would be dead (31 Oct: 189-252 C)
JUMP_DEG = 25.0  # no servo turns this far in one tick: a reading that jumps this far is garbled
JUMP_TICKS = 3  # unless it keeps reading there
TAKEOVER_DEG = 8.0
RELEASE_S = 2.0
LEADER_STILL_DEG = 1.0  # a leader that stays within this of where it stopped is resting
FOLLOW_DEG_S = 180.0
BUS_STOP_TICKS = 5
BUS_LOST_TICKS = 100
CONTACT_M = 0.004
CONTACT_TICKS = 3
LEARN_AFTER = 15  # ticks an arm must hold still before its sag is learned from it
LEARN_EVERY = 5
LEARN_MIN_Z = 0.03  # m: only from fingertips this high (not resting on the table)
SQUEEZING_MM = 2.0  # the gripper is commanded this much tighter than it is: it's holding something
STREAM_DEG_S = 150.0  # how fast a streamed goal may move the joints
STREAM_MM_S = 120.0  # and the gripper


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
        self.target = {}  # streamed pose targets, while a policy drives
        self.sag = {s: SagModel() for s in ios}
        self.arms_k = {}  # side -> kin.Arm, for the gravity offsets (set_arms)
        self.still = {s: 0 for s in ios}  # ticks each arm's goal hasn't changed
        self.jumps = {}  # side -> readings in a row that jumped impossibly far from the last good one
        self.contact = {}  # side -> where it last touched down (tip and lowest finger heights, m), until it moves
        self.touch = {s: 0 for s in ios}
        self.touch_base = {}
        self.touch_z = {}  # recent measured fingertip heights, per arm
        self.bus_errors = {}  # (arm, call) -> failures in a row: reads, writes and temperatures count apart
        self.bus_faults = 0  # motions stopped by a bus drop-out
        self.leaders = {}  # side -> leader arm io (read only)
        self.leader_ref = {}  # where each leader rested when its follower was last given back
        self.leader_q = {}
        self.manual = {}  # side -> dict(anchor=pose the leader last stopped at, still=ticks since)
        self.takeovers = 0
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

    def set_arms(self, arms, sag_file=None):
        """The arms' kinematics (scene.arms): turns on the gravity feed-forward and the contact watch. sag_file keeps
        the learned sag between runs."""
        self.arms_k = dict(arms)
        self.sag_file = sag_file
        if sag_file:
            self.sag = gravity.load(sag_file, self.ios)
        return self

    def set_leaders(self, leaders):
        """Leader arms (read-only ios, by side) whose movement takes their follower over."""
        self.leaders = dict(leaders)
        return self

    def save_sag(self):
        if getattr(self, "sag_file", None):
            gravity.save(self.sag_file, self.sag)

    def start(self):
        self.thread.start()
        return self

    def command(self, s, goal=None):
        """What is written to arm s's servos for a goal: the goal plus the learned sag offset."""
        g = self.goal[s] if goal is None else goal
        if s not in self.arms_k:
            return g.copy()
        cmd = g.copy()
        cmd[:5] += self.sag[s].offset(self.arms_k[s], g)
        return cmd

    def close(self):
        self.running = False
        self.thread.join(timeout=1)
        self.save_sag()

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
                try:
                    io.torque(True)
                except ConnectionError as e:
                    raise ConnectionError(f"{s} arm: {e}") from e
                self.mode[s] = "hold"
        self.on_event("armed: torque on, holding position")

    def disarm(self):
        with self.lock:
            self._abort("torque off")
            self.target = {}
            self.manual = {}
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
            if self.target:
                return False, "a policy is driving the arms"
            if self.manual:
                return False, f"manual: Vlad has the {next(iter(self.manual))} arm"
            first = trajs[0]
            for s, rows in first.rows.items():
                off = float(np.max(np.abs(rows[0, :5] - self.goal[s][:5])))
                if off > START_TOLERANCE:
                    return False, f"the plan starts {off:.0f} deg from where {s} is"
            self.queue = list(trajs)
            self.k = self.settle = 0
            self.touch = {s: 0 for s in self.touch}
            self.touch_base = {}
            self.touch_z = {}
            self.reason = None
            self.idle.clear()
            for s in first.rows:
                self.mode[s] = "moving"
        while not self.idle.wait(0.05):
            if cancel is not None and cancel.is_set():
                self.stop("stop requested")
        return self.reason is None, self.reason

    def stream(self, targets):
        """Policy mode: each arm in `targets` heads for its pose (5 joints in degrees, gripper in mm), rate-limited.
        Returns False (and changes nothing) when torque is off or a trajectory is playing."""
        with self.lock:
            if not self.armed() or self.queue or self.manual:
                return False
            for s, g in targets.items():
                self.target[s] = np.asarray(g, dtype=float).copy()
                self.mode[s] = "policy"
            return True

    def stop(self, reason="stop requested"):
        """Both arms hold where they are now."""
        with self.lock:
            if self.queue or self.target:
                self.on_event(f"stopped: {reason}", "warn")
            self._abort(reason)
            self.target = {}
            self._hold_all()

    def _hold_all(self):
        """Every powered arm holds where it is, except one Vlad has with its leader (it keeps following)."""
        for s in self.ios:
            if self.mode[s] not in ("off", "manual"):
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
                if not self.queue:
                    time.sleep(0.002)  # idle: leave the CPU to the planner
                continue
            next_t += TICK / (self.speed if self.world is not None else 1.0)
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()

    def _bus(self, s, fn, *args, key=None):
        """One bus call for arm s; counts failures in a row and stops motion when they persist."""
        key = key or (s, fn.__name__)
        try:
            out = fn(*args)
        except Exception as e:
            n = self.bus_errors[key] = self.bus_errors.get(key, 0) + 1
            if n == BUS_STOP_TICKS:
                self.bus_faults += 1
                self.on_event(f"{s} bus not answering ({e}): stopped, holding", "warn")
                self._abort(f"blocked: bus: {s} not answering")
                self.target = {}
                self._hold_all()
            elif n == BUS_LOST_TICKS:
                self.on_event(f"{s} bus lost for {n * TICK:.0f} s: check its USB cable and power", "error")
            return None
        if self.bus_errors.get(key, 0) >= BUS_STOP_TICKS:
            self.on_event(f"{s} bus answering again")
        self.bus_errors[key] = 0
        return out

    def _tick(self):
        with self.lock:
            for s, io in self.ios.items():
                q = self._bus(s, io.read)
                if q is not None and np.max(np.abs(q[:5] - self.q[s][:5])) > JUMP_DEG:
                    self.jumps[s] = self.jumps.get(s, 0) + 1
                    if self.jumps[s] < JUMP_TICKS:
                        q = None
                if q is not None:
                    self.jumps[s] = 0
                    self.q[s] = q
            for s, lio in self.leaders.items():
                lq = self._bus(s, lio.read, key=(s, "leader"))
                if lq is not None:
                    self._leader(s, lq)
            if self.queue:
                traj = self.queue[0]
                n = len(next(iter(traj.rows.values())))
                if self.k < n:
                    for s, rows in traj.rows.items():
                        self.goal[s] = rows[self.k].copy()
                    self.k += 1
                    if traj.contact:
                        self._touched(traj)
                else:
                    self.settle += 1
                    far = max(float(np.max(np.abs(self.goal[s][:5] - self.q[s][:5]))) for s in traj.rows)
                    if self.settle >= SETTLE_TICKS and far > SETTLE_FAIL_DEG:
                        stuck = max(traj.rows, key=lambda s: float(np.max(np.abs(self.goal[s][:5] - self.q[s][:5]))))
                        self.on_event(f"{stuck} stopped {far:.0f} deg short of its goal, holding", "warn")
                        self._abort(f"blocked: {stuck} stopped {far:.0f} deg short of its goal")
                        self._hold_all()
                    elif far < SETTLE_DEG or self.settle >= SETTLE_TICKS or len(self.queue) > 1:
                        self.queue.pop(0)
                        self.k = self.settle = 0
                        if not self.queue:
                            for s in self.mode:
                                if self.mode[s] == "moving":
                                    self.mode[s] = "hold"
                            self.idle.set()
            elif self.target:
                for s, tgt in self.target.items():
                    step = np.full(6, STREAM_DEG_S * TICK)
                    step[5] = STREAM_MM_S * TICK
                    self.goal[s] = self.goal[s] + np.clip(tgt - self.goal[s], -step, step)
            for s in self.manual:  # the follower trails its leader
                tgt = self.leader_q[s].copy()
                if s in self.arms_k:
                    lim = self.arms_k[s].limits
                    tgt[:5] = np.clip(tgt[:5], lim[:, 0], lim[:, 1])
                step = np.full(6, FOLLOW_DEG_S * TICK)
                step[5] = STREAM_MM_S * TICK
                self.goal[s] = self.goal[s] + np.clip(tgt - self.goal[s], -step, step)
            for s, io in self.ios.items():
                if self.mode[s] != "off":
                    self._bus(s, io.write, self.command(s))
            self._reflexes()
            self._learn_sag()
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

    def _leader(self, s, lq):
        """Takeover and give-back for arm s, from its leader's latest reading."""
        self.leader_q[s] = lq
        if self.mode[s] == "off" or s not in self.leader_ref:
            self.leader_ref[s] = lq.copy()
            return
        if s in self.manual:
            m = self.manual[s]
            if np.max(np.abs(lq[:5] - m["anchor"][:5])) > LEADER_STILL_DEG:
                m["anchor"], m["still"] = lq.copy(), 0
            else:
                m["still"] += 1
            if m["still"] * TICK >= RELEASE_S:
                del self.manual[s]
                self.leader_ref[s] = lq.copy()
                self.goal[s] = self.q[s].copy()
                self.mode[s] = "hold"
                self.on_event(f"Vlad gave the {s} arm back")
            return
        if np.max(np.abs(lq[:5] - self.leader_ref[s][:5])) > TAKEOVER_DEG:
            self._abort(f"manual: Vlad took over the {s} arm")
            self.target = {}
            for t in self.ios:
                if self.mode[t] in ("moving", "policy"):
                    self.goal[t] = self.q[t].copy()
                    self.mode[t] = "hold"
            self.mode[s] = "manual"
            self.manual[s] = dict(anchor=lq.copy(), still=0)
            self.takeovers += 1
            self.on_event(f"Vlad took over the {s} arm", "warn")

    def _touched(self, traj):
        """The contact watch, for the arms this trajectory moves. Ends the trajectory on contact."""
        for s, rows in traj.rows.items():
            if s not in self.arms_k or np.allclose(rows[0], rows[-1]):
                continue
            arm = self.arms_k[s]
            z_cmd = arm.tool(self.goal[s])[0][2]
            z_meas = arm.tool(self.q[s])[0][2]
            base = self.touch_base.setdefault(s, z_meas - z_cmd)
            hist = self.touch_z.setdefault(s, [])
            hist.append(z_meas)
            stopped = len(hist) > CONTACT_TICKS and z_meas > hist[-1 - CONTACT_TICKS] - 0.0005  # not going down
            self.touch[s] = self.touch[s] + 1 if z_meas - z_cmd - base > CONTACT_M and stopped else 0
            if self.touch[s] >= CONTACT_TICKS:
                self.touch = {t: 0 for t in self.touch}
                p, R = arm.tool(self.q[s])
                fingers = [p + R @ arm.jaw * self.q[s][5] / 2000 * k for k in (1, -1)]
                self.contact[s] = dict(tip=float(z_meas), finger=float(min(f[2] for f in fingers)))
                self.goal[s] = self.q[s].copy()  # hold where it rests
                self.goal[s][5] = rows[min(self.k, len(rows) - 1)][5]
                self.reason = f"contact: {s}"
                self.queue = []
                self.idle.set()
                for t in self.mode:
                    if self.mode[t] == "moving":
                        self.mode[t] = "hold"
                return

    def _learn_sag(self):
        for s in self.ios:
            moving = self.mode[s] in ("moving", "policy")
            last = getattr(self, "_last_goal", {}).get(s)
            if moving or last is None or not np.allclose(last[:5], self.goal[s][:5]):
                self.still[s] = 0
                if moving:
                    self.contact.pop(s, None)
            else:
                self.still[s] += 1
            holding = self.goal[s][5] < self.q[s][5] - SQUEEZING_MM
            if (self.mode[s] == "hold" and s in self.arms_k and s not in self.contact and not holding
                    and self.still[s] >= LEARN_AFTER and self.still[s] % LEARN_EVERY == 0):
                arm = self.arms_k[s]
                if arm.tool(self.q[s])[0][2] > LEARN_MIN_Z:
                    self.sag[s].learn(arm, self.command(s), self.q[s])
        self._last_goal = {s: v.copy() for s, v in self.goal.items()}

    def _reflexes(self):
        for s in self.ios:
            if self.mode[s] == "off":
                continue
            err = float(np.max(np.abs(self.goal[s][:5] - self.q[s][:5])))
            self.lag[s] = self.lag[s] + 1 if err > TRACK_LIMIT else 0
            if self.lag[s] >= TRACK_TICKS:
                self.lag[s] = 0
                self._abort(f"blocked: {s} is {err:.0f} deg behind its goal")
                self.target = {}
                self.on_event(f"reflex: {s} is {err:.0f} deg behind its goal, holding", "warn")
                self._hold_all()
        if self.ticks % TEMP_EVERY == 0:
            for s, io in self.ios.items():
                temps = self._bus(s, io.temperatures)
                if temps is None or max(temps) > TEMP_MAX_C:
                    continue
                self.temps[s] = temps
                hot = max(temps)
                if hot >= HOT_C and (self.queue or self.target):
                    self._abort(f"hot: {s} servo at {hot:.0f} C")
                    self.target = {}
                    self._hold_all()
                    self.on_event(f"{s} servo at {hot:.0f} C: stopped, let it cool", "warn")
                elif hot >= WARM_C and self.ticks % (TEMP_EVERY * 15) == 0:
                    self.on_event(f"{s} servo at {hot:.0f} C", "warn")
