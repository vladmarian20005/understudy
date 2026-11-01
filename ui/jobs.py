"""One run from the dashboard: a video and a task in words, taken through the whole loop.

    watch    Muse Spark watches the video and says what the hands did and which objects they touched
    segment  SAM 3.1 segments those objects and the hands      (both: python -m watch.sam_ego4d video, as a process)
    plan     what the brain gets: the description (else: do what the video shows), the video and Muse's reading of it
             (and, after a hand-to-hand pass, who gives and who takes, from the score: watch/delegate.py)
    act      the brain (brain/plan.py) looks, plans once, runs the steps on the arms through robotd and checks where it
             matters, recorded as an episode; every answer, step and image is an event
    judge    the judge looks at the recorded episode and gives its verdict; robotd decides whether it trains

Every step reports events (job.emit), kept in ui_runs/<id>/events.jsonl, so the page can follow a run live or replay
it after a reload. One run at a time: there's one pair of arms.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = ["watch", "segment", "plan", "act", "judge"]
MAX_VIDEO_BYTES = 400 * 1024 * 1024
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


class Stopped(Exception):
    pass


class StageError(Exception):
    pass


class Job:
    def __init__(self, root, video_name, task, job_id=None):
        self.id = job_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.dir = os.path.join(root, self.id)
        os.makedirs(self.dir, exist_ok=True)
        self.video_name = video_name
        self.task = task
        self.created = time.time()
        self.stage = None
        self.status = "running"  # running | done | failed | stopped
        self.events = []
        self.cond = threading.Condition()
        self.stop_requested = False
        self.episode = None

    @property
    def video_path(self):
        ext = os.path.splitext(self.video_name)[1].lower() or ".mp4"
        return os.path.join(self.dir, "upload" + ext)

    def emit(self, kind, **data):
        e = dict(i=len(self.events), t=round(time.time(), 3), stage=self.stage, kind=kind, **data)
        with self.cond:
            self.events.append(e)
            self.cond.notify_all()
        with open(os.path.join(self.dir, "events.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(e, default=str) + "\n")
        return e

    def enter(self, stage, text):
        if self.stop_requested:
            raise Stopped()
        self.stage = stage
        self.emit("stage", state="running", text=text)

    def finish(self, status):
        self.stage = None
        self.emit("end", status=status)  # before the status changes: a stream that sees the end has sent "end"
        self.status = status
        with self.cond:
            self.cond.notify_all()
        self.save()

    def save(self):
        json.dump(self.summary(), open(os.path.join(self.dir, "job.json"), "w"), indent=1)

    def summary(self):
        return dict(id=self.id, video=self.video_name, task=self.task, created=self.created, status=self.status,
                    stage=self.stage, episode=self.episode, events=len(self.events))

    def url(self, path):
        return f"/api/jobs/{self.id}/files/{os.path.relpath(path, self.dir).replace(os.sep, '/')}"

    def wait(self, after, timeout=15.0):
        """Events after index `after` once there are some (or the run ended, or timeout)."""
        with self.cond:
            self.cond.wait_for(lambda: len(self.events) > after + 1 or self.status != "running", timeout=timeout)
            return self.events[after + 1:]

    @classmethod
    def load(cls, root, job_id):
        """A finished run from disk, for replaying it."""
        d = os.path.join(root, job_id)
        meta = json.load(open(os.path.join(d, "job.json")))
        job = cls.__new__(cls)
        job.id, job.dir, job.video_name, job.task = job_id, d, meta["video"], meta["task"]
        job.created, job.status, job.stage, job.episode = meta["created"], meta["status"], None, meta.get("episode")
        job.cond, job.stop_requested = threading.Condition(), False
        path = os.path.join(d, "events.jsonl")
        job.events = [json.loads(line) for line in open(path, encoding="utf-8")] if os.path.exists(path) else []
        if job.status == "running":  # the server stopped mid-run
            job.status = "stopped"
        return job


# ---- watch + segment: the SAM pipeline's video mode, as a process -----------------------------------------------------
LINE_STAGE = [(re.compile(r"Muse '|Muse \(cached\)"), "watch"), (re.compile(r"SAM '|sam_|segmenting"), "segment")]


def run_watch(job, log_line=None):
    """python -m watch.sam_ego4d video on the upload; returns the folder with score.json, muse.json, the clips."""
    out = os.path.join(job.dir, "sam")
    cmd = [sys.executable, "-m", "watch.sam_ego4d", "--out", out, "video", "--video", job.video_path]
    p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONUNBUFFERED="1"))
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail = (tail + [line])[-12:]
        if job.stop_requested:
            p.kill()
            raise Stopped()
        (log_line or (lambda s: None))(line)
    if p.wait() != 0:
        raise StageError("the SAM pipeline stopped: " + " | ".join(tail[-4:]))
    stem = os.path.splitext(os.path.basename(job.video_path))[0]
    return os.path.join(out, "videos", stem)


def watch_and_segment(job, run=run_watch):
    job.enter("watch", "Muse Spark is watching the video")
    reported = {"segment": False}

    def on_line(line):
        job.emit("log", text=line)
        m = re.match(r"\s*summary:\s*(.*)", line)
        if m:
            job.emit("summary", text=m.group(1))
        m = re.match(r"\s*objects:\s*(.*)", line)
        if m:
            job.emit("objects", names=[o.strip() for o in m.group(1).split(",") if o.strip() and o.strip() != "(none)"])
        m = re.match(r"\s*SAM '([^']+)'", line)
        if m:
            if not reported["segment"]:
                reported["segment"] = True
                job.enter("segment", "SAM 3.1 is segmenting what Muse named")
            job.emit("segmenting", concept=m.group(1))

    folder = run(job, on_line)
    if not reported["segment"]:  # every SAM result came from the cache
        job.enter("segment", "SAM 3.1 segmentation (cached)")
    score = json.load(open(os.path.join(folder, "score.json")))
    files = {name: job.url(os.path.join(folder, name)) for name in ("clip.mp4", "preview.mp4")
             if os.path.exists(os.path.join(folder, name))}
    objects = (score.get("vision") or {}).get("objects") or []
    job.emit("segmented", objects=[{k: o.get(k) for k in ("name", "visible", "instances", "color", "muse_name")}
                                   for o in objects],
             pass_seen=bool((score.get("quality") or {}).get("pass_seen")), giver=score.get("giver"),
             taker=score.get("taker"), files=files, held=score.get("held", ""))
    return score, folder


# ---- plan --------------------------------------------------------------------------------------------------------------
def plan(job, score, folder=None):
    """What the brain is given: the person's description (else: do what the video shows), the video itself (the
    brain's planner reads Muse's summary of it and sees frames from it), and after a hand-to-hand pass who gives and
    who takes, from the score (watch/delegate.py)."""
    from watch import delegate

    job.enter("plan", "Turning the video into a task for the arms")
    description = (job.task or "").strip()
    muse = (score.get("source") or {}).get("muse") or {}
    summary = muse.get("summary") or ""
    target, giver, taker = None, score.get("giver"), score.get("taker")
    passed = bool((score.get("quality") or {}).get("pass_seen")) and giver in ("left", "right") and \
        taker in ("left", "right") and giver != taker
    if passed:
        target = delegate.target_for(score)
    if not description and not summary:
        raise StageError("no description, and Muse couldn't say what the video shows")
    task = description
    if description and passed:
        task += f" (In the video the object went from the {giver} hand to the {taker} hand.)"
    clip = os.path.join(folder, "clip.mp4") if folder else None
    job.emit("plan", task=task or "Do with the two arms what the person in the video does.", video_summary=summary,
             target=target, giver=giver if passed else None, taker=taker if passed else None)
    return dict(task=task, target=target, giver=giver if passed else None, taker=taker if passed else None,
                score=score, video=clip if clip and os.path.exists(clip) else None, objects=muse.get("objects") or [])


# ---- act -----------------------------------------------------------------------------------------------------------------
SIM_HINT = "In the simulator, start robotd with the scene that has them: python -m harness.robotd --sim --hd --objects balls"
GENERIC = {"object", "thing", "item", "held", "hand", "in", "tool", "small", "round"}


def words(name):
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", str(name).lower())} - GENERIC


def on_table(wanted, found):
    """Which of the video's objects SAM found on the robot's table (a 'tennis ball' is a 'ball')."""
    same = lambda a, b: any(x in y or y in x for x in words(a) for y in words(b))
    return [w for w in wanted if any(same(w, f) for f in found)]


def action_text(a):
    a = a or {}
    do = a.get("do", "?")
    xyz = lambda v: "(" + ", ".join(f"{float(x):g}" for x in v) + ")"
    if do == "grasp":
        return f"grasps the {a.get('object', '?')}"
    if do == "move":
        if a.get("above"):
            return f"moves above the {a['above']}" + (f" by {a['by']:g} cm" if isinstance(a.get("by"), (int, float)) else "")
        if a.get("to"):
            return f"moves to {xyz(a['to'])}"
        if a.get("rel"):
            return f"shifts by {xyz(a['rel'])}"
        return "moves"
    if do == "tilt":
        return f"tilts {a.get('deg', 0):g}°"
    if do == "nudge":
        bits = [f"{k} {a[k]:g}" for k in ("forward", "right", "up") if isinstance(a.get(k), (int, float)) and a[k]]
        return "nudges " + (", ".join(bits) or "a little")
    return {"open": "lets go", "grip": "grips", "park": "parks out of view"}.get(do, do)


def step_text(step):
    arms = [f"{side.title()} {action_text(step[side])}" for side in ("left", "right") if step.get(side)]
    return " · ".join(arms) or "nothing"


def say(out):
    """One of Muse's JSON answers, as what it'd say."""
    if "find" in out:
        return "Looking on the table for: " + ", ".join(out["find"] or ["(nothing)"]) + "."
    if "steps" in out:
        lines = [out.get("understanding", "").strip(), ""]
        lines += [f"{k}. {step_text(st)}" + (f" — check: {st['check']}" if st.get("check") else "")
                  for k, st in enumerate(out.get("steps") or [], 1)]
        if out.get("done_when"):
            lines += ["", f"Done when: {out['done_when']}"]
        return "\n".join(lines).strip()
    text = ("Looks right. " if out.get("ok") else "Not right yet. ") + str(out.get("why", "")).strip()
    if out.get("done"):
        text += f"\n{'Finished' if out['done'].get('success') else 'Giving up'}: {out['done'].get('summary', '')}"
    elif out.get("next"):
        text += "\n" + ("Re-planning: " if out.get("replace_rest") else "First: ") + \
            "; ".join(step_text(st) for st in out["next"])
    return text


def narrated_run(job, backend, run_dir, wanted):
    """brain.plan.Run, with every answer from Muse, every step and every image as an event."""
    from brain import plan as planner
    from brain import tools
    from harness import robot

    class Narrated(planner.Run):
        def __init__(self):
            super().__init__(backend, log=self.on_log, run_dir=run_dir)
            self.pending, self.looks = [], 0

        def on_log(self, line):
            job.emit("log", text=line)

        def look(self, sides=(), tag=None):
            self.looks += 1
            tag = tag or f"look{self.looks:02d}"
            seen = super().look(sides, tag)
            self.pending += [(cam, os.path.join(run_dir, f"{tag}_{cam}.jpg")) for cam in seen.get("images", {})]
            return seen

        def ask(self, content, effort=None, keep_images=1):
            if job.stop_requested:
                raise Stopped()
            if any(isinstance(p, dict) and p.get("text") == planner.PLAN_ASK for p in content):
                found = [o.get("kind") or o["name"] for o in tools._memory["objects"]]
                if wanted and not on_table(wanted, found):  # nothing from the video is there: don't flail
                    raise StageError(f"The robot's table has none of what the video shows ({', '.join(wanted)}): "
                                     f"SAM found {', '.join(found) or 'nothing'}. {SIM_HINT}")
            if hasattr(self.llm, "last_reasoning"):
                self.llm.last_reasoning = None
            t0, before = time.time(), dict(self.stats)
            out = super().ask(content, effort, keep_images)
            job.emit("thought", turn=self.calls, text=say(out), reasoning=getattr(self.llm, "last_reasoning", None),
                     calls=["done"] if out.get("done") else [], seconds=round(time.time() - t0, 1),
                     tokens_in=self.stats["tokens_in"] - before["tokens_in"],
                     tokens_out=self.stats["tokens_out"] - before["tokens_out"])
            for cam, path in self.pending:
                if os.path.exists(path):
                    job.emit("seen", turn=self.calls, camera=cam, url=job.url(path))
            self.pending = []
            return out

        def starting(self, n, step):
            if job.stop_requested:
                robot.stop("stopped from the dashboard")
            job.emit("call", name=f"{n}. {step_text(step)}", args=None, step=step)

        def save(self, **rec):
            super().save(**rec)
            if "step" in rec:
                res = rec.get("results") or {}
                failed = [f"{side}: {r['failed']}" for side, r in res.items() if isinstance(r, dict) and "failed" in r]
                job.emit("result", outcome="failed" if failed else "done", why="; ".join(failed) or None,
                         results=res, seconds=rec.get("seconds"))

    return Narrated()


def act(job, p, backend, max_steps=None, record=True):
    from brain import plan as planner
    from brain import tools
    from harness import robot

    job.enter("act", "Muse Spark plans, the arms act, Muse checks")
    try:
        st = robot.status()
    except robot.RobotdDown:
        raise StageError("robotd isn't running: start it with python -m harness.robotd --sim --hd --objects balls "
                         "(or on the real arms)")
    if not st.get("armed"):
        raise StageError("the arms are off: switch torque on in robotd")
    ep, sidecar = None, None
    if record:
        score = p["score"]
        source = {"kind": "agent", "model": getattr(backend, "model", ""), "ui": {"job": job.id, "video": job.video_name},
                  "score": {"source": (score.get("source") or {}).get("kind"), "giver": p["giver"],
                            "taker": p["taker"], "handover": score.get("handover")}}
        begin = robot.episode_begin(p["task"] or "the task in the video", source, giver=p["giver"],
                                    taker=p["taker"], target_cm=p["target"])
        if begin.get("ok"):
            ep = job.episode = begin["episode"]
            job.emit("recording", episode=ep)
        else:
            job.emit("warning", text="not recorded: " + "; ".join(begin.get("errors", [])))
    run = narrated_run(job, backend, os.path.join(job.dir, "brain"), p.get("objects") or [])
    out, t0 = None, time.time()
    try:
        try:
            out = run.go(p["task"], p.get("video"), max_steps or planner.MAX_STEPS)
        except robot.RobotdDown as e:
            out = run.end(False, f"robotd stopped answering: {e}", t0)
    finally:
        tools.FIND = []  # the planner's SAM names stay set otherwise: the next look in this process would use them
        if ep is not None:
            ok = bool(out and out["success"])
            note = None if ok else f"the brain says: {out['summary'] if out else 'the run stopped'}"
            try:
                end = robot.episode_end(valid=True, note=note)
                sidecar = end.get("sidecar")
                if out is not None:
                    robot.episode_label(ep, "agents", {k: v for k, v in out.items() if k != "record"})
                job.emit("recorded", episode=ep, frames=end.get("frames"), duration_s=end.get("duration_s"))
            except robot.RobotdDown as e:
                job.emit("warning", text=f"episode {ep} not ended: {e}")
    job.emit("acted", success=out["success"], summary=out["summary"], turns=out["calls"], steps=out["steps"],
             cost=out["cost"], seconds=out["seconds"], llm_seconds=out["llm_seconds"])
    return out, sidecar


# ---- judge ---------------------------------------------------------------------------------------------------------------
def judge_episode(job, sidecar, backend):
    import cv2

    from harness import robot
    from harness.scene import Scene
    from judge import frames as fr
    from judge.judge import judge, pick_frames

    job.enter("judge", "The judge is checking the recorded episode")
    label = json.load(open(sidecar))
    root = os.path.dirname(os.path.dirname(sidecar))
    ep = int(label["episode"])
    picks = pick_frames(label)
    raw = fr.read(root, ep, [i for i, _ in picks])
    fps = float(label.get("fps") or 30)
    target = (label.get("robot") or {}).get("target_cm")
    scene = Scene()
    images, urls = [], []
    for (i, why), img in zip(picks, raw):
        a = fr.annotate(img, scene, target, f"frame {i}  t={i / fps:.1f}s  {why}")
        images.append((i, why, a))
        path = os.path.join(job.dir, "judge", f"frame_{i:05d}.jpg")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, a)
        urls.append({"url": job.url(path), "frame": i, "why": why})
    verdict, cost = judge(label, images, backend, log=lambda s: None)
    final = robot.episode_label(ep, "judge", verdict).get("final")
    job.emit("verdict", success=verdict["success"], confidence=verdict["confidence"], reason=verdict["reason"],
             failure=verdict["failure"], frames=urls, final=final, cost=round(cost, 4))
    return verdict


# ---- the whole run -------------------------------------------------------------------------------------------------------
class Runner:
    def __init__(self, root, backend_factory=None, watch=run_watch, max_steps=None, record=True):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.backend_factory = backend_factory
        self.watch = watch
        self.max_steps = max_steps
        self.record = record
        self.current = None
        self.lock = threading.Lock()

    def backend(self):
        if self.backend_factory is not None:
            return self.backend_factory()
        from brain import llm

        from .thinking import with_reasoning
        return with_reasoning(llm.backend())

    def busy(self):
        return self.current is not None and self.current.status == "running"

    def start(self, data, filename, task):
        ext = os.path.splitext(filename or "")[1].lower()
        if ext not in VIDEO_EXTS:
            raise ValueError(f"not a video file: {filename!r}")
        if len(data) > MAX_VIDEO_BYTES:
            raise ValueError("the video is over 400 MB: trim it to the task")
        with self.lock:
            if self.busy():
                raise RuntimeError("a run is still going: stop it first")
            job = Job(self.root, os.path.basename(filename), task)
            with open(job.video_path, "wb") as f:
                f.write(data)
            job.emit("start", video=job.url(job.video_path), name=job.video_name, task=task)
            job.save()
            self.current = job
        threading.Thread(target=self._run, args=(job,), daemon=True, name=f"job-{job.id}").start()
        return job

    def stop(self):
        job = self.current
        if job is None or job.status != "running":
            return False
        job.stop_requested = True
        try:
            from harness import robot
            robot.stop("stopped from the dashboard")
        except Exception:
            pass
        return True

    def _run(self, job):
        try:
            score, folder = watch_and_segment(job, self.watch)
            p = plan(job, score, folder)
            backend = self.backend()
            out, sidecar = act(job, p, backend, self.max_steps, self.record)
            if sidecar and not job.stop_requested:
                try:
                    judge_episode(job, sidecar, self.backend())
                except Exception as e:  # the run itself happened: a judge problem doesn't fail it
                    job.emit("error", text=f"the judge couldn't run: {type(e).__name__}: {e}")
            elif not sidecar:
                job.emit("note", text="not recorded, so there's nothing to judge")
            job.finish("stopped" if job.stop_requested else "done")
        except Stopped:
            job.finish("stopped")
        except StageError as e:
            job.emit("error", text=str(e))
            job.finish("failed")
        except Exception as e:
            job.emit("error", text=f"{type(e).__name__}: {e}")
            job.finish("failed")

    def get(self, job_id):
        if self.current is not None and self.current.id == job_id:
            return self.current
        if not re.fullmatch(r"[\w-]+", job_id or "") or not os.path.exists(os.path.join(self.root, job_id, "job.json")):
            return None
        return Job.load(self.root, job_id)

    def history(self, n=20):
        out = []
        for d in sorted(os.listdir(self.root), reverse=True):
            path = os.path.join(self.root, d, "job.json")
            if os.path.exists(path):
                out.append(json.load(open(path)))
            if len(out) >= n:
                break
        if self.current is not None and out and out[0]["id"] == self.current.id:
            out[0] = self.current.summary()
        return out
