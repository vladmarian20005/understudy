"""One run from the dashboard: a video and a task in words, taken through the whole loop.

    watch    Muse Spark watches the video and says what the hands did and which objects they touched
    segment  SAM 3.1 segments those objects and the hands      (both: python -m watch.sam_ego4d video, as a process)
    plan     the task the brain gets: the description, with what the video showed (and, after a hand-to-hand pass,
             who gives and who takes, from the score: watch/delegate.py)
    act      the brain drives the arms through robotd, recorded as an episode; every thought, call and image is an event
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
    job.emit("segmented", objects=[{"name": o["name"], "visible": o.get("visible")} for o in objects],
             pass_seen=bool((score.get("quality") or {}).get("pass_seen")), giver=score.get("giver"),
             taker=score.get("taker"), files=files, held=score.get("held", ""))
    return score


# ---- plan --------------------------------------------------------------------------------------------------------------
def plan(job, score):
    """The brain's task and target: the person's description first; what the video showed as context."""
    from watch import delegate

    job.enter("plan", "Turning the video into a task for the arms")
    description = (job.task or "").strip()
    summary = ((score.get("source") or {}).get("muse") or {}).get("summary") or ""
    target, giver, taker = None, score.get("giver"), score.get("taker")
    passed = bool((score.get("quality") or {}).get("pass_seen")) and giver in ("left", "right") and \
        taker in ("left", "right") and giver != taker
    if passed:
        target = delegate.target_for(score)
    if description:
        task = description
        if passed:
            task += f" (In the video the object went from the {giver} hand to the {taker} hand.)"
    elif passed:
        task = delegate.task_for(score, target)
    elif summary:
        task = f"Do with the arms what the person in the video did: {summary}"
    else:
        raise StageError("no description, and the video didn't show a hand-to-hand pass to copy")
    if summary and description:
        task += f"\n\nWhat the video shows: {summary}"
    job.emit("plan", task=task, target=target, giver=giver if passed else None, taker=taker if passed else None)
    return dict(task=task, target=target, giver=giver if passed else None, taker=taker if passed else None,
                score=score)


# ---- act -----------------------------------------------------------------------------------------------------------------
def first_json(text):
    try:
        return json.loads(text.splitlines()[0])
    except (ValueError, IndexError):
        return None


def act(job, p, backend, max_turns=40, record=True):
    from brain import agent
    from harness import robot

    from .thinking import Narrated

    job.enter("act", "Muse Spark is driving the arms")
    try:
        st = robot.status()
    except robot.RobotdDown:
        raise StageError("robotd isn't running: start it with python -m harness.robotd --sim (or the real arms)")
    if not st.get("armed"):
        raise StageError("the arms are off: switch torque on in robotd")
    ep, sidecar = None, None
    if record:
        score = p["score"]
        source = {"kind": "agent", "model": getattr(backend, "model", ""), "ui": {"job": job.id, "video": job.video_name},
                  "score": {"source": (score.get("source") or {}).get("kind"), "giver": p["giver"],
                            "taker": p["taker"], "handover": score.get("handover")}}
        begin = robot.episode_begin(p["task"], source, giver=p["giver"], taker=p["taker"], target_cm=p["target"])
        if begin.get("ok"):
            ep = job.episode = begin["episode"]
            job.emit("recording", episode=ep)
        else:
            job.emit("warning", text="not recorded: " + "; ".join(begin.get("errors", [])))
    run_dir = os.path.join(job.dir, "brain")
    shown = set()

    def new_images():
        if not os.path.isdir(run_dir):
            return
        for name in sorted(os.listdir(run_dir)):
            if name.endswith(".jpg") and name not in shown:
                shown.add(name)
                turn, cam = name[:-4].split("_", 1)
                job.emit("seen", turn=int(turn), camera=cam, url=job.url(os.path.join(run_dir, name)))

    def on_reply(turn, reply, reasoning):
        new_images()
        job.emit("thought", turn=turn, text=reply.text.strip(), reasoning=reasoning,
                 calls=[c.name for c in reply.calls], tokens_in=reply.tokens_in, tokens_out=reply.tokens_out,
                 seconds=round(reply.seconds, 1))
        if job.stop_requested:
            robot.stop("stopped from the dashboard")

    def on_log(line):
        new_images()
        if line.startswith("  -> "):
            name, _, args = line[5:].partition(" ")
            try:
                args = json.loads(args)
            except ValueError:
                pass
            job.emit("call", name=name, args=args)
        elif line.startswith("     "):
            text = line.strip()
            r = first_json(text) or {}
            job.emit("result", text=text, outcome=r.get("outcome"), fail=r.get("fail"), why=r.get("why"))

    model = Narrated(backend, on_reply)
    out = None
    try:
        out = agent.run(p["task"], model, max_turns, log=on_log, run_dir=run_dir)
    finally:
        new_images()
        if ep is not None:
            ok = bool(out and out["success"])
            note = None if ok else f"the brain says: {out['summary'] if out else 'the run crashed'}"
            try:
                end = robot.episode_end(valid=True, note=note)
                sidecar = end.get("sidecar")
                if out is not None:
                    robot.episode_label(ep, "agents", {k: v for k, v in out.items() if k != "transcript"})
                job.emit("recorded", episode=ep, frames=end.get("frames"), duration_s=end.get("duration_s"))
            except robot.RobotdDown as e:
                job.emit("warning", text=f"episode {ep} not ended: {e}")
    job.emit("acted", success=out["success"], summary=out["summary"], turns=out["turns"],
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
    def __init__(self, root, backend_factory=None, watch=run_watch, max_turns=40, record=True):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.backend_factory = backend_factory
        self.watch = watch
        self.max_turns = max_turns
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
            score = watch_and_segment(job, self.watch)
            p = plan(job, score)
            backend = self.backend()
            out, sidecar = act(job, p, backend, self.max_turns, self.record)
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
