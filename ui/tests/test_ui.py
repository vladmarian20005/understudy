"""The dashboard: a run from upload to the end against the simulator (with stand-ins for the SAM pipeline, the brain's
SAM and Muse), what the planner's answers and steps become as events, the task it's given, stopping when the table
lacks the video's objects, reasoning caught from the API, and the HTTP side: upload, the event stream, files, the
cameras."""
import json
import os
import socket
import threading
import time
import urllib.request
from types import SimpleNamespace

import pytest

from brain.llm import Reply, Scripted
from harness import robot
from harness.robotd import Core, serve as serve_robotd
from ui import jobs
from ui.server import serve
from ui.thinking import Narrated, reasoning_of, with_reasoning

SCORE = {"version": 1, "kind": "score", "task": "x", "object": {"name": "paintbrush"},
         "source": {"kind": "video", "muse": {"summary": "left hand passes the paintbrush to the right hand",
                                         "objects": ["paintbrush"]}},
         "giver": "left", "taker": "right", "handover": {"t": 1.0, "t_end": 1.4, "both_s": 0.4},
         "target_cm": None, "quality": {"pass_seen": True},
         "vision": {"objects": [{"name": "paintbrush", "visible": 0.9}, {"name": "left hand", "visible": 1.0}]},
         "held": "LLBBRR"}


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---- the task the brain gets ---------------------------------------------------------------------------------------
def job(tmp_path, task):
    return jobs.Job(str(tmp_path), "v.mp4", task)


def test_the_description_is_the_task_with_the_video_beside_it(tmp_path):
    folder = tmp_path / "sam"
    folder.mkdir()
    (folder / "clip.mp4").write_bytes(b"\0")
    p = jobs.plan(job(tmp_path, "Hand the brush to the right arm."), SCORE, str(folder))
    assert p["task"] == "Hand the brush to the right arm. (In the video the object went from the left hand to the right hand.)"
    assert p["video"] == str(folder / "clip.mp4") and p["objects"] == ["paintbrush"]
    assert p["target"] == [12.0, 24.0] and p["giver"] == "left" and p["taker"] == "right"


def test_without_a_description_the_brain_does_what_the_video_shows_and_with_neither_it_stops(tmp_path):
    no_pass = dict(SCORE, quality={"pass_seen": False}, giver=None, taker=None)
    p = jobs.plan(job(tmp_path, ""), no_pass)
    assert p["task"] == "" and p["video"] is None and p["target"] is None  # brain/plan.py words it from the video
    with pytest.raises(jobs.StageError, match="no description"):
        jobs.plan(job(tmp_path, ""), dict(no_pass, source={"kind": "video"}))


def test_the_videos_objects_are_matched_to_the_tables_loosely():
    assert jobs.on_table(["tube", "ball"], ["tennis ball", "cylinder"]) == ["ball"]
    assert jobs.on_table(["paintbrush"], ["brush"]) == ["paintbrush"]
    assert jobs.on_table(["tennis ball can", "tennis ball"], ["brush", "box"]) == []


def test_muses_answers_read_as_words():
    plan = {"understanding": "Put each ball in the tube.", "steps": [
        {"left": {"do": "grasp", "object": "ball 1"}, "right": {"do": "park"}, "check": "left holds ball 1"},
        {"left": {"do": "move", "above": "tube", "by": 3}}], "done_when": "all balls in the tube"}
    text = jobs.say(plan)
    assert text.startswith("Put each ball in the tube.\n\n1. Left grasps the ball 1 · Right parks out of view")
    assert "check: left holds ball 1" in text and "2. Left moves above the tube by 3 cm" in text
    assert jobs.say({"find": ["ball", "tube"]}) == "Looking on the table for: ball, tube."
    assert jobs.say({"ok": False, "why": "it slipped", "next": [{"left": {"do": "open"}}]}) == \
        "Not right yet. it slipped\nFirst: Left lets go"


# ---- reasoning, narrated replies --------------------------------------------------------------------------------------
def response(**message):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(model_extra={}, **message))])


def test_reasoning_is_caught_whichever_field_the_api_uses():
    assert reasoning_of(response(reasoning_content="  the brush is left  ")) == "the brush is left"
    r = response()
    r.choices[0].message.model_extra = {"reasoning": {"summary": "short plan"}}
    assert reasoning_of(r) == "short plan"
    assert reasoning_of(response()) is None and reasoning_of(object()) is None


def test_with_reasoning_wraps_the_client_and_narrated_reports_each_turn():
    completions = SimpleNamespace(create=lambda **kw: response(reasoning_content="thinking hard"))
    backend = SimpleNamespace(client=SimpleNamespace(chat=SimpleNamespace(completions=completions)), model="m")

    def chat(system, messages, tools):
        backend.client.chat.completions.create()
        return Reply("I'll open the jaws.", [], {"role": "assistant", "content": ""}, model="m")

    backend.chat = chat
    with_reasoning(backend)
    seen = []
    n = Narrated(backend, lambda turn, reply, reasoning: seen.append((turn, reply.text, reasoning)))
    n.chat("s", [], [])
    n.chat("s", [], [])
    assert seen == [(1, "I'll open the jaws.", "thinking hard"), (2, "I'll open the jaws.", "thinking hard")]
    assert n.model == "m" and with_reasoning(Scripted([])).model == "scripted"  # no client: left as it is


# ---- a whole run, against the simulator ----------------------------------------------------------------------------
@pytest.fixture
def sim(monkeypatch):
    port = free_port()
    core = Core(sim_mode=True, speed=0, quiet=True, dataset=None)
    stop = threading.Event()
    threading.Thread(target=serve_robotd, args=(core, port, stop), daemon=True).start()
    monkeypatch.setattr(robot, "PORT", port)
    yield core
    stop.set()
    core.close()


def fake_watch(job, log_line=None, objects=("paintbrush",)):
    """Stands in for the SAM pipeline: a few of its lines, then a score, Muse's reading and a clip."""
    folder = os.path.join(job.dir, "sam", "videos", "upload")
    os.makedirs(folder, exist_ok=True)
    for line in ["  Muse (cached): 1 object(s)", "  summary: left hand passes the paintbrush to the right hand",
                 "  objects: " + ", ".join(objects), "  SAM 'paintbrush' on upload (4.0s, budget used 0/3600s)"]:
        log_line(line)
    summary = SCORE["source"]["muse"]["summary"]
    json.dump(dict(SCORE, source=dict(kind="video", muse=dict(summary=summary, objects=list(objects)))),
              open(os.path.join(folder, "score.json"), "w"))
    json.dump(dict(summary=summary, objects=list(objects)), open(os.path.join(folder, "muse.json"), "w"))
    open(os.path.join(folder, "preview.mp4"), "wb").write(b"\0" * 64)
    return folder


@pytest.fixture
def colour_sam(monkeypatch):
    """The brain's SAM, stood in for by robotd's colour detector: no API key needed."""
    from brain import sam

    monkeypatch.setattr(sam, "objects", lambda scene, img, names, q, held_px=(): [
        dict(o, found_as=o["name"], shape="flat") for o in robot.detect()["objects"] if not o.get("held_by")])


def planner(answers):
    """A scripted Muse for brain/plan.py: its JSON answers in order, each with a bit of reasoning."""
    state = {"n": 0}

    def ask(system, messages, effort=None):
        text = json.dumps(answers[min(state["n"], len(answers) - 1)])
        state["n"] += 1
        backend.last_reasoning = f"thinking about answer {state['n']}"
        return Reply(text, [], dict(role="assistant", content=text), 100, 10, 0.01, "t")

    backend = SimpleNamespace(ask=ask, model="scripted-planner", last_reasoning=None)
    return backend


GRIP_PLAN = [{"find": ["brush"]},
             {"understanding": "Grip the brush with the left arm.",
              "steps": [{"left": {"do": "grasp", "object": "brush"}, "check": "the left arm holds the brush"}],
              "done_when": "the left arm holds the brush"},
             {"ok": True, "why": "the brush is in the left jaws", "done": {"success": True, "summary": "gripped it"}}]


def wait(j, s=120):
    deadline = time.time() + s
    while j.status == "running" and time.time() < deadline:
        time.sleep(0.05)


def test_a_run_goes_from_upload_to_the_end_as_events(sim, colour_sam, tmp_path):
    runner = jobs.Runner(str(tmp_path), backend_factory=lambda: planner(GRIP_PLAN), watch=fake_watch, record=False)
    j = runner.start(b"\0" * 1000, "clip.mov", "Grip the brush with the left arm.")
    with pytest.raises(RuntimeError, match="still going"):
        runner.start(b"\0", "again.mp4", "")
    wait(j)
    kinds = [e["kind"] for e in j.events]
    assert j.status == "done", [e for e in j.events if e["kind"] == "error"]
    assert [e["stage"] for e in j.events if e["kind"] == "stage"] == ["watch", "segment", "plan", "act"]
    assert {"summary", "objects", "segmenting", "segmented", "plan", "thought", "call", "result", "seen",
            "acted", "note", "end"} <= set(kinds)
    thoughts = [e for e in j.events if e["kind"] == "thought"]
    assert [t["text"].split("\n")[0] for t in thoughts] == [
        "Looking on the table for: brush.", "Grip the brush with the left arm.",
        "Looks right. the brush is in the left jaws"]
    assert thoughts[1]["reasoning"] == "thinking about answer 2" and thoughts[2]["calls"] == ["done"]
    assert [e["name"] for e in j.events if e["kind"] == "call"] == ["1. Left grasps the brush"]
    assert [e["outcome"] for e in j.events if e["kind"] == "result"] in (["done"], ["failed"])
    acted = next(e for e in j.events if e["kind"] == "acted")
    assert acted["success"] and acted["steps"] == 1 and acted["turns"] == 3
    seg = next(e for e in j.events if e["kind"] == "segmented")
    assert seg["pass_seen"] and seg["files"]["preview.mp4"].endswith("/files/sam/videos/upload/preview.mp4")
    # what Muse looked at is shown under the answer it gave
    assert any(e["kind"] == "seen" and e["camera"] == "overhead" and e["turn"] == 2 for e in j.events)
    # replayed from disk, the run is the same
    again = jobs.Job.load(str(tmp_path), j.id)
    assert [e["kind"] for e in again.events] == kinds and again.status == "done"
    assert runner.history()[0]["id"] == j.id


def test_a_table_without_the_videos_objects_stops_before_planning(sim, colour_sam, tmp_path):
    watch = lambda job, log_line=None: fake_watch(job, log_line, objects=("tube", "ball"))
    runner = jobs.Runner(str(tmp_path), backend_factory=lambda: planner(GRIP_PLAN), watch=watch, record=False)
    j = runner.start(b"\0", "balls.mp4", "Put each ball in the tube.")
    wait(j)
    assert j.status == "failed"
    error = next(e["text"] for e in j.events if e["kind"] == "error")
    assert "none of what the video shows (tube, ball)" in error and "SAM found brush" in error
    assert "--hd --objects balls" in error
    assert not [e for e in j.events if e["kind"] == "call"]  # no step ran


def test_without_robotd_the_run_stops_at_act_with_a_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(robot, "PORT", free_port())  # nothing there
    runner = jobs.Runner(str(tmp_path), backend_factory=lambda: planner(GRIP_PLAN), watch=fake_watch, record=False)
    j = runner.start(b"\0", "v.mp4", "do it")
    wait(j, 60)
    assert j.status == "failed"
    assert "robotd isn't running" in next(e["text"] for e in j.events if e["kind"] == "error")


def test_only_videos_are_taken(tmp_path):
    runner = jobs.Runner(str(tmp_path), watch=fake_watch)
    with pytest.raises(ValueError, match="not a video"):
        runner.start(b"\0", "notes.txt", "")


# ---- over HTTP --------------------------------------------------------------------------------------------------------
class FakeLive:
    running = True
    frames = {"overhead": b"\xff\xd8jpeg\xff\xd9"}

    def wait_frame(self, name, after, timeout=2.0):
        time.sleep(0.02)
        return after + 1, self.frames[name]

    def snapshot(self):
        return {"robotd": False}


def test_the_server_takes_an_upload_streams_its_events_and_serves_its_files(sim, colour_sam, tmp_path):
    runner = jobs.Runner(str(tmp_path), backend_factory=lambda: planner(GRIP_PLAN), watch=fake_watch, record=False)
    port = free_port()
    httpd = serve(FakeLive(), runner, "127.0.0.1", port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        req = urllib.request.Request(f"{base}/api/jobs?name=clip.mp4&task=Grip%20it", data=b"\0" * 2048, method="POST")
        created = json.loads(urllib.request.urlopen(req).read())
        assert created["task"] == "Grip it"
        # the event stream runs until the end of the run, then closes
        body = urllib.request.urlopen(f"{base}/api/jobs/{created['id']}/events?since=-1", timeout=60).read().decode()
        events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
        assert events[0]["kind"] == "start" and events[-1]["kind"] == "end" and events[-1]["status"] == "done"
        assert [e["i"] for e in events] == list(range(len(events)))
        # resuming after event 3 sends only what came after it
        resumed = urllib.request.Request(f"{base}/api/jobs/{created['id']}/events", headers={"Last-Event-ID": "3"})
        rest = [json.loads(line[6:]) for line in urllib.request.urlopen(resumed, timeout=10).read().decode().splitlines()
                if line.startswith("data: ")]
        assert rest[0]["i"] == 4
        # files of the run, with ranges for the video player, and nothing outside it
        video = urllib.request.Request(base + events[0]["video"], headers={"Range": "bytes=0-99"})
        r = urllib.request.urlopen(video)
        assert r.status == 206 and len(r.read()) == 100
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{base}/api/jobs/{created['id']}/files/..%2F..%2Fetc%2Fpasswd")
        # the page, its Lottie and the cameras
        assert b"Show it once." in urllib.request.urlopen(base + "/").read()
        assert json.loads(urllib.request.urlopen(base + "/static/scan.json").read())["nm"] == "scan"
        stream = urllib.request.urlopen(base + "/api/stream/overhead", timeout=5)
        assert stream.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        assert b"Content-Type: image/jpeg" in stream.read(200)
        bad = urllib.request.Request(f"{base}/api/jobs?name=a.txt", data=b"x", method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 400
    finally:
        httpd.shutdown()
