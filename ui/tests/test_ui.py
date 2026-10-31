"""The dashboard: a run from upload to the end against the simulator (with a stand-in for the SAM pipeline and a
scripted model), what the brain's turns become as events, the task it's given, reasoning caught from the API, and
the HTTP side: upload, the event stream, files, the cameras."""
import json
import os
import socket
import threading
import time
import urllib.request
from types import SimpleNamespace

import pytest

from brain.llm import Reply, Scripted, parse_calls
from harness import robot
from harness.robotd import Core, serve as serve_robotd
from ui import jobs
from ui.server import serve
from ui.thinking import Narrated, reasoning_of, with_reasoning

SCORE = {"version": 1, "kind": "score", "task": "x", "object": {"name": "paintbrush"},
         "source": {"kind": "video", "muse": {"summary": "left hand passes the paintbrush to the right hand"}},
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


def test_the_description_comes_first_and_the_video_is_context(tmp_path):
    p = jobs.plan(job(tmp_path, "Hand the brush to the right arm."), SCORE)
    assert p["task"].startswith("Hand the brush to the right arm. (In the video the object went from the left hand")
    assert "What the video shows: left hand passes" in p["task"]
    assert p["target"] == [12.0, 24.0] and p["giver"] == "left" and p["taker"] == "right"


def test_without_a_description_a_pass_becomes_delegates_task(tmp_path):
    p = jobs.plan(job(tmp_path, ""), SCORE)
    assert p["task"].startswith("Pick up the brush with the left arm, hand it over in mid-air to the right arm")


def test_without_a_pass_the_summary_is_the_task_and_with_neither_it_stops(tmp_path):
    no_pass = dict(SCORE, quality={"pass_seen": False}, giver=None, taker=None)
    p = jobs.plan(job(tmp_path, ""), no_pass)
    assert p["task"].startswith("Do with the arms what the person in the video did: left hand passes")
    assert p["target"] is None and p["giver"] is None
    with pytest.raises(jobs.StageError, match="no description"):
        jobs.plan(job(tmp_path, ""), dict(no_pass, source={"kind": "video"}))


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


def fake_watch(job, log_line=None):
    """Stands in for the SAM pipeline: a few of its lines, then a score and a clip."""
    folder = os.path.join(job.dir, "sam", "videos", "upload")
    os.makedirs(folder, exist_ok=True)
    for line in ["  Muse (cached): 1 object(s)", "  summary: left hand passes the paintbrush to the right hand",
                 "  objects: paintbrush", "  SAM 'paintbrush' on upload (4.0s, budget used 0/3600s)"]:
        log_line(line)
    json.dump(SCORE, open(os.path.join(folder, "score.json"), "w"))
    open(os.path.join(folder, "preview.mp4"), "wb").write(b"\0" * 64)
    return folder


def talker(x, y):
    """A scripted model that says something before every call, like Muse is asked to."""
    steps = [("Opening the left jaws.", [("open", dict(arm="left"))]),
             ("Going to the brush.", [("move", dict(left=[{"xy": [x, y - 3], "z": 8, "pitch": 60}]))]),
             ("Gripping.", [("grip", dict(arm="left"))]),
             ("Holding it: done.", [("done", dict(success=True, summary="gripped the brush"))])]
    state = {"n": 0}

    def chat(system, messages, tools):
        text, calls = steps[min(state["n"], len(steps) - 1)]
        state["n"] += 1
        tc = [dict(id=f"c{state['n']}_{k}", type="function", function=dict(name=n, arguments=json.dumps(a)))
              for k, (n, a) in enumerate(calls)]
        return Reply(text, parse_calls(tc), dict(role="assistant", content=text, tool_calls=tc), 100, 10, 0.01, "t")

    return SimpleNamespace(chat=chat, model="scripted-talker")


def test_a_run_goes_from_upload_to_the_end_as_events(sim, tmp_path):
    x, y = robot.detect()["objects"][0]["center_cm"]
    runner = jobs.Runner(str(tmp_path), backend_factory=lambda: talker(x, y), watch=fake_watch, record=False)
    j = runner.start(b"\0" * 1000, "clip.mov", "Grip the brush with the left arm.")
    with pytest.raises(RuntimeError, match="still going"):
        runner.start(b"\0", "again.mp4", "")
    deadline = time.time() + 60
    while j.status == "running" and time.time() < deadline:
        time.sleep(0.05)
    kinds = [e["kind"] for e in j.events]
    assert j.status == "done", [e for e in j.events if e["kind"] == "error"]
    assert [e["stage"] for e in j.events if e["kind"] == "stage"] == ["watch", "segment", "plan", "act"]
    assert {"summary", "objects", "segmenting", "segmented", "plan", "thought", "call", "result", "seen",
            "acted", "note", "end"} <= set(kinds)
    thoughts = [e["text"] for e in j.events if e["kind"] == "thought"]
    assert thoughts == ["Opening the left jaws.", "Going to the brush.", "Gripping.", "Holding it: done."]
    calls = [e["name"] for e in j.events if e["kind"] == "call"]
    results = [e["outcome"] for e in j.events if e["kind"] == "result"]
    assert calls == ["open", "move", "grip"] and results == ["done", "done", "done"]
    seg = next(e for e in j.events if e["kind"] == "segmented")
    assert seg["pass_seen"] and seg["files"]["preview.mp4"].endswith("/files/sam/videos/upload/preview.mp4")
    assert any(e["kind"] == "seen" and e["camera"] == "overhead" for e in j.events)
    # replayed from disk, the run is the same
    again = jobs.Job.load(str(tmp_path), j.id)
    assert [e["kind"] for e in again.events] == kinds and again.status == "done"
    assert runner.history()[0]["id"] == j.id


def test_without_robotd_the_run_stops_at_act_with_a_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(robot, "PORT", free_port())  # nothing there
    runner = jobs.Runner(str(tmp_path), backend_factory=lambda: talker(0, 0), watch=fake_watch, record=False)
    j = runner.start(b"\0", "v.mp4", "do it")
    deadline = time.time() + 60
    while j.status == "running" and time.time() < deadline:
        time.sleep(0.05)
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


def test_the_server_takes_an_upload_streams_its_events_and_serves_its_files(sim, tmp_path):
    x, y = robot.detect()["objects"][0]["center_cm"]
    runner = jobs.Runner(str(tmp_path), backend_factory=lambda: talker(x, y), watch=fake_watch, record=False)
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
