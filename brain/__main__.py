"""Give the arms a task in words or a video of it; Muse drives them one small step at a time (brain/loop.py).

    python -m brain "<task>"                        (robotd running: ./understudy.sh --sim, or the real arms)
    python -m brain --video clip.mp4                do what the person in the video does
    python -m brain "<task>" --record               the run is a recorded episode, labelled with it
    LLM=<name> python -m brain "<task>"             another backend (brain/llm.py)

Each run's transcript (without the images), steps, tokens, cost and time go to logs/brain/<time>.json; every image
Muse was sent and each step and answer as it happens (turns.jsonl), to logs/brain/<time>/.
"""
import argparse
import json
import os
import time

from harness import robot

from . import llm, loop

HERE = os.path.dirname(os.path.abspath(__file__))


def without_images(messages):
    out = []
    for m in messages:
        m = dict(m)
        if isinstance(m.get("content"), list):
            m["content"] = [p if p.get("type") != "image_url" else {"type": "image"} for p in m["content"]]
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", nargs="?", default="")
    ap.add_argument("--video", help="a video of a person doing the task")
    ap.add_argument("--max-turns", type=int, default=loop.MAX_TURNS, help="the most steps")
    ap.add_argument("--effort", default="low", help="Muse's reasoning effort each turn (low, medium, high)")
    ap.add_argument("--record", action="store_true", help="record the run as an episode (robotd's --dataset)")
    ap.add_argument("--llm", help="backend (default: LLM in .env, else muse)")
    ap.add_argument("--no-home", action="store_true", help="start from where the arms are, not the ready pose")
    ap.add_argument("--held", action="append", default=[], metavar="SIDE:NAME:LENGTH:BELOW:ROLL0[:AHEAD]",
                    help="what an arm already holds: its length (cm), how much of it is under the grip point, the "
                         "wrist roll when it was taken, and how far its axis is in front of the grip point (cm)")
    a = ap.parse_args()
    if not a.task and not a.video:
        ap.error("give a task or --video")
    if a.video and not os.path.exists(a.video):
        ap.error(f"no video at {a.video}")
    model = llm.backend(a.llm)
    # sims run side by side, each on its own port
    stamp = time.strftime("%Y%m%d-%H%M%S") + (f"-{robot.PORT}" if os.environ.get("ROBOTD_PORT") else "")
    logs = os.path.join(HERE, "..", "logs", "brain")
    if not robot.status().get("armed"):
        raise SystemExit("the arms are off: switch torque on (key a in robotd)")
    task = a.task
    if a.video:
        task = loop.video_task(task, loop.watch(a.video))
    ep = None
    if a.record:
        begin = robot.episode_begin(a.task or f"the video {os.path.basename(a.video)}",
                                    {"kind": "agent", "model": model.model, "video": a.video})
        if not begin.get("ok"):
            raise SystemExit("; ".join(begin.get("errors", [])))
        ep = begin["episode"]
    print(f"model {model.model}" + ("" if ep is None else f", recording episode {ep}") + f"\ntask: {task}", flush=True)
    held = {}
    for h in a.held:
        side, name, length, below, roll0, *ahead = h.split(":")
        held[side] = dict(name=name, length=float(length), below=float(below), roll0=float(roll0),
                          ahead=float(ahead[0]) if ahead else 0.0)
    out = loop.run(task, model, a.max_turns, log=lambda s: print(s, flush=True), run_dir=os.path.join(logs, stamp),
                   effort=a.effort, home=not a.no_home, held=held)
    out = dict(out, steps_record=[x for x in out.pop("record") if "step" in x])
    os.makedirs(logs, exist_ok=True)
    path = os.path.join(logs, stamp + ".json")
    json.dump(dict(out, task=task, video=a.video, episode=ep, transcript=without_images(out["transcript"])),
              open(path, "w"), indent=1, default=str)
    if ep is not None:  # after the log is saved: robotd may be gone by now
        try:
            end = robot.episode_end(valid=out["success"], note=None if out["success"] else out["summary"])
            robot.episode_label(ep, "agents", {k: v for k, v in out.items() if k != "transcript"})
            print(f"episode {ep}: {end.get('frames', 0)} frames" + ("" if end.get("ok") else f", {end.get('errors')}"))
        except robot.RobotdDown as e:
            print(f"episode {ep} not ended: {e}")
    print(f"{out['calls']} model calls, {out['steps']} steps, {out['tokens_in']} + {out['tokens_out']} tokens, "
          f"${out['cost']:.3f}, {out['seconds']} s ({out['llm_seconds']} s in the model); log {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
