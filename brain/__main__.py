"""Give the arms a task in words; a model looks through the cameras and drives them with robotd's primitives.

    python -m brain "pass the brush to the other arm"             (robotd running: --sim or the real arms)
    python -m brain "put the brush near the right arm" --record   the run is a recorded episode, labelled with it
    LLM=<name> python -m brain "..."                              another backend (brain/llm.py)

Each run's transcript (without the images), tool calls, tokens, cost and time go to logs/brain/<time>.json; every
image the model was sent, and each turn as it happens (turns.jsonl), to logs/brain/<time>/.
"""
import argparse
import json
import os
import time

from harness import robot

from . import agent, llm

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
    ap.add_argument("task")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--record", action="store_true", help="record the run as an episode (robotd's --dataset)")
    ap.add_argument("--llm", help="backend (default: LLM in .env, else muse)")
    a = ap.parse_args()
    model = llm.backend(a.llm)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    logs = os.path.join(HERE, "..", "logs", "brain")
    if not robot.status().get("armed"):
        raise SystemExit("the arms are off: Vlad switches torque on (key a in robotd)")
    ep = None
    if a.record:
        begin = robot.episode_begin(a.task, {"kind": "agent", "model": model.model})
        if not begin.get("ok"):
            raise SystemExit("; ".join(begin.get("errors", [])))
        ep = begin["episode"]
    print(f"model {model.model}" + ("" if ep is None else f", recording episode {ep}") + f"\ntask: {a.task}",
          flush=True)
    out = agent.run(a.task, model, a.max_turns, log=lambda s: print(s, flush=True), run_dir=os.path.join(logs, stamp))
    os.makedirs(logs, exist_ok=True)
    path = os.path.join(logs, stamp + ".json")
    json.dump(dict(out, task=a.task, episode=ep, transcript=without_images(out["transcript"])), open(path, "w"),
              indent=1, default=str)
    if ep is not None:  # after the log is saved: robotd may be gone by now
        try:
            end = robot.episode_end(valid=out["success"], note=None if out["success"] else out["summary"])
            robot.episode_label(ep, "agents", {k: v for k, v in out.items() if k != "transcript"})
            print(f"episode {ep}: {end.get('frames', 0)} frames" + ("" if end.get("ok") else f", {end.get('errors')}"))
        except robot.RobotdDown as e:
            print(f"episode {ep} not ended: {e}")
    print(f"{out['turns']} turns, {len(out['calls'])} calls, {out['tokens_in']} + {out['tokens_out']} tokens, "
          f"${out['cost']:.3f}, {out['seconds']} s ({out['llm_seconds']} s in the model); log {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
