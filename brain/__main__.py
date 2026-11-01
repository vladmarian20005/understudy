"""Give the arms a task in words or a video of it; Muse plans it, the arms run the plan, Muse checks (brain/plan.py).

    python -m brain "<task>"                        (robotd running: ./understudy.sh --sim, or the real arms)
    python -m brain --video clip.mp4                do what the person in the video does
    python -m brain "<task>" --record               the run is a recorded episode, labelled with it
    python -m brain "<task>" --plan-only --save-plan plan.json     Muse's plan, for brain.rehearse; nothing moves
    python -m brain "<task>" --steer                the older loop: Muse steers every move itself
    LLM=<name> python -m brain "<task>"             another backend (brain/llm.py)

Each run's transcript (without the images), steps, tokens, cost and time go to logs/brain/<time>.json; every image
the model was sent, and each step and answer as it happens (turns.jsonl), to logs/brain/<time>/.
"""
import argparse
import json
import os
import time

from harness import robot

from . import agent, llm, plan

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
    ap.add_argument("--steer", action="store_true", help="the step-by-step loop instead of plan, act, check")
    ap.add_argument("--plan-only", action="store_true", help="look and plan, move nothing")
    ap.add_argument("--save-plan", metavar="FILE", help="write Muse's plan to FILE: brain.rehearse replays it")
    ap.add_argument("--max-turns", type=int, default=40, help="--steer: the most model calls")
    ap.add_argument("--record", action="store_true", help="record the run as an episode (robotd's --dataset)")
    ap.add_argument("--llm", help="backend (default: LLM in .env, else muse)")
    a = ap.parse_args()
    if not a.task and not a.video:
        ap.error("give a task or --video")
    if a.video and not os.path.exists(a.video):
        ap.error(f"no video at {a.video}")
    if a.steer and not a.task:
        ap.error("--steer needs a task in words")
    model = llm.backend(a.llm)
    # sims run side by side, each on its own port
    stamp = time.strftime("%Y%m%d-%H%M%S") + (f"-{robot.PORT}" if os.environ.get("ROBOTD_PORT") else "")
    logs = os.path.join(HERE, "..", "logs", "brain")
    if not robot.status().get("armed") and not a.plan_only:
        raise SystemExit("the arms are off: switch torque on (key a in robotd)")
    ep = None
    if a.record and not a.plan_only:
        begin = robot.episode_begin(a.task or f"the video {os.path.basename(a.video)}",
                                    {"kind": "agent", "model": model.model, "video": a.video})
        if not begin.get("ok"):
            raise SystemExit("; ".join(begin.get("errors", [])))
        ep = begin["episode"]
    print(f"model {model.model}" + ("" if ep is None else f", recording episode {ep}") +
          f"\ntask: {a.task or a.video}", flush=True)
    log, run_dir = (lambda s: print(s, flush=True)), os.path.join(logs, stamp)
    if a.steer:
        out = agent.run(a.task, model, a.max_turns, log=log, run_dir=run_dir)
    else:
        r = plan.Run(model, log, run_dir)
        t0 = time.time()
        try:
            out = r.go(a.task, a.video, plan_only=a.plan_only)
        except robot.RobotdDown as e:
            out = r.end(False, f"robotd stopped answering: {e}", t0)
        except KeyboardInterrupt:
            robot.stop("brain interrupted")
            out = r.end(False, "interrupted", t0)
        if a.save_plan and r.plan:
            json.dump(r.plan, open(a.save_plan, "w"), indent=1)
        out = dict(out, turns=out["calls"], calls=[x for x in out.pop("record") if "step" in x],
                   transcript=r.messages)
    os.makedirs(logs, exist_ok=True)
    path = os.path.join(logs, stamp + ".json")
    json.dump(dict(out, task=a.task, video=a.video, episode=ep, transcript=without_images(out["transcript"])),
              open(path, "w"), indent=1, default=str)
    if ep is not None:  # after the log is saved: robotd may be gone by now
        try:
            end = robot.episode_end(valid=out["success"], note=None if out["success"] else out["summary"])
            robot.episode_label(ep, "agents", {k: v for k, v in out.items() if k != "transcript"})
            print(f"episode {ep}: {end.get('frames', 0)} frames" + ("" if end.get("ok") else f", {end.get('errors')}"))
        except robot.RobotdDown as e:
            print(f"episode {ep} not ended: {e}")
    print(f"{out['turns']} model calls, {len(out['calls'])} steps, {out['tokens_in']} + {out['tokens_out']} tokens, "
          f"${out['cost']:.3f}, {out['seconds']} s ({out['llm_seconds']} s in the model); log {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
