"""Delegate: a score (what a person did in a video, watch/SCORE.md) becomes the brain's task on the arms.

    python -m watch.delegate sam_out/scores/<clip_id>/score.json --dry-run      the task it would give, no model
    python -m watch.delegate sam_out/scores/<clip_id>/score.json --record       run it as a recorded episode
    python -m watch.delegate score.json --target -12 24 --record               put the object down there instead

It runs the brain's own loop (brain/agent.py) with the brain's model (LLM= in .env, default Muse Spark), so robotd has
to be running: ./understudy.sh --sim, or the real arms. With --record the episode is labelled with the score it came
from (source.score) and gets the target, so robotd's end check and the judge can both say whether it worked.
"""
import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# where an object goes when the score can't say (a moving head camera): in front of the taker, as the ping-pong does
DEFAULT_TARGET = {"left": [-12.0, 24.0], "right": [12.0, 24.0]}


class ScoreError(ValueError):
    pass


def load(path):
    score = json.load(open(path))
    if score.get("kind") != "score" or score.get("version") != 1:
        raise ScoreError(f"{path} isn't a version 1 score (watch/SCORE.md)")
    return score


def check(score, allow_unseen=False):
    """Can the brain be given this score? Raises ScoreError saying why not."""
    giver, taker = score.get("giver"), score.get("taker")
    if giver not in ("left", "right") or taker not in ("left", "right") or giver == taker:
        raise ScoreError(f"the score doesn't say who gives and who takes (giver {giver!r}, taker {taker!r})")
    if not (score.get("quality") or {}).get("pass_seen") and not allow_unseen:
        raise ScoreError("SAM saw no pass in this clip (quality.pass_seen is false); --allow-unseen to use it anyway")


def target_for(score, override=None):
    """Where the object should end: --target, else the score's own (overhead clips), else in front of the taker."""
    if override is not None:
        return [float(v) for v in override]
    if score.get("target_cm") is not None:
        return [float(v) for v in score["target_cm"]]
    return list(DEFAULT_TARGET[score["taker"]])


def task_for(score, target, robot_object="brush"):
    """The task in words, as the brain gets it: the arms' job first, then what the person did, as a hint."""
    giver, taker = score["giver"], score["taker"]
    seen = ((score.get("vision") or {}).get("objects") or [{}])[0].get("name")  # scores from before "object"
    person = (score.get("object") or {}).get("name") or seen or "object"
    hint = f"In the video a person passed the {person} from their {giver} hand to their {taker} hand"
    h = score.get("handover")
    if h and h.get("both_s") is not None:
        hint += f", holding it with both hands for {h['both_s']:.1f} s while the {taker} hand took it"
    if h and h.get("at_cm") is not None:
        hint += f", around ({h['at_cm'][0]:.0f}, {h['at_cm'][1]:.0f}) cm on the table"
    return (f"Pick up the {robot_object} with the {giver} arm, hand it over in mid-air to the {taker} arm, then put it "
            f"down with its centre at x {target[0]:g}, y {target[1]:g} (cm) and send both arms home. "
            f"{hint}: do the same with the arms.")


def source_for(score, path, model):
    """What the episode's label records about where its task came from (no narration text: Ego4D's)."""
    src = score.get("source") or {}
    return {"kind": "agent", "model": model, "score": {
        "file": os.path.relpath(os.path.abspath(path), HERE), "source": src.get("kind"),
        "video_uid": src.get("video_uid"), "start_s": src.get("start_s"),
        "giver": score["giver"], "taker": score["taker"], "handover": score.get("handover")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("score", help="a score.json")
    ap.add_argument("--target", type=float, nargs=2, metavar=("X", "Y"), help="where to put it down, table cm")
    ap.add_argument("--object", default="brush", help="the object on the table (default brush)")
    ap.add_argument("--record", action="store_true", help="record the run as an episode, labelled with the score")
    ap.add_argument("--dry-run", action="store_true", help="print the task, don't run it")
    ap.add_argument("--allow-unseen", action="store_true", help="use a score whose clip showed no pass")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--llm", help="backend (default: LLM in .env, else muse)")
    a = ap.parse_args()

    score = load(a.score)
    try:
        check(score, a.allow_unseen)
    except ScoreError as e:
        raise SystemExit(f"{a.score}: {e}")
    target = target_for(score, a.target)
    task = task_for(score, target, a.object)
    print(f"task: {task}\ntarget: {target}")
    if a.dry_run:
        return

    from brain import agent, llm
    from brain.__main__ import without_images
    from harness import robot

    model = llm.backend(a.llm)
    if not robot.status().get("armed"):
        raise SystemExit("the arms are off: Vlad switches torque on (key a in robotd)")
    ep = None
    if a.record:
        begin = robot.episode_begin(task, source_for(score, a.score, model.model), giver=score["giver"],
                                    taker=score["taker"], target_cm=target)
        if not begin.get("ok"):
            raise SystemExit("; ".join(begin.get("errors", [])))
        ep = begin["episode"]
        print(f"recording episode {ep}")
    out = agent.run(task, model, a.max_turns)
    if ep is not None:
        # the episode stays valid when the brain says it failed: whether it worked is robotd's check and the judge's
        # call, and a failed try is still one the judge should see
        end = robot.episode_end(valid=True, note=None if out["success"] else f"the brain says: {out['summary']}")
        robot.episode_label(ep, "agents", {k: v for k, v in out.items() if k != "transcript"})
        print(f"episode {ep}: {end.get('frames', 0)} frames" + ("" if end.get("ok") else f", {end.get('errors')}"))
    os.makedirs(os.path.join(HERE, "logs", "delegate"), exist_ok=True)
    path = os.path.join(HERE, "logs", "delegate", time.strftime("%Y%m%d-%H%M%S") + ".json")
    json.dump(dict(out, task=task, score=a.score, target=target, episode=ep,
                   transcript=without_images(out["transcript"])), open(path, "w"), indent=1, default=str)
    print(f"{'success' if out['success'] else 'failure'}: {out['summary']}\n"
          f"{out['turns']} turns, {out['tokens_in']} + {out['tokens_out']} tokens, ${out['cost']:.3f}, "
          f"{out['seconds']} s; log {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
