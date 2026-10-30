"""Judge recorded episodes: did each one do its task? The verdict becomes the label's judge section.

    python -m judge                                   every valid episode of datasets/understudy_pass without a verdict
    python -m judge --dataset sim_pass --episodes 3 7
    python -m judge --force                           judge again, replacing the verdict
    python -m judge --dry-run                         the episodes and frames it would use, no model calls
    python -m judge --no-send                         keep the verdicts in logs/ only, don't store them through robotd

The verdict is stored through robotd (episode_label), so robotd has to be running, and robotd then writes the
training decision (final). Each verdict and the frames it was given also go to logs/judge/<dataset>/<episode>/.
The model is the brain's: LLM= in .env (default Muse Spark, with MODEL_API_KEY).
"""
import argparse
import glob
import json
import os

import cv2

from brain import llm as llms
from harness import robot
from harness.recorder import DATASETS
from harness.scene import Scene

from . import frames as fr
from .judge import JudgeError, judge, pick_frames

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def labels(root, episodes=None):
    out = []
    for path in sorted(glob.glob(os.path.join(root, "sidecar", "*.json"))):
        label = json.load(open(path))
        if episodes is None or int(label["episode"]) in episodes:
            out.append(label)
    return out


def check_dataset(name):
    """robotd stores the verdict in the dataset it was started with, whichever one we read: refuse to judge when it
    says (status' "dataset") that it's on another one, or the verdicts would land on the wrong episodes."""
    try:
        have = robot.status().get("dataset")
    except robot.RobotdDown as e:
        raise SystemExit(f"{e}\n(--no-send judges without robotd, keeping the verdicts in logs/judge/)")
    if have is not None and have != name:
        raise SystemExit(f"robotd is on dataset {have!r}, not {name!r}: restart it with --dataset {name}, "
                         f"or use --no-send")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="understudy_pass", help="its name under datasets/")
    ap.add_argument("--episodes", type=int, nargs="+", help="only these (default: all)")
    ap.add_argument("--force", action="store_true", help="judge episodes that already have a verdict again")
    ap.add_argument("--invalid-too", action="store_true", help="also judge episodes robotd marked invalid")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-send", action="store_true", help="don't store the verdicts through robotd")
    ap.add_argument("--llm", help="backend (default: LLM in .env, else muse)")
    a = ap.parse_args()

    root = os.path.join(DATASETS, a.dataset)
    todo = []
    for label in labels(root, set(a.episodes) if a.episodes else None):
        ep, robot_ = int(label["episode"]), label.get("robot") or {}
        if "judge" in label and not a.force:
            continue
        if not robot_.get("valid") and not a.invalid_too:
            print(f"episode {ep}: skipped, robotd marked it invalid ({robot_.get('note')})")
            continue
        todo.append(label)
    print(f"{len(todo)} episode(s) of {a.dataset} to judge")
    if a.dry_run:
        for label in todo:
            print(f"  {label['episode']}: {label.get('task')!r}, frames {pick_frames(label)}")
        return
    if not todo:
        return
    if not a.no_send:
        check_dataset(a.dataset)

    model = llms.backend(a.llm)
    scene = Scene()
    results = []
    total_cost = 0.0
    for label in todo:
        ep = int(label["episode"])
        picks = pick_frames(label)
        if not picks:
            print(f"episode {ep}: no frames, skipped")
            continue
        target = (label.get("robot") or {}).get("target_cm")
        fps = float(label.get("fps") or 30)
        raw = fr.read(root, ep, [i for i, _ in picks])
        images = [(i, why, fr.annotate(img, scene, target, f"ep {ep}  frame {i}  t={i / fps:.1f}s  {why}"))
                  for (i, why), img in zip(picks, raw)]
        out_dir = os.path.join(HERE, "logs", "judge", a.dataset, f"{ep:06d}")
        os.makedirs(out_dir, exist_ok=True)
        for i, _, img in images:
            cv2.imwrite(os.path.join(out_dir, f"frame_{i:05d}.jpg"), img)
        try:
            verdict, cost = judge(label, images, model)
        except JudgeError as e:
            print(f"episode {ep}: no verdict ({e}); left unjudged")
            continue
        total_cost += cost
        json.dump(verdict, open(os.path.join(out_dir, "judge.json"), "w"), indent=1)
        final = None
        if not a.no_send:
            try:
                r = robot.episode_label(ep, "judge", verdict)
            except robot.RobotdDown as e:
                raise SystemExit(f"{e}\n(the verdicts so far are in logs/judge/; --no-send judges without robotd)")
            if not r.get("ok"):
                print(f"episode {ep}: robotd didn't store it: {r.get('errors')}")
            final = r.get("final")
        check = ((label.get("robot") or {}).get("check") or {}).get("passed")
        results.append(dict(episode=ep, success=verdict["success"], confidence=verdict["confidence"], check=check,
                            final=final))
        word = "success" if verdict["success"] else f"failure ({(verdict['failure'] or {}).get('kind', '?')})"
        print(f"episode {ep}: {word}, confidence {verdict['confidence']:.2f}: {verdict['reason']}"
              + (f"\n  final: {final}" if final else ""))

    if results:
        ok = sum(r["success"] for r in results)
        checked = [r for r in results if r["check"] is not None]
        agree = sum(r["success"] == r["check"] for r in checked)
        review = sum(bool((r["final"] or {}).get("review")) for r in results)
        print(f"\n{len(results)} judged: {ok} success, {len(results) - ok} failure; "
              f"agrees with robotd's check on {agree}/{len(checked)}; {review} for review; ${total_cost:.3f}")
        path = os.path.join(HERE, "logs", "judge", a.dataset, "summary.json")
        json.dump(results, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
