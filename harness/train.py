"""Train a policy on recorded episodes with lerobot, using only the episodes their labels say to.

    python -m harness.train DATASET --steps 30000                     every trainable episode
    python -m harness.train DATASET --episodes 50 --name act_50       the first 50 (for the scaling curve)
    python -m harness.train DATASET --dry-run                         show the episodes and the command
    python -m harness.train DATASET --from outputs/RUN --name NAME    fine-tune a trained policy on this dataset

An episode is trainable when its label's final decision says train (the judge and robotd's own check agree it
worked). One no judge has seen yet counts when it's valid and robotd's end check passed, unless --judged-only.
It runs lerobot-train (ACT, on the GPU) into outputs/<name>, with its output in logs/train_<name>.log.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

from .policy import resolve
from .recorder import DATASETS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSTANT_RANGE = 1.0  # a dimension moving less than this (deg, or % of the gripper) over a dataset counts as constant


def trainable(root, judged_only=False):
    """Episode indices to train on, in order, from datasets/<name>/sidecar/*.json."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, "sidecar", "*.json"))):
        label = json.load(open(path))
        if "final" in label:
            ok = bool(label["final"].get("train"))
        else:
            robot = label.get("robot", {})
            ok = not judged_only and robot.get("valid") and (robot.get("check") or {}).get("passed")
        if ok:
            out.append(int(label["episode"]))
    return out


def fix_constant_stats(root):
    """A joint that never moves (a parked arm, a wrist roll the plan never turns) has std 0, and lerobot divides by
    std + 1e-8: float rounding in its mean became normalised values up to -11444 (loss 1226 at step 200, against 6.9 on
    data where everything moves). Such a dimension gets its true value as the mean and std 1, so it normalises to about
    0. The first original is kept as meta/stats.before_fix.json. Returns the dimensions set, by feature."""
    path = os.path.join(root, "meta", "stats.json")
    stats = json.load(open(path))
    fixed = {}
    for key in ("action", "observation.state"):
        s = stats[key]
        fixed[key] = [i for i in range(len(s["mean"])) if s["max"][i] - s["min"][i] < CONSTANT_RANGE]
        for i in fixed[key]:
            s["mean"][i], s["std"][i] = (s["min"][i] + s["max"][i]) / 2, 1.0
    backup = os.path.join(root, "meta", "stats.before_fix.json")
    if not os.path.exists(backup):
        shutil.copy(path, backup)
    json.dump(stats, open(path, "w"), indent=4)
    return fixed


def command(dataset, name, episodes, steps, batch_size, policy="act", device="mps", save_freq=5000, start=None):
    model = f"--policy.path={resolve(start)}" if start else f"--policy.type={policy}"
    return ["lerobot-train", f"--dataset.repo_id=local/{dataset}", f"--dataset.root={os.path.join(DATASETS, dataset)}",
            f"--dataset.episodes={json.dumps(episodes)}", model, f"--policy.device={device}",
            "--policy.push_to_hub=false", f"--output_dir={os.path.join(HERE, 'outputs', name)}", f"--job_name={name}",
            f"--steps={steps}", f"--batch_size={batch_size}", f"--save_freq={save_freq}", "--log_freq=200",
            "--eval_freq=0", "--num_workers=8", "--wandb.enable=false"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", help="the dataset's name under datasets/")
    ap.add_argument("--name", help="the run's name under outputs/ (default: act_<dataset>[_<episodes>])")
    ap.add_argument("--episodes", type=int, help="only the first N trainable episodes")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-freq", type=int, default=5000, help="steps between checkpoints (~0.6 GB each for ACT)")
    ap.add_argument("--policy", default="act")
    ap.add_argument("--from", dest="start", help="keep training this checkpoint or run on the dataset (its stats replace the old ones)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--judged-only", action="store_true", help="only episodes the judge has labelled")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = os.path.join(DATASETS, a.dataset)
    eps = trainable(root, a.judged_only)
    if a.episodes:
        eps = eps[:a.episodes]
    if not eps:
        sys.exit(f"no trainable episodes in {root}")
    name = a.name or f"{a.policy}_{a.dataset}" + (f"_{a.episodes}" if a.episodes else "")
    cmd = command(a.dataset, name, eps, a.steps, a.batch_size, a.policy, a.device, a.save_freq, a.start)
    print(f"{len(eps)} trainable episodes of {len(glob.glob(os.path.join(root, 'sidecar', '*.json')))} -> outputs/{name}")
    if a.dry_run:
        print(" ".join(cmd))
        return 0
    if os.path.exists(os.path.join(HERE, "outputs", name)):
        sys.exit(f"outputs/{name} exists: pick another --name or remove it")
    print("constant dimensions set to std 1:", fix_constant_stats(root))
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    log = os.path.join(HERE, "logs", f"train_{name}.log")
    print(f"training; log: {log}")
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


if __name__ == "__main__":
    sys.exit(main())
