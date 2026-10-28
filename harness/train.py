"""Train a policy on recorded episodes with lerobot, using only the episodes their labels say to.

    python -m harness.train sim_pass_v2 --steps 30000                       every trainable episode
    python -m harness.train understudy_pass --episodes 50 --name act_50      the first 50 (for the scaling curve)
    python -m harness.train understudy_pass --dry-run                        show the episodes and the command

An episode is trainable when its label's final decision says train (the judge and robotd's own check agree it
worked). One no judge has seen yet counts when it's valid and robotd's end check passed, unless --judged-only.
It runs lerobot-train (ACT, on the GPU) into outputs/<name>, with its output in logs/train_<name>.log.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

from .recorder import DATASETS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def command(dataset, name, episodes, steps, batch_size, policy="act", device="mps", save_freq=5000):
    return ["lerobot-train", f"--dataset.repo_id=local/{dataset}", f"--dataset.root={os.path.join(DATASETS, dataset)}",
            f"--dataset.episodes={json.dumps(episodes)}", f"--policy.type={policy}", f"--policy.device={device}",
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
    ap.add_argument("--policy", default="act")
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
    cmd = command(a.dataset, name, eps, a.steps, a.batch_size, a.policy, a.device)
    print(f"{len(eps)} trainable episodes of {len(glob.glob(os.path.join(root, 'sidecar', '*.json')))} -> outputs/{name}")
    if a.dry_run:
        print(" ".join(cmd))
        return 0
    if os.path.exists(os.path.join(HERE, "outputs", name)):
        sys.exit(f"outputs/{name} exists: pick another --name or remove it")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    log = os.path.join(HERE, "logs", f"train_{name}.log")
    print(f"training; log: {log}")
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


if __name__ == "__main__":
    sys.exit(main())
