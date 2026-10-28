"""Choosing the episodes to train on from their labels."""
import json

from harness import train


def write(root, episode, **label):
    (root / "sidecar").mkdir(parents=True, exist_ok=True)
    (root / "sidecar" / f"{episode:06d}.json").write_text(json.dumps(dict(episode=episode, **label)))


def test_only_episodes_that_worked_are_trained_on(tmp_path):
    ok = dict(valid=True, check=dict(passed=True))
    write(tmp_path, 0, robot=ok)  # no judge yet, robotd's check passed
    write(tmp_path, 1, robot=dict(valid=False, check=dict(passed=True)))  # invalid
    write(tmp_path, 2, robot=dict(valid=True, check=dict(passed=False)))  # missed the target
    write(tmp_path, 3, robot=ok, final=dict(train=True))
    write(tmp_path, 4, robot=ok, final=dict(train=False))  # the judge disagreed
    assert train.trainable(tmp_path) == [0, 3]
    assert train.trainable(tmp_path, judged_only=True) == [3]


def test_the_command_names_the_episodes(tmp_path):
    cmd = train.command("sim_pass_v2", "act_x", [0, 3, 7], 1000, 8)
    assert "--dataset.episodes=[0, 3, 7]" in cmd and "--steps=1000" in cmd and "--policy.type=act" in cmd
