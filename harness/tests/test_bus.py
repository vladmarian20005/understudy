"""Bus drop-outs: a missed read is shrugged off, a run of them stops the motion and spoils the episode."""
import json
import time

import pytest

from harness import controller
from harness.robotd import Core


@pytest.fixture
def core(tmp_path):
    c = Core(sim_mode=True, speed=0, quiet=True, dataset="bus", datasets=str(tmp_path))
    yield c
    c.close()


def run(core, name, **args):
    assert core.start_skill(name, args)["ok"]
    while core.busy():
        time.sleep(0.005)
    return core.skill["result"]


def flaky(core, side, fails):
    """The arm's reads raise on the read numbers in `fails`, counted while a motion plays."""
    io = core.world.io[side]
    real, n = io.read, [0]

    def read():
        if core.ctl.queue:
            n[0] += 1
            if n[0] in fails:
                raise IOError("no status packet")
        return real()

    io.read = read


def test_a_few_missed_reads_dont_stop_a_motion(core):
    flaky(core, "left", {3, 10, 11})
    assert run(core, "move_to", arm="left", xy=[-12, 26], z=12)["outcome"] == "done"
    assert core.ctl.bus_faults == 0


def test_a_run_of_missed_reads_stops_the_motion_and_spoils_the_episode(core):
    assert core.handle(dict(cmd="episode_begin", task="t"))["ok"]
    flaky(core, "left", set(range(10, 10 + controller.BUS_STOP_TICKS + 3)))
    r = run(core, "move_to", arm="left", xy=[-12, 26], z=12)
    assert r["outcome"] == "stopped" and r["fail"] == "blocked" and "bus" in r["why"]
    texts = [e["text"] for e in core.events_since(0)]
    assert any("bus not answering" in t for t in texts) and any("answering again" in t for t in texts)
    assert run(core, "home")["outcome"] == "done"  # it carries on once the bus is back
    end = core.handle(dict(cmd="episode_end"))
    robot = json.load(open(end["sidecar"]))["robot"]
    assert robot["valid"] is False and robot["bus_faults"] == 1 and "bus drop-out" in robot["note"]
