"""The preflight check, against a simulated robotd on its own port."""
import threading

import pytest

from harness import preflight, robot
from harness.robotd import Core, serve


@pytest.fixture
def robotd(monkeypatch):
    core = Core(sim_mode=True, speed=0, quiet=True, dataset=None)
    stop = threading.Event()
    t = threading.Thread(target=serve, args=(core, 5892, stop), daemon=True)
    t.start()
    monkeypatch.setattr(robot, "PORT", 5892)
    yield core
    stop.set()
    t.join(timeout=2)
    core.close()


def test_a_simulated_session_is_ready(robotd, capsys, monkeypatch):
    monkeypatch.setattr(preflight, "MIN_FREE_GB", 0.0)  # not this machine's disk
    assert preflight.main() == 0
    out = capsys.readouterr().out
    assert "robotd answers (simulator)" in out and "ready" in out


def test_no_robotd_is_not_ready(monkeypatch, capsys):
    monkeypatch.setattr(robot, "PORT", 5893)
    monkeypatch.setattr(robot, "call", lambda *a, **k: (_ for _ in ()).throw(robot.RobotdDown("not running")))
    assert preflight.main() == 1
    assert "FAIL" in capsys.readouterr().out


def test_the_api_key_is_found_in_dot_env(tmp_path, monkeypatch):
    for k in preflight.KEY_VARS:
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text("OTHER=1\nMUSE_API_KEY=abc\n")
    assert preflight.env_keys(str(env)) == {"MUSE_API_KEY"}
