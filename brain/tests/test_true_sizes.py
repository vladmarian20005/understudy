"""In the simulator the brain works with where things really are and how big, not one camera's estimates."""
from brain import tools
from harness import robot

SAM = [dict(name="can", found_as="tennis ball can", center_cm=[4.5, 31.0], size_cm=[9.1, 9.1, 24.0], z_cm=12.0),
       dict(name="ball", found_as="tennis ball", center_cm=[-30.0, 20.0], size_cm=[7.0, 7.0, 7.0], z_cm=3.5)]
TRUTH = [dict(name="tennis ball can", center_cm=[4.0, 30.0], size_cm=[7.8, 7.8, 21.0], z_cm=10.5)]


def test_the_simulators_sizes_replace_the_estimates_only_in_the_simulator(monkeypatch):
    monkeypatch.setattr(robot, "detect", lambda *a, **k: dict(objects=TRUTH))
    for sim, can_size in ((True, [7.8, 7.8, 21.0]), (False, [9.1, 9.1, 24.0])):
        tools.reset()
        monkeypatch.setattr(robot, "status", lambda sim=sim: dict(sim=sim))
        can, ball = tools.true_sizes([dict(o) for o in SAM])
        assert can["size_cm"] == can_size
        assert can["center_cm"] == ([4.0, 30.0] if sim else [4.5, 31.0])  # in the simulator, where it really is
        assert can.get("sim_name") == ("tennis ball can" if sim else None)
        assert ball["size_cm"] == [7.0, 7.0, 7.0]  # nothing in the simulator near it: the estimate stays
