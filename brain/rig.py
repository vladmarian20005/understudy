"""The rig as the brain knows it: the calibrated scene, loaded once, each arm's wrist camera, and whether it's real."""
from harness import robot
from harness.scene import Scene

WRISTS = {"left": "left_wrist", "right": "right_wrist"}
_scene = None
_real = None


def scene():
    global _scene
    if _scene is None:
        _scene = Scene()
    return _scene


def real():
    """Whether robotd drives the real arms: the simulators render the wrist cameras from the mount's CAD."""
    global _real
    if _real is None:
        try:
            _real = not robot.status().get("sim", True)
        except robot.RobotdDown:
            _real = False
    return _real
