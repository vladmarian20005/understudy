"""Before a session: is everything ready? Checks what can be checked without moving anything.

    ./understudy.sh check            (or python -m harness.preflight; robotd running, real or --sim)

Required (the session shouldn't start without them): robotd answers, every camera streams at a usable rate, the
servos are cool, the followers' calibration files exist, there's disk space for recordings. Advisory: torque is on,
the arms' sag has been learned before, a model API key is set for the brain. Exit code 1 if anything required fails.
"""
import os
import shutil
import sys

from . import robot
from .kin import CALIB_DIR
from .recorder import DATASETS
from .scene import SIDES, Scene

MIN_FPS = 25.0
MAX_TEMP_C = 50.0
MIN_FREE_GB = 20.0
SAG_DIR = os.path.expanduser("~/.cache/understudy")
KEY_VARS = ("MUSE_API_KEY", "LLM_API_KEY")


def env_keys(path=".env"):
    """Variables set in the environment or in .env."""
    keys = set(k for k in KEY_VARS if os.environ.get(k))
    try:
        with open(path) as f:
            for line in f:
                k, _, v = line.strip().partition("=")
                if k in KEY_VARS and v.strip():
                    keys.add(k)
    except OSError:
        pass
    return keys


def checks():
    """(required, ok, what) for each check."""
    out = []
    try:
        st = robot.status()
    except robot.RobotdDown as e:
        return [(True, False, f"robotd: {e}")]
    sim = bool(st.get("sim"))
    out.append((True, True, f"robotd answers ({'simulator' if sim else 'real arms'})"))
    out.append((False, st["armed"], "torque on" if st["armed"] else "torque off: press a in robotd's terminal"))
    bad = {n: h for n, h in st["cameras"].items() if not h["ok"] or h["fps"] < MIN_FPS}
    out.append((True, not bad, "cameras: " + (", ".join(f"{n} {h['fps']:.0f} fps" for n, h in st["cameras"].items())
                                             if not bad else "not streaming well: " +
                                             ", ".join(f"{n} ({h['fps']:.0f} fps, ok={h['ok']})"
                                                       for n, h in bad.items()))))
    if not sim:
        temps = {s: a.get("temp_c") for s, a in st["arms"].items()}
        hot = {s: t for s, t in temps.items() if t is not None and t > MAX_TEMP_C}
        out.append((True, not hot, "servos: " + ", ".join(f"{s} {t:.0f} C" for s, t in temps.items() if t is not None)
                    + (" (let them cool first)" if hot else "")))
        scene = Scene()
        missing = [c for c in scene.calibration_ids.values() if not os.path.exists(os.path.join(CALIB_DIR, f"{c}.json"))]
        out.append((True, not missing, "calibration files: " + ("present" if not missing else
                                                                 f"missing {', '.join(missing)} in {CALIB_DIR}")))
        sag = os.path.join(SAG_DIR, "_".join(scene.calibration_ids[s] for s in SIDES) + ".json")
        out.append((False, os.path.exists(sag), "arm sag: " + ("learned in an earlier session" if os.path.exists(sag)
                                                              else "not learned yet: hold the arms still a moment")))
    os.makedirs(DATASETS, exist_ok=True)
    free = shutil.disk_usage(DATASETS).free / 1e9
    out.append((True, free >= MIN_FREE_GB, f"disk: {free:.0f} GB free for datasets"))
    keys = env_keys()
    out.append((False, bool(keys), "model API key: " + (", ".join(sorted(keys)) if keys else
                                                       f"none ({' or '.join(KEY_VARS)} in .env) for the brain")))
    return out


def main():
    results = checks()
    for required, ok, what in results:
        mark = "ok  " if ok else ("FAIL" if required else "warn")
        print(f"  {mark}  {what}")
    failed = [what for required, ok, what in results if required and not ok]
    print("ready" if not failed else f"not ready: {len(failed)} required check{'s' if len(failed) > 1 else ''} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
