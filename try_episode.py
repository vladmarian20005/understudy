"""Record one scripted handover in Vlad's simulator, with no model: pick, handover, place, home.

    python -m harness.robotd --sim        (first, in another terminal)
    python try_episode.py                 puts the brush on the target: should pass
    python try_episode.py --miss          puts it 10 cm short of the target: should fail

Then judge it with: python -m judge
"""
import sys

from harness import robot

TARGET = [12, 24]
place_at = [12, 14] if "--miss" in sys.argv else TARGET

r = robot.episode_begin("Pass the brush to the other arm and put it on the target", {"kind": "scripted"},
                        giver="left", taker="right", target_cm=TARGET)
if not r.get("ok"):
    sys.exit(f"robotd refused: {r.get('errors')}")
print(f"recording episode {r['episode']}")
for name, args in [("pick", dict(arm="left", target="brush")),
                   ("handover", dict(giver="left", taker="right")),
                   ("place", dict(arm="right", xy=place_at)),
                   ("home", {})]:
    res = getattr(robot, name)(**args)
    print(f"  {name}: {res['outcome']}" + (f" ({res.get('fail')}: {res.get('why')})" if res["outcome"] != "done" else ""))
end = robot.episode_end()
print(f"saved: {end.get('frames')} frames, {end.get('duration_s')} s, label {end.get('sidecar')}")