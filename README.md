# Understudy

Show two robot arms a task, in words or in a video of a person doing it. Muse Spark plans it, SAM 3.1 finds the things
on the table, and every motion is checked before it moves. Each run can be recorded and trained into an ACT policy.

Our entry to Meta's Global AI Developer Hackathon 2026 (Vlad Marian, Andrei Voinea).

## Try it in the simulator

No arms or cameras needed, only a Model API key.

```bash
conda create -n lerobot python=3.12 -y && conda activate lerobot
pip install -r requirements.txt
echo "MODEL_API_KEY=..." > .env

./understudy.sh --sim --hd --objects balls        # robotd on the MuJoCo sim (scenes: balls, cubes, blocks, brush)
python -m brain "put the tennis balls in the tube"
python -m brain --video watch/test_video_balls.mp4
```

Each run's steps, images, tokens and cost go to `logs/brain/`.

## On real arms

Two SO-101 followers and three UVC cameras (one overhead, one on each wrist).

1. Put your ports and camera modes in `harness/scene.json` and `cameras/cameras.json`.
2. Calibrate the servos with `lerobot-calibrate`, under the ids in `scene.json`.
3. Run `cameras/setup.sh` once, for the camera server.
4. Run `./understudy.sh` and press `a` in it to switch torque on.
5. Calibrate the rig with a ChArUco board on a tablet: `python -m harness.calib_board` (its docstring has the steps).
6. Run `python -m brain "<task>"`.

Space stops the arms. `./understudy.sh check` says whether everything is ready.

## How it works

1. **Look.** Muse names what the task involves and SAM finds those things in the overhead image, in table cm.
2. **Plan.** One Muse call turns the task, the table and what the arms can reach into a list of steps.
3. **Act.** The steps run with no model calls in between. robotd checks each motion first.
4. **Check.** After a step that asks for it, or one that failed, Muse looks at fresh images and goes on, re-plans or stops.

No task lives in the code. To change the rig, change `harness/scene.json`; to change the model, set `LLM=` (`brain/llm.py`).

| Folder | What it does |
|---|---|
| `brain/` | the Muse loop: `plan.py` the loop, `prompt.py` everything Muse is told, `act.py` the arms' actions, `tools.py` what it sees and remembers, `reach.py`, `wrist.py`, `sam.py`, `video.py`, `llm.py` |
| `harness/` | `robotd` (the only process on the servos): kinematics, checked motion, the 50 Hz controller, the sims, the recorder, the policy runner, training, calibration |
| `watch/` | a video of a person into a score, with SAM 3.1 and Muse |
| `judge/` | Muse's verdict on recorded episodes |
| `cameras/` | the camera server |

## Train a policy

```bash
python -m brain "<task>" --plan-only --save-plan plans/mine.json
python -m brain.rehearse plans/mine.json ...        # replays the plan and records episodes; its docstring has the options
python -m harness.train <dataset>
```

`brain.rehearse` currently judges success by whether objects ended stacked (`--on`).

## Safety

Only robotd touches the servos, only a person can switch torque on, and no API call can. The models never reach the
motors: every motion is checked for reach, the table and the gap between the arms before anything moves.

## Tests

```bash
python -m pytest harness/tests brain/tests judge/tests watch/tests -q
```
