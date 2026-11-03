# Understudy

Show two robot arms a task, in words or in a video of a person doing it. Muse Spark drives the arms one small step at a
time and looks after every step, SAM 3.1 finds the things on the table, and every motion is checked before it moves.
Each run can be recorded and trained into an ACT policy.

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

1. **Look.** Every turn Muse gets the overhead image with a cm grid and both grippers drawn on, both wrist images with
   the fingertips and the grip point drawn on, and the arms' numbers.
2. **Act.** It answers with one small action for one arm: a step in the gripper's own directions, a move to a table
   point, a tilt or a wrist roll, open, close (until the jaws meet a load), home, or find (SAM 3.1 on the overhead
   image, in table cm). robotd checks every motion before it moves.
3. **Guard.** Rules in the loop refuse what failed on the real arms: closing before a thing is deep in the jaws,
   letting go before what it holds is over the opening, tilting a held container, the arms crowding each other.
4. **Again**, until Muse says the task is done.

To change the rig, change `harness/scene.json`; to change the model, set `LLM=` (`brain/llm.py`).

| Folder | What it does |
|---|---|
| `brain/` | the Muse loop: `loop.py` the loop, everything Muse is told and the rules, `see.py` the images it gets and what SAM finds, `sam.py`, `llm.py` |
| `harness/` | `robotd` (the only process on the servos): kinematics, checked motion, the 50 Hz controller, the sims, the recorder, the policy runner, training, calibration |
| `watch/` | a video of a person into a score, with SAM 3.1 and Muse |
| `judge/` | Muse's verdict on recorded episodes |
| `cameras/` | the camera server |
| `ui/` | mission control: the rig live and the whole loop from one page (`python -m ui`) |

## Train a policy

```bash
python -m brain "<task>" --record                   # the run is an episode in robotd's --dataset, labelled with it
python -m harness.train <dataset>
```

## Safety

Only robotd touches the servos, only a person can switch torque on, and no API call can. The models never reach the
motors: every motion is checked for reach, the table and the gap between the arms before anything moves.

## Tests

```bash
python -m pytest harness/tests brain/tests judge/tests watch/tests ui/tests -q
```
