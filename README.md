# Understudy

Show two robot arms a task, in words or in a video of a person doing it. Muse Spark drives the arms one small step at a
time and looks after every step, SAM 3.1 finds the things on the table, and every motion is checked before it moves.
Each run can be recorded and trained into an ACT policy.

Our entry to Meta's Global AI Developer Hackathon 2026 (Vlad Marian, Andrei Voinea).

The film (2:51): https://youtu.be/ZgwyoJ4Eaok

## The Meta models

Both through Meta's Model API.

| Model | What it does here |
|---|---|
| Muse Spark (`muse-spark-1.3`) | Watches the person's video and says what each hand does and which objects matter. Gives each object's kind, size and colour for the simulator. Drives the arms, one action per turn, from the overhead and both wrist images. Judges recorded episodes |
| SAM 3.1 (`sam-3.1`) | Segments the objects Muse named, and both hands, in every frame of the video: who held what, when. Finds named objects in the overhead image and places them on the table in cm |

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

Or the whole loop from the dashboard, which runs its own simulator (don't start robotd for it):

```bash
python -m ui --sim        # http://127.0.0.1:8000: drop in watch/test_video_balls.mp4 and describe the task
```

The simulator takes about two minutes to start.

## From a phone video to the arms

The dashboard takes a video through every stage and shows each one live:

1. **Watch.** Muse Spark watches the video and writes down what each hand does and which objects matter.
2. **Segment.** SAM 3.1 segments those objects and both hands in every frame; the score says who held what, when
   (`watch/SCORE.md`).
3. **Scene.** Muse gives each object's kind, size and colour, and the simulator is rebuilt with them on the table where
   the arms reach (`watch/simscene.py`). With `python -m ui` on the real arms, this stage is skipped.
4. **Act.** The brain does the task from Muse's reading of the video, one small step at a time (below), recorded as an
   episode.
5. **Judge.** Muse looks at the recorded episode and says whether it worked, which decides whether it trains. Episodes
   stop at 3 minutes, so longer runs aren't judged yet.

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

## Licence

Apache-2.0 (`LICENSE`). The SO-101 arm model in `harness/assets/` is TheRobotStudio's SO-ARM100, also Apache-2.0: where
each file comes from and what changed is in `harness/assets/SOURCE.md`.
