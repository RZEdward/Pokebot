# Project Hitmonlee: learning workbook for Steps 1–4

This project is being implemented for you now so there is a working base to inspect. The goal of this workbook is to make each layer understandable and give you a way to take over the next step yourself.

## The smallest end-to-end path

```text
MJCF (robot + physics) → Gymnasium env (one decision step) → PPO (learn actions)
        │                         │                            │
        ├→ native viewer          ├→ proprioceptive observation├→ checkpoint
        └→ camera renderer        └→ reward / done / info       ├→ evaluation metrics
                                                                └→ live telemetry → browser
```

### Step 1 — why this stack first

The inventoried host is a Windows 11 laptop with an Intel Core Ultra 5 125H (14 cores / 18 logical processors), Intel Arc graphics, 16 GiB RAM and a CPU-only PyTorch install; it has no CUDA device. Native MuJoCo is therefore the small, inspectable physics baseline that fits this machine. Gymnasium's `reset`/`step` contract keeps the task easy to reason about and makes later algorithm/simulator comparisons possible. PPO is the first learner because a small continuous-action baseline is readily available and easy to compare. It is not selected because PPO is always best.

Open the ADR and compare the repo with MuJoCo Playground's train/evaluate/render workflow. The big next experiment is the same task on a supported GPU backend, with identical model, seeds, update cadence, and held-out metrics.

### Step 2 — understand the robot description

Open `src/hitmonlee_sim/models/hitmonlee_planar.xml` and trace one leg from the pelvis through the hip, knee, ankle, and foot. For each joint, identify its axis, range, damping, and actuator. Then inspect masses, contact geometry, timestep, and the camera frame. Run `python -m hitmonlee_sim.inspect_model --output runs/model-audit.json` and compare the compiled model report with the XML. MuJoCo uses world z as up in this model. The root is constrained to x/z/pitch, which makes this a 2D learning exercise; the model cannot test lateral balance.

Next, use a known torque/position step to see how the simulated actuator responds. Once a real actuator exists, log its command, encoder, current/torque estimate, temperature and timestamps, fit the response and replace placeholder gains. Keep the raw logs and calibration version.

### Step 3 — understand what the policy learns

- In `env.py`, inspect observation construction, action scaling, simulation frame-skip, reward, and fall criteria.
- In `train.py`, inspect seed management, PPO rollout length, evaluation callback, checkpoint metadata and telemetry writes.
- Use `--model-path` in training and `--robot-model` in evaluation to pin the exact MJCF. Compare the SHA-256 values in `run.json` and the evaluation JSON before comparing policies.
- Run first with a few thousand steps; compare an untrained policy with the trained checkpoint on the same held-out seeds.
- Plot root height/pitch and return. If training reward rises but deterministic success does not, investigate the reward/reset/action design instead of declaring success.
- Inspect the per-episode `initial_state` beside the fall result. Compare starts that fail for many training seeds with starts that only one policy fails. Change one reset range, keep the evaluation seeds fixed, and see whether the failure pattern changes.
- Recreate the multi-seed summary with `python -m hitmonlee_sim.summarize`; compare its per-seed aggregation to the source evaluation JSON before interpreting a mean success rate.
- The current action means six desired joint positions. It does not mean motor torques. Read the actuator module before interpreting any learned action.

Reference study: MuJoCo Playground separates task configuration/training/evaluation and uses randomization/privileged observations for transfer. For this repo, first understand the non-randomized mechanics; then add one physically justified uncertainty at a time.

### Step 4 — understand the senses and action gate

`visual_perception.py` receives a timestamped calibrated RGB frame and emits a small target cue; its red threshold is only a camera-pipeline exercise. `voice.py` optionally transcribes a command from a local microphone or PCM WAV using Vosk; `commands.py` maps the transcript to a typed allowlist and available skill. None writes a joint command. The behavior layer will later choose among validated skills, and a low-level supervisor will clamp/stop commands. A hardware e-stop remains electrically independent.

Exercises:

1. Record a `head_camera` MP4 and verify the red target appears.
2. Pass one frame to `detect_red_target` and plot the normalized image center/confidence.
3. Change the target's position in MJCF and confirm the cue changes; then add camera noise, delay, and dropped frames.
4. Try accepted and unrecognized transcripts. Make sure unsupported `WALK_FORWARD` is a typed intent but cannot dispatch because no walking policy exists.
5. Install the optional speech extra, download the small Vosk model, and compare a WAV transcription with what was spoken. Check how confidence gating handles noise and unsupported words.
6. Replace the color cue with a trained detector only after you have labeled evaluation data and measurable false-positive/false-negative behavior.

## Your handover point after the foundation

The first learning task should end at a clear gate: reproduce the environment, understand every reward/observation/action term, and compare a trained checkpoint with baseline metrics. Then you can choose the next question with evidence—actuator identification, 3D morphology, walking, or visual target tracking—instead of adding more software layers at once.
