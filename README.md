# Project Hitmonlee

Simulation-first engineering workspace for a custom Hitmonlee-inspired humanoid robot.

## Current executable slices

The repository contains two MuJoCo paths:

- `robot_twin` is the new generic **3D 28-DoF humanoid** foundation. It has a free pelvis, explicit per-link mass/inertia allocations, 28 finite-gain torque-limited position actuators, collision proxies, named sensor frames, CAD-reference validation, PPO standing training, a local live monitor, and replayable surface evaluation.
- `hitmonlee_sim` retains the earlier **planar biped** example and its command/voice experiments. It remains useful as a small, reproducible baseline; it is not the 3D body model.

The 3D model is a provisional simulation embodiment, not a calibrated digital twin or manufacturing release. Its 25 kg mass case, link mass/inertias, actuator limits, contact compliance and 1.35 m working height are design assumptions until components are selected and physical measurements are made. Only proprioceptive standing is implemented; walking and whole-body action skills remain untrained.

See [the phase 1–4 engineering report](docs/design/phase1-4-progress.html), [robot-twin architecture](docs/design/robot-twin-architecture.md), and [runbook](docs/design/robot-twin-runbook.md).

### Train the 3D embodiment

~~~powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m robot_twin.train --steps 100000 --run-dir runs\my-balance --device cpu
.\.venv\Scripts\python.exe -m robot_twin.monitor --run-dir runs\my-balance
~~~

Open `http://127.0.0.1:8770` to watch the local snapshot/telemetry dashboard. After training, record a deterministic run on all configured surface profiles:

~~~powershell
.\.venv\Scripts\python.exe -m robot_twin.evaluate --model runs\my-balance\policy.zip --output runs\my-balance\surfaces.mp4
~~~

The evaluation writes a JSON report next to the MP4. `runs/` is ignored by Git so checkpoints and videos stay local unless deliberately archived. The evaluator currently covers nominal, low/medium friction, forward/lateral slope, and compliant-contact cases. A “compliance” profile is an effective MuJoCo contact-softness setting, not a measured foam model.

## Earlier planar example

`hitmonlee_sim` is the original 2D balance/voice/command path. Its plan-view simplification cannot demonstrate lateral balance, 3D walking, or whole-body skills.

The current host is a Windows 11 laptop with an Intel Core Ultra 5 125H (14 cores / 18 threads), Intel Arc graphics, and 16 GiB RAM. The installed PyTorch build is CPU-only; this machine exposes no CUDA device. The baseline therefore uses native MuJoCo CPU and Stable-Baselines3 PPO. MJWarp, mjlab and Isaac Lab's Newton GPU path are NVIDIA-Warp/CUDA candidates for a later workstation or remote GPU, not acceleration options for this Intel Arc laptop. Genesis World currently documents CPU physics on Intel GPUs, with the Intel GPU used for viewing/rendering; it is a possible cross-stack CPU comparison if a concrete need arises. Isaac Sim is deferred until camera/synthetic-data work and suitable NVIDIA hardware justify the added runtime.

## Quick start (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Train headless and write checkpoints, evaluation logs, and live telemetry:

```powershell
.\.venv\Scripts\python.exe -m hitmonlee_sim.train --steps 50000 --run-dir runs\balance-01
```

Train at approximately real time while watching the MuJoCo window (this intentionally reduces training throughput):

```powershell
.\.venv\Scripts\python.exe -m hitmonlee_sim.train --steps 50000 --live --run-dir runs\balance-live
```

Open the live browser monitor in another terminal while training:

```powershell
.\.venv\Scripts\python.exe -m hitmonlee_sim.dashboard --run-dir runs\balance-live
```

Then open <http://127.0.0.1:8765>. The monitor shows the robot's planar pose, reward/height history, and episode state. It reads telemetry from the run directory and is local-only.

After training, render the selected best policy to an MP4:

```powershell
.\.venv\Scripts\python.exe -m hitmonlee_sim.evaluate --model runs\balance-01\best\best_model.zip --episodes 3 --record runs\balance-01\rollout.mp4
```

Use `--camera head_camera` to record the robot's RGB sensor view. Open the MP4 in any player. To get evaluation metrics without recording a video, omit `--record`.

## Current training evidence

Across four controlled 50,000-step training seeds (42–45), the best checkpoints completed 25/30, 9/30, 4/30 and 6/30 randomized held-out starts. The descriptive mean was 11/30; the range was 4–25 completions. Three reset seeds (12010, 12020 and 12029) failed for all four policies. A zero-action baseline completed 0/30. Extending seed 43 to 100,000 steps selected the same 10,000-step best checkpoint, while its final checkpoint completed 0/30. This is evidence of large run/checkpoint sensitivity and shows why training return alone is inadequate; it is not a formal robustness estimate. Evaluation JSON now records initial pose and velocity to make those cases inspectable. The run history, reported outcomes, reproducibility details, and limits are summarized in [`docs/verification/first-run.md`](docs/verification/first-run.md) and the [visual overview](docs/overview/phase_1_to_4_foundation.html). Generated checkpoints, per-seed JSON, and videos stay under `runs/` locally and are excluded by `.gitignore`.

To reproduce the controlled seed-43 run against the exact MJCF archived with seed 42:

```powershell
.\.venv\Scripts\python.exe -m hitmonlee_sim.train --steps 50000 --seed 43 --n-steps 512 --eval-every 5000 --model-path runs\balance-v1\training-model.xml --run-dir runs\balance-exact-v43
.\.venv\Scripts\python.exe -m hitmonlee_sim.evaluate --model runs\balance-exact-v43\best\best_model.zip --robot-model runs\balance-v1\training-model.xml --episodes 30 --seed 12000
```

The per-episode evaluator writes JSON beside the checkpoint, including robot-MJCF hash and randomized reset state. Evaluation seeds 12000–12029 are the same ones used to compare all policies and the zero-action baseline.

Regenerate the four-seed summary from the run manifests and held-out evaluation records:

```powershell
.\.venv\Scripts\python.exe -m hitmonlee_sim.summarize `
  --run-dir runs\balance-v1 `
  --run-dir runs\balance-exact-v43 `
  --run-dir runs\balance-exact-v44 `
  --run-dir runs\balance-exact-v45 `
  --output runs\balance-multiseed-summary.json
```

The summarizer checks that task, robot-MJCF hash, PPO and reset settings, step count and evaluation cadence match, and that every run used the same held-out seed set. The training-run seed and callback evaluation seed are allowed to differ by design.

## Architecture

```text
robot specification + MJCF model
                 │
                 ├── MuJoCo environment (physics, actuator model, sensor frames)
                 │        ├── proprioceptive observation/action contract
                 │        ├── PPO trainer → checkpoints + metrics + telemetry
                 │        ├── live MuJoCo viewer
                 │        └── local browser monitor → replay renderer → MP4
                 │
                 ├── current sensor prototypes
                 │        ├── head-camera RGB → red-target cue
                 │        └── optional Vosk ASR → typed skill request
                 └── future adapters
                          └── URDF / ros2_control hardware interface
```

## Character appearance study

The mechanics model intentionally stays independent from the surface design. A generated, Hitmonlee-inspired industrial-design concept and its review notes are in [`docs/design/hitmonlee-robot-appearance.html`](docs/design/hitmonlee-robot-appearance.html); the source image and exact prompt are in `assets/concepts/`. Treat the render as a silhouette/material direction only. It does not define dimensions, collision, mass properties, actuator packaging, or a safe build. A scaled front/side silhouette now gives the design a reviewable 1.5 m character reference. The next CAD gate is to reconcile that scale with the 1.2–1.4 m project envelope, then fit real motor/gearbox volumes and center-of-mass constraints.

Inspect the model contract and reset contacts before editing the MJCF:

```powershell
.\.venv\Scripts\python.exe -m hitmonlee_sim.inspect_model --model src\hitmonlee_sim\models\hitmonlee_planar.xml --output runs\model-audit.json
```

The report checks required joints and actuators, mass, timestep, head camera, and floor contacts at reset. It also prints the MJCF SHA-256, joint limits, servo gains, and force ranges. All current mechanics values remain provisional.

The learned policy is one skill. It does not access hardware or issue bus commands directly. A future deployment path will place it behind a supervisor that enforces joint/torque limits, watchdogs, safe-stop behavior, and a tested command interface. Perception and voice inputs will produce typed intents and state estimates; they will not map language directly to motor outputs.

## Repository map

- `docs/overview/phase_1_to_4_foundation.html` — visual overview of the runnable base, training evidence, live viewing, and upstream references.
- `docs/design/hitmonlee-robot-appearance.html` — generated industrial-design concept, editable proportion sheet, design critique and CAD gate.
- `docs/design/body-design-and-motion-validation.md` — CAD/motion distinction, current model boundaries, validation plan and staged path toward calibrated simulator and supplier geometry.
- `design/cad/body-layout-v1.json` + `tools/cad/build_hitmonlee_body_v1.FCMacro` — reproducible FreeCAD concept envelope assembly inputs and generator; not a manufacturing release.
- `design/cad/body-v1/` — editable FCStd, concept STEP, shell-only STEP/STL prototypes, validation report and engineering preview images. The last verified full STEP round-trip passed 87/87 solids, and all 87 solids carry their JSON source path and resolved value. The JSON/generator now define a 28-joint study chain, but the saved FCStd/report still contain the earlier 14 axis references pending rebuild. The shell STL contains 30 hollow split panels (285,718 facets); the shell STEP round-trip imports 30 valid features but narrowly misses its volume-delta threshold. All geometry remains prototype-only; physical fit and manufacturing validation are open.
- `design/cad/reference-hitmonlee/` — licensed appearance reference import and attribution; source mesh is visual reference only.
- `assets/concepts/` — versioned appearance studies and image-generation prompt provenance.
- `docs/research/step_1_to_3_research_proposal.html` — visual toolchain research and staged proposal.
- `docs/planning/` — original project brief and engineering concept.
- `docs/research/step_1_to_3_research_proposal.md` — readable research and staged proposal.
- `docs/research/open-source-reference-workflows.md` — architecture patterns and upstream references.
- `docs/architecture/` — software boundaries, state/action schemas, and model-format decisions.
- `docs/architecture/voice-command-pipeline.md` — optional offline ASR and safe intent-dispatch path.
- `src/hitmonlee_sim/models/` — packaged simulator-native MJCF model; runtime and distribution use the same source.
- `src/hitmonlee_sim/` — Gymnasium environment, PPO training/evaluation, dashboard, camera cue, typed command gate, and optional offline voice input.
- `src/hitmonlee_sim/summarize.py` — validates comparable run manifests and aggregates held-out multi-seed results.
- `src/hitmonlee_sim/inspect_model.py` — audits MJCF structure, actuator limits and reset floor contacts.
- `src/hitmonlee_sim/static/` — dependency-free live monitor page served by the local dashboard.
- `src/robot_twin/` — generic 3D humanoid embodiment, joint/CAD validation, surface-randomized balance environment, PPO training/evaluation, run snapshotting, and camera-render monitor.
- `design/cad/body-v2/` — editable FreeCAD document, concept STEP, prototype shell exports, and validation report.
- `runs/` — generated checkpoints, CSV/JSON logs, and videos; ignored by Git by default.

## Stages and limits

1. **Toolchain:** native MuJoCo is the runnable reference; compare Newton/Isaac Lab when GPU/platform constraints and model asset are known.
2. **Robot model:** the new 3D model has validated neutral joint frames, link inertias, contact feet, joint limits, finite-gain/torque-saturated position actuators, and reserved sensor mounts. Mass and actuator values remain provisional until components and measurements are available.
3. **Motion learning:** a proprioceptive PPO standing task and paired seeded surface evaluations are implemented. The latest 100,096-step policy failed every case and underperformed neutral hold; this is a functional training example, not a successful balance skill. Weight shifts, recovery, multi-direction walking, sitting/get-up, prone/supine recovery, jump, kick, and punch remain separate gated tasks.
4. **Senses/commands:** the older planar MVP renders a head-camera frame, extracts a deliberately simple red-target cue, and maps transcript text into allowlisted typed intents. The 3D model reserves eight camera frames and four microphone sites plus one speaker site; the camera/audio policies and perception calibration are not implemented.

Try the transcript-to-skill gate without an ASR dependency:

```powershell
.\.venv\Scripts\python.exe -m hitmonlee_sim.command_cli "stand"
.\.venv\Scripts\python.exe -m hitmonlee_sim.command_cli "walk forward"
.\.venv\Scripts\python.exe -m hitmonlee_sim.command_cli "stop"
```

`stand` resolves to the registered balance skill; walking is rejected until a walking skill exists. `stop` only produces a safe-stop request for a future supervisor. It is never a substitute for an independent physical emergency-stop circuit.

For offline voice input, install the optional speech libraries and download the official small US-English model (about 40 MB):

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[speech]"
.\.venv\Scripts\python.exe -m hitmonlee_sim.voice --download-model
.\.venv\Scripts\python.exe -m hitmonlee_sim.voice --listen
```

To transcribe a mono 16 kHz, 16-bit PCM WAV instead, run `.\.venv\Scripts\python.exe -m hitmonlee_sim.voice --audio path\to\command.wav`. The model and its source/license record are stored under ignored `runs/asr/`; the model is not bundled with the repo. Speech is processed locally. Accent and microphone performance still need evaluation with your own voice.

Read `docs/architecture/system-design.md` before extending the packages and `docs/verification/first-run.md` for the current acceptance procedure.
