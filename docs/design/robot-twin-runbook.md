# Robot twin runbook

Use the existing project environment from the repository root (PowerShell):

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m robot_twin.train --steps 100000 --run-dir runs\my-balance-run --device cpu --control-mode ankle_pd_residual --residual-scale 0.05 --torch-threads 2
.\.venv\Scripts\python.exe -m robot_twin.evaluate --model runs\my-balance-run\policy.zip --output runs\my-balance-run\surfaces.mp4
```

The evaluator runs paired neutral-hold, configurable ankle-PD, and PPO controllers on the same seeded initial state for each named surface and mass scale, with synthetic per-link mass redistribution. By default it checks 0.8×, 1.0×, and 1.2× total mass; use `--mass-scales 1.0` for a lighter nominal-only pass. It writes a detailed `.evaluation.json` and an MP4 showing PPO by default; `--record-controller ankle_upright_pd` records the reference controller instead. Training uses two PyTorch threads by default, and evaluation uses one. Lower `--torch-threads` if the laptop is getting hot. For a live simulation window, add `--live` to training or evaluation (use only one interactive simulation window). Without a GUI, start `.\.venv\Scripts\python.exe -m robot_twin.monitor --run-dir runs\my-balance-run`, then open `http://127.0.0.1:8770`. Training snapshots update the 3D MuJoCo view and display telemetry/joint positions. For experiment progress, read `telemetry.json` and `run.json` in the run directory. `runs/` is intentionally ignored by Git; preserve important run bundles separately with their model/config hashes.

The stand environment trains balance from randomized near-upright states. The separate command-conditioned velocity task is experimental and has not passed its walking criterion. Other named curriculum entries remain future tasks. These are simulation studies, not a physically calibrated digital twin.

## Experimental planar locomotion

The command-conditioned velocity-tracking task is experimental and has not passed the walking criterion. Its four configured body-frame commands are forward, reverse, and lateral translations; yaw is reserved and not trained. Five PPO variants (20k to 100k transitions) stayed near zero velocity. Adding gait-phase observations did not improve the result. Training a residual over the hand-authored gait reference performed worse (0.164 score; 5/12 no-fall episodes) than the ankle-PD residual (0.174; 8/12). The evaluator records sampled mass, friction, slope, compliance, and initial state for each episode. A reliable next step is to source or construct a validated motion prior or walking expert, verify the expert on this model, then test a bounded residual. The phase-conditioned PPO preprint and [RuN residual locomotion paper](https://arxiv.org/abs/2509.20696) are research leads; they are not dependencies in this repository.

Replay the archived phase-aware experiment and open the curated report:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m robot_twin.evaluate_locomotion_policy `
  --model docs\design\results\locomotion-v3\policy.zip `
  --episodes 3 --seconds 6 --randomization-intensity 1 --torch-threads 1 `
  --output runs\locomotion-v3-replay.mp4
Start-Process docs\design\phase1-4-progress.html
```

To launch a new experiment rather than replay the archived one:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m robot_twin.train_locomotion --steps 100000 --run-dir runs\walk-ppo --torch-threads 1
.\.venv\Scripts\python.exe -m robot_twin.train_locomotion --steps 100000 --seed 61 --reference-mode phase_gait_plus_ankle_pd --run-dir runs\walk-ppo-prior --torch-threads 1
.\.venv\Scripts\python.exe -m robot_twin.monitor --run-dir runs\walk-ppo --port 8775
.\.venv\Scripts\python.exe -m robot_twin.evaluate_locomotion_policy --model runs\walk-ppo\policy.zip --episodes 3 --seconds 6 --output runs\walk-ppo\held-out-commands.mp4
```

The evaluator pairs PPO and phase-reference trajectories on the same seeded commands. The success target requires surviving and maintaining a mean command-tracking score of at least 0.70 in every held-out command across three seeds. An unpassed score is kept as experimental evidence, not called walking. Model masses/inertias, contact, and actuator behavior remain provisional.

Every 10,000 steps, training atomically saves `policy-checkpoint.zip`. To continue after an interrupted session, create a new run directory and resume from that run's checkpoint; the trainer verifies that the model and config snapshots match before loading it:

```powershell
.\.venv\Scripts\python.exe -m robot_twin.train_locomotion `
  --steps 100000 `
  --run-dir runs\walk-ppo-resumed `
  --resume runs\walk-ppo\policy-checkpoint.zip `
  --torch-threads 1
```

`--steps` is the additional transition budget for the resumed segment. The new run records the source run and starting timestep.

Use `--episodes 3` or more for multi-seed evaluation. The archived residual policy under `docs/design/results/residual-v1/` was trained in `ankle_pd_residual` mode; its accompanying `run.json` makes the evaluator select that mode automatically.
