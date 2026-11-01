# Robot twin runbook

Use the existing project environment from the repository root (PowerShell):

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m robot_twin.train --steps 100000 --run-dir runs\my-balance-run --device cpu --control-mode ankle_pd_residual --residual-scale 0.05 --torch-threads 2
.\.venv\Scripts\python.exe -m robot_twin.evaluate --model runs\my-balance-run\policy.zip --output runs\my-balance-run\surfaces.mp4
```

The evaluator runs paired neutral-hold, configurable ankle-PD, and PPO controllers on the same seeded initial state for each named surface and mass scale, with synthetic per-link mass redistribution. By default it checks 0.8×, 1.0×, and 1.2× total mass; use `--mass-scales 1.0` for a lighter nominal-only pass. It writes a detailed `.evaluation.json` and an MP4 showing PPO by default; `--record-controller ankle_upright_pd` records the reference controller instead. Training uses two PyTorch threads by default, and evaluation uses one. Lower `--torch-threads` if the laptop is getting hot. For a live simulation window, add `--live` to training or evaluation (use only one interactive simulation window). Without a GUI, start `.\.venv\Scripts\python.exe -m robot_twin.monitor --run-dir runs\my-balance-run`, then open `http://127.0.0.1:8770`. Training snapshots update the 3D MuJoCo view and display telemetry/joint positions. For experiment progress, read `telemetry.json` and `run.json` in the run directory. `runs/` is intentionally ignored by Git; preserve important run bundles separately with their model/config hashes.

The current objective is standing from randomized near-upright states. Do not assume the named curriculum entries have been implemented: `robot_config.json` lists only `proprioceptive_stand` as implemented. This task is not a physically calibrated digital twin and trained policies are simulation artifacts only.

Use `--episodes 3` or more for multi-seed evaluation. The archived residual policy under `docs/design/results/residual-v1/` was trained in `ankle_pd_residual` mode; its accompanying `run.json` makes the evaluator select that mode automatically.
