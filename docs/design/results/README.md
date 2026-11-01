# Balance experiment artifacts

These curated artifacts keep the training evidence viewable from a fresh checkout. The full run folders remain in local `runs/` and are excluded from Git.

## v4: curriculum + 50 Hz PPO

- `balance-v4-run.json` records 100,096 PPO steps, seed, package versions, and source/config/model hashes.
- `balance-v4-robot-config.json` is the exact runtime config snapshot used by the run.
- `balance-v4-policy.zip` is the resulting SB3 policy checkpoint.
- `balance-v4-evaluation.json` contains the full paired matrix: six surfaces × three total-mass scales × two controllers, one matched reset seed per condition. The nominal total mass is 25 kg; per-link mass redistribution is synthetic.
- `balance-v4-nominal-video-evaluation.json` records the separate nominal-mass video run and its seeds.
- `../assets/balance-v4-surface-matrix.mp4` shows that nominal-mass PPO run across the six surface profiles.

Result: PPO fell in all 18 full-matrix cases, with 2.37 s mean survival. Neutral hold completed 5/6 low-mass cases and fell in the other 13; its overall mean was 4.51 s. By mass scale, neutral hold completed 5/6 cases at 0.8×, 0/6 at 1.0×, and 0/6 at 1.2×; PPO completed 0/6 at each scale. The policy regressed against this simple baseline. One reset per condition is only an initial diagnostic, not a robustness estimate.

## v3: corrected model, initial training schedule

- `balance-v3-run.json`, `balance-v3-robot-config.json`, and `balance-v3-evaluation.json` preserve the earlier run's manifest, config, and 18-episode report.
- `../assets/balance-v3-surfaces.mp4` preserves that policy's six-profile replay.

Result: 0 of 18 rollouts avoided a fall. This policy checkpoint is not archived because v4 is the current executable example.

Both experiments use placeholder masses and actuator values. Neither is calibrated to an assembled robot.

## Reproduce v4 evaluation

From the repository root, this runs the archived policy on the full matrix and writes the report under `runs/`:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m robot_twin.evaluate `
  --model docs\design\results\balance-v4-policy.zip `
  --config docs\design\results\balance-v4-robot-config.json `
  --mass-scales 0.8 1.0 1.2
```

The adjacent `robot_model.xml` is the exact MJCF hash recorded in the run manifest. Add `--output runs\balance-v4-replay.mp4` to save a new PPO replay.
