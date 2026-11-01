# Balance experiment artifacts

These curated artifacts keep the training evidence viewable from a fresh checkout. The full run folders remain in local `runs/` and are excluded from Git.

## v4: curriculum + 50 Hz PPO

- `balance-v4-run.json` records 100,096 PPO steps, seed, package versions, and source/config/model hashes.
- `balance-v4-robot-config.json` is the exact runtime config snapshot used by the run.
- `balance-v4-policy.zip` is the resulting SB3 policy checkpoint.
- `balance-v4-evaluation.json` contains the full paired matrix: six surfaces × three total-mass scales × three controllers, one matched reset seed per condition. The nominal total mass is 25 kg; per-link mass redistribution is synthetic.
- `balance-v4-nominal-video-evaluation.json` records the separate nominal-mass video run and its seeds.
- `balance-v4-ankle-pd-video-evaluation.json` records the ankle-PD reference replay and seeds.
- `../assets/balance-v4-surface-matrix.mp4` shows that nominal-mass PPO run across the six surface profiles.
- `../assets/balance-v4-ankle-pd-reference.mp4` shows the ankle-PD baseline on the same profiles.

Result: direct-action PPO fell in all 18 full-matrix cases, with 2.37 s mean survival. Neutral hold completed 5/18; ankle PD completed 16/18 with 7.29 s mean survival. One reset per condition is only an initial diagnostic.

## Residual PPO: current balance foundation

`residual-v1/` holds the four-stage training chain (30k checkpoint, 20k continuation, 50k continuation, then 50k with a lower residual scale), its manifests/configs, the final policy, and exact MJCF. `balance-residual-v1-evaluation.json` has 162 paired episodes: 18 surface/mass conditions × three seeds × three controllers. `../assets/balance-residual-v1-surfaces.mp4` is a separate nominal-mass PPO replay.

At residual scale 0.05, ankle PD and residual PPO both completed 48/54 held-out-seed cases and averaged 7.29 s of an 8 s episode. Per mass scale, both passed 18/18 at 0.8×, 15/18 at 1.0×, and 15/18 at 1.2×. The residual policy matches the stabilizer in this test; it has not demonstrated an improvement. Mass properties and controller gains remain synthetic design allocations.

## v3: corrected model, initial training schedule

- `balance-v3-run.json`, `balance-v3-robot-config.json`, and `balance-v3-evaluation.json` preserve the earlier run's manifest, config, and 18-episode report.
- `../assets/balance-v3-surfaces.mp4` preserves that policy's six-profile replay.

Result: 0 of 18 rollouts avoided a fall. This policy checkpoint is not archived because v4 is the current executable example.

Both experiments use placeholder masses and actuator values. Neither is calibrated to an assembled robot.

## Command-conditioned locomotion: experimental only

- `locomotion-v0/` preserves the 20,224-step PPO checkpoint, exact model/config, first evaluation, and the failed phase-reference comparison.
- `locomotion-v1/` preserves the 100,352-step nominal-only PPO checkpoint plus paired evaluations at mild and full randomization.
- `locomotion-v2/` preserves a 100,352-step PPO run with the corrected randomization curriculum and its full randomized held-out evaluation.
- `locomotion-v3/` preserves the 100,352-step, phase-aware PPO run, its exact model/config/policy and 12-episode held-out evaluation. The gait phase input was added following the phase/frequency seam used in MuJoCo Playground's Apollo joystick task.
- `locomotion-v4/` preserves the 100,352-step experiment that added the hand-authored phase gait as a motion prior under PPO residuals. The run manifest records this reference mode; source snapshots, MJCF, config, policy, evaluation and MP4 are retained.
- `../assets/locomotion-ppo-v1.mp4` through `../assets/locomotion-ppo-v5-gait-prior-residual.mp4` show the five PPO evaluations and the first failed reference. Episode reports include commands, seeds, initial states, randomization samples, velocities, falls, and displacements.

The 20k policy stayed upright but averaged only 0.568 velocity-tracking score and remained near zero velocity. The nominal-only 100k policy scored 0.175 at full randomization (7/12 episodes without falls) and 0.190 at 0.25 randomization (10/12 without falls). The corrected-curriculum policy scored 0.167 at full randomization and avoided falls in 9/12 cases. The phase-aware policy scored 0.174 and avoided falls in 8/12 cases. Adding that hand-authored gait as a prior under PPO residuals scored 0.164 and avoided falls in 5/12. Neither phase input nor this unvalidated prior helped; all policies remained near zero velocity and failed the 0.70 per-command walking criterion. The next useful experiment needs a validated, stable motion prior or walking expert, not further tuning of this heuristic.

Reproduce the archived gait-prior residual evaluation with the exact policy/config/model and reference mode from its run manifest:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m robot_twin.evaluate_locomotion_policy `
  --model docs\design\results\locomotion-v4\policy.zip `
  --episodes 3 --seconds 6 --randomization-intensity 1 --torch-threads 1 `
  --output runs\locomotion-v4-replay.mp4
```

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

## Reproduce the residual-policy evaluation

The archived `residual-v1/run.json` selects `ankle_pd_residual` mode automatically. This command repeats the three-seed paired evaluation and saves a local MP4:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m robot_twin.evaluate `
  --model docs\design\results\residual-v1\balance-residual-v1-policy.zip `
  --config docs\design\results\residual-v1\balance-residual-v1-robot-config.json `
  --mass-scales 0.8 1.0 1.2 `
  --episodes 3 `
  --torch-threads 1 `
  --output runs\balance-residual-v1-replay.mp4
```

The evaluation JSON records the exact seeds, mass distribution, controller type, residual scale, and per-episode outcome. The policy and archived run history are all simulation artifacts; they do not represent physical validation.
