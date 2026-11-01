# First-run acceptance

This is the evidence to gather for the current runnable slice, not a claim of hardware validity.

1. Install the project dependencies in the documented Python range and record exact versions in each run manifest.
2. Compile `src/hitmonlee_sim/models/hitmonlee_planar.xml` with MuJoCo and inspect the robot in the viewer.
3. Run a short headless PPO training job with a fixed seed. Confirm it writes `run.json`, `progress.csv`, checkpoints, and `telemetry.json`.
4. Run deterministic evaluation with the callback-selected best checkpoint and confirm per-episode results are recorded.
5. Run the local monitor and confirm it shows live pose/reward telemetry from that exact run directory.
6. Record a short `rgb_array` rollout to MP4 and verify that it plays and shows the simulated model.
7. Compare the trained checkpoint with a neutral zero-action controller on the same evaluation seeds; report the metrics honestly. Do not call the policy successful if it does not improve survival or stability.

## Run performed on 1 November 2026

- Installed the project editable in a clean `.venv` on Windows 11 / Python 3.13.
- Compiled the packaged MJCF, confirmed 17.2 kg summed body mass, and checked that both feet contact the floor at reset.
- Trained PPO for 50,000 environment steps on CPU (about 520 simulated decisions/s during this run). Held-out evaluation was non-monotonic: the callback selected the 30,000-step checkpoint rather than the final checkpoint.
- That selected checkpoint held balance for the full 8 seconds in all three deterministic evaluation episodes at seeds 9000–9002 (400 decisions each), with max absolute torso pitch about 0.10 rad. This is a narrow demonstration on one provisional planar model, not a statistically strong robustness claim.
- A zero-action nominal-pose baseline on the exact same seeds fell after 92–99 decisions (1.84–1.98 simulated seconds). This shows a clear improvement for this narrow task and model; it is not evidence of walking or transfer.
- A wider follow-up on seeds 12000–12029 completed 25/30 episodes (83.3%); falls occurred at seeds 12010, 12020–12022 and 12029. The zero-action baseline fell on all 30, with a mean 108.1 steps versus 358.7 for the policy. See `runs/balance-v1/heldout-30-comparison.json` for per-seed data.
- Three controlled repeats used training seeds 43, 44 and 45 with the same 50,000 steps, PPO settings, 5,000-step evaluation cadence, and exact MJCF hash as v1. Their callback-selected checkpoints completed 9/30, 4/30 and 6/30 held-out starts, respectively; seed 42 completed 25/30. The best checkpoints were selected at 30k, 10k, 25k and 40k steps for seeds 42–45. Across the four runs the descriptive mean was 11/30 (36.7%) and median was 7.5/30; the observed range was 4/30–25/30. Each evaluation report includes a nominal Wilson 95% interval for its 30-start success rate, alongside every episode. Per-run data is in `runs/balance-multiseed-summary.json` and each `runs/balance-v1` and `runs/balance-exact-v43` through `balance-exact-v45` folder. This is a warning about between-run variation, not a formal robustness estimate.
- The same 30 evaluation starts are now recorded with each episode's reset pose and velocity. Three seeds (12010, 12020 and 12029) failed across all four selected policies; every start caused at least one of the four policies to fall. This identifies useful cases to inspect, but it does not yet explain why those starts are harder. The cross-run breakdown and initial state records are in `runs/balance-multiseed-summary.json` and the per-run evaluation JSON files.
- Extending seed 43 to 100,000 steps with evaluation every 10,000 selected the same 10,000-step checkpoint; it still completed 9/30. Its final checkpoint completed 0/30. See `runs/balance-v2/best/best_model.evaluation.json` and `runs/balance-v2/final_model.evaluation.json`. More training did not help this run; v1 remains the teaching/demo checkpoint. A larger controlled multi-seed study is needed before calling either policy robust.
- Recorded `runs/balance-v1/best_rollout.mp4` from the overview camera and generated a head-camera frame. The simple red-blob detector returned the target near the horizontal image center.
- The model SHA-256 in `run.json` matches the exact MJCF used for learning. Later changes only added offscreen rendering dimensions and lighting; both the training model hash and the current render-model hash are recorded.
- Ran a short training process while polling the browser monitor every 200 ms. Training completed 2,048 steps, the monitor observed 14 changing telemetry samples, and the Windows file-replacement race did not recur.
- Downloaded the official Vosk small English model outside the source tree and passed four mono 16 kHz PCM WAV clips through it: “stand” dispatched to `balance`, “walk forward” was rejected because the skill is unavailable, “stop” produced a safe-stop request, and “don't stop” was rejected as unknown. All were synthesized on Windows and scored confidence 1.0; real-speaker accuracy and microphone quality remain to be measured.
- Walking, 3D lateral balance, learned visual perception, and a physical actuator model remain unimplemented.

Future gates: calibrate actuator dynamics on a constrained bench, update mass/inertias from actual CAD and hardware, validate the same low-level command semantics in a one-joint ROS 2 hardware interface, then stage single-leg/lower-body tests under a tether/harness before walking trials.
