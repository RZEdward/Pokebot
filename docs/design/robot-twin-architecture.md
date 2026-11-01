# Robot twin architecture and phase-one status

## The model is a versioned engineering hypothesis

The source chain is intentionally explicit:

1. `design/cad/body-layout-v2.json` is the mechanical packaging and coordinate-frame study. `tools/cad/build_hitmonlee_body_v2.FCMacro` produces the editable FreeCAD assembly, STEP envelopes, and prototype shell outputs. Those CAD outputs are still marked concept / not for manufacture.
2. `src/robot_twin/robot_config.json` selects one simulation embodiment and its task/actuator assumptions. It is robot-neutral in schema; the ID and dimensions identify this particular embodiment.
3. `src/robot_twin/models/humanoid_28dof.xml` is the runnable MuJoCo multibody model: free pelvis, 28 revolute joints, finite-gain torque-limited position actuators, collision geometry, reserved eye cameras and microphone/speaker sites, and a floor contact model.
4. `src/robot_twin/env.py` implements the Gymnasium contract: seeded reset, proprioceptive state, control decimation, contact variation, reward, termination and metrics. `train.py` and `evaluate.py` own experiments; model state and run metadata are persisted under ignored `runs/`.

CAD and MJCF share named joints and reference dimensions, but their geometry is not yet generated from one parametric source. `robot_config.json` snapshots the CAD joint frame/axis/range contract; `model_validation.py` checks every compiled MJCF joint against it and checks cameras, acoustic sites, total mass, and neutral COM on environment startup. Current maximum joint anchor error is below numerical tolerance. A later architecture milestone can generate both formats from one canonical parametric source rather than duplicating dimensions. Do not treat either this first XML or the current CAD assembly as build-to-print hardware.

The fixed nominal inertial allocation sums to exactly 25 kg and corresponds to the CAD placeholder ledger. This is only a consistency check for simulation. It is not a mass prediction: no battery, motors, transmissions, bearings, fasteners, wiring, shell material, electronics, vendor limits, measured link COMs, or assembled weighing have authorized the physical mass properties. Actuator position gains/torque caps are likewise tunable placeholders, not component selections. No controller output is suitable for physical deployment.

## Package boundaries

- `hitmonlee_sim` contains the older planar biped experiment and its existing local telemetry UI. It is retained as a reproducible baseline; do not import its task logic into new robot environments.
- `robot_twin` contains generic simulation, configuration, learning, evaluation, and visualization interfaces. Future robot embodiments should add a config/model asset and task adapters, not Pokémon-specific branches in the simulator loop.
- `design/cad` and `tools/cad` are mechanical engineering artifacts and builders. STEP/STL files have their own QA status and cannot be inferred to be physically valid from the MJCF results.
- `docs` records evidence, assumptions, and reproducible commands; `runs` contains local non-source artifacts and is ignored by Git.

Audio and vision currently have frames only. MuJoCo camera renders can be captured, but the policy receives no image/audio features; microphones have no waveform/acoustic propagation simulation, and the speaker has no sound source model. Keep time synchronization, calibration records, raw/simulated observations, and sensor interfaces outside reward/task code so later modalities can be added without changing robot identity or controller APIs.

## Experiment status (2026-11-01)

- The XML compiles with 35 generalized coordinates (7 free-base + 28 joints), 34 velocities, and 28 actuators. Its compiled aggregate inertial mass is 25.0 kg and computed neutral COM is 0.7081 m, 4.6 mm from the provisional CAD ledger's 0.7126 m.
- PPO executed for 10,000 steps on CPU and emitted a checkpoint, metadata, telemetry, surface evaluation JSON and an MP4. This is a real training pipeline exercise, not a successful learned balance result: evaluation fell on all six recorded surface cases in 76–88 control steps (0.76–0.88 simulated seconds). Those outcomes are stored in `runs/phase1-humanoid-balance-10k/` on the development machine and are ignored by Git.
- The early 10k run and first 100k iteration used a joint-range midpoint as actuator neutral; that gave knees and other asymmetric-range joints biased commands. The interface was corrected to use zero-radian neutral, a later geometry audit corrected foot inertial locations and leg/arm anchors, and the uncompleted intermediate 100k run was explicitly marked interrupted. These checkpoints are not final evidence.
- The corrected 100,000-step CPU run is in `runs/phase1-humanoid-balance-100k-v3/` (100,096 PPO steps). Its cross-condition replay covered six contact profiles and three seeded starts per profile; **0 of 18 rollouts avoided a fall**, with failures after 166–186 control steps (1.66–1.86 s at 100 Hz). The MP4 and per-episode JSON report are retained as negative evidence under `docs/design/assets/` and `docs/design/results/`.
- The follow-up run completed 100,096 PPO steps in 7m41s on CPU, using nominal-to-full randomization over the first 60% of steps and a 50 Hz control rate. Its six-case nominal-total-mass comparison used identical seeded starts for PPO and neutral hold, with synthetic per-link mass redistribution. Both fell on every surface; mean survival was 2.14 s for PPO and 3.76 s for neutral hold. The policy regressed against this simple baseline. Checkpoint, manifest, exact config, metrics, and replay are archived in `docs/design/results/` and `docs/design/assets/`. One seed per surface is diagnostic, not a robustness estimate. Do not report “trained balance” without passing a predeclared multi-seed, full-episode test against a baseline.
- Surface profiles use friction, equivalent gravity tilt for slopes, and a compliance-to-MuJoCo-solref heuristic. The compliance mapping is a simulation knob, not a measured foam model. Eight camera definitions and four microphone plus one speaker site exist in MJCF, but camera extrinsics/lens calibration and all audio behavior remain unvalidated.

## External open-source references informing the structure

- [MuJoCo](https://github.com/google-deepmind/mujoco) is used for inspectable native MJCF, free base multibody physics, control, contact, and render. Preserve the XML source; compiled model binaries are version-dependent.
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) is a current open-source reference for robot-learning task organization and MJX/MuJoCo-Warp acceleration. Its task suites are a future candidate after the local dynamics and evaluations are credible.
- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) and [Isaac Sim](https://github.com/isaac-sim/IsaacSim) are better candidates when GPU-parallel training, sensor rendering, synthetic data, and ROS 2 integration become a priority. They carry substantially greater installation/runtime cost than the current CPU-first MuJoCo path.
- [Newton](https://github.com/newton-physics/newton) is an emerging open-source Warp/OpenUSD GPU-physics ecosystem. It merits a backend comparison when we have CUDA hardware and stable reference experiments; a claimed throughput win is not a reason to switch before contact/actuator fidelity comparisons.
- [Legged Gym](https://github.com/leggedrobotics/legged_gym) demonstrates a proven sim-to-real engineering pattern: actuator models, randomized friction/mass, observation noise, and pushes belong to the training environment. We currently implement only a subset and have no real robot data to calibrate those distributions.
- [ASAP (2025)](https://arxiv.org/abs/2502.01143) studies aligning simulation and physical humanoid dynamics for agile whole-body skills. It reinforces that broad randomization alone is not a digital twin: measure hardware response, identify parameters, close the sim/real loop, and validate skill transfer.
- [LeRobot](https://github.com/huggingface/lerobot) is a useful later reference for synchronized robot state/action/video datasets and recorded demonstrations. No teleoperation dataset or imitation learning is included in this initial phase.

## Near-term sequence

1. Establish stable, repeatable static/contact checks for the MJCF and correct the model before tuning reward weights.
2. Train stand balance with fixed-seed baselines, then progressively widen mass/friction/slope/compliance domains; maintain held-out seeds and report confidence, failure modes, foot slip and joint/torque saturation.
3. Add walking as a separate command-conditioned task with gait phase/foot clearance/contact objectives; build imitation/retargeted-motion baselines for sitting, supine/prone recovery, jump, kick and punch before asking unconstrained RL to discover them.
4. Add camera side-by-side recording/live preview and sensor timestamp/calibration schemas. Only then add perception policies; keep audio processing separate from locomotion initially.
5. Replace placeholder link properties with actual assembly BOM, CAD-derived link COM/inertia, measured mass, identified actuator dynamics, thermal/current limits, sensor extrinsics, and ground friction tests. Require simulation-to-simulation and safe hardware validation before calling the model a digital twin.
