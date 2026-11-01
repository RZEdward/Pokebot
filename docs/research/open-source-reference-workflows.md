# Open-source reference workflows

Reviewed 1 November 2026. These projects are architecture references, not dependencies to copy wholesale. Their robots, actuator modes, licenses, and compute assumptions differ from Hitmonlee.

## 1. MuJoCo Playground — closest training/model reference

- Source: [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground) and [MuJoCo Playground paper](https://arxiv.org/abs/2502.08844).
- License: Apache-2.0 for repository content, except cited third-party assets/textures (verify each asset before reuse).
- What to study: model/environment definitions, task configs separated from learners, reproducible training entrypoints, evaluation/replay, proprioceptive policies, domain randomization, and camera-based workflows. Its README documents bipedal locomotion, MJX/JAX and MuJoCo Warp paths, plus an `rscope` visualization that can display trajectories during training.
- What we adopt: native MuJoCo-compatible model, Gymnasium task boundary, explicit train/eval commands, deterministic seeded evaluation, documented policy observations/actions, telemetry and a separate visualization process.
- What we defer: JAX, Warp, GPU batching and high-dimensional vision. The current host is a Windows 11 laptop with Intel Arc graphics, a 14-core/18-thread Core Ultra 5 125H, and 16 GiB RAM; the installed PyTorch is CPU-only with no CUDA device. The custom biped/actuator model is the primary uncertainty.
- Study path: `locomotion/` environment → task config → `learning/train_*` → policy evaluation → training visualization. Compare their performance metrics and randomization parameter ranges with our initial single-environment CPU baseline.

## 2. Isaac Lab 3.0 — configuration and scale reference

- Source: [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab), [3.0 documentation](https://docs.nvidia.com/isaaclab/isaac-lab), and [framework paper](https://arxiv.org/abs/2511.04831). The 3.0 release is now the current line and includes a kitless Newton workflow; Isaac Sim is optional for that path.
- License: framework BSD-3-Clause; Mimic extension/scripts Apache-2.0; individual assets and Isaac Sim components have additional licenses.
- What to study: separation between assets, scene configuration, task/MDP, environment registration, algorithm runner, logs, checkpoints, and policy export. The project has many humanoid/legged models and supports multiple RL frameworks/backends.
- What we adopt: asset versioning, explicit environment/task configuration, clear training/evaluation stages, and backend-version compatibility as recorded decisions.
- Current host fit: the machine inventory shows an Intel Arc GPU and no CUDA device. The v3 Newton/Warp acceleration path is therefore for a later NVIDIA workstation or remote runner; avoid adding it to the active environment here. Full Isaac Sim's sensor/render stack is also unnecessary for the current balance example.
- What we defer: introducing its framework into the core repo. Keep the Python 3.12 Newton environment isolated if we benchmark it on a supported NVIDIA host.
- Study path: Isaac Lab reference architecture → one humanoid locomotion environment config → training runner → policy deployment examples. First reproduce their example on a stock robot to learn the tool; then port the custom model only if the bake-off supports the added stack.

## 3. ROS Controls MuJoCo integration — future hardware seam

- Source: [ros-controls/mujoco_ros2_control](https://github.com/ros-controls/mujoco_ros2_control) and [current ROS Control documentation](https://control.ros.org/lyrical/doc/mujoco_ros2_control/doc/index.html).
- License: inspect the repository's package-level license files before reusing individual code/assets; main project is maintained under ROS Controls.
- What to study: simulation implements a `ros2_control` hardware/system interface, exposing state and command interfaces so the same controller-manager layer can run against sim and hardware. The repo includes sensor plugins, demos, and model-conversion tools.
- Important limit from its own documentation: URDF does not cover all MJCF physics features, and conversion is explicitly experimental. Therefore keep MJCF-specific actuator/contact/sensor details visible and validate conversions rather than assuming URDF is a lossless source of truth.
- What we adopt: later, use the standard interface boundary for simulated vs. physical I/O. Keep RL training at the native simulator API for throughput; connect ROS for SIL and deployment.
- Study path: demo 2-DOF robot → hardware interface configuration → actuator and sensor mappings → conversion/modeling limitations → one measured real joint.

## 4. Humanoid-Gym and FastSAC — baselines, not architecture dependencies

- [Humanoid-Gym](https://github.com/roboterax/humanoid-gym) demonstrates a packaged locomotion task and sim-to-real workflow, but is built around the older Isaac Gym generation. Treat it as historical method/code-study material, not the chosen foundation.
- [FastSAC humanoid paper and code](https://arxiv.org/abs/2512.01996) report rapid sim-to-real training for particular Unitree G1 / Booster T1 setups. Study its environment/action/reward and deployment assumptions; do not apply its “minutes” headline to a custom robot with unknown actuators or CPU-only training.
- [VIRAL](https://arxiv.org/abs/2511.15200) is a useful later reference for visual humanoid transfer: privileged teacher, rendered student observations, online DAgger/behavior cloning, camera alignment, and appearance randomization. It is a later perception milestone, not what the balance policy should begin with.

## 5. mjlab — strongest current architecture study for this repository

- Source: [mujocolab/mjlab](https://github.com/mujocolab/mjlab), [architecture overview](https://github.com/mujocolab/mjlab/blob/main/docs/source/architecture_overview.rst), and [paper](https://arxiv.org/abs/2601.22074).
- Its current design separates a simulation layer (entities, scene composition, actuators and sensors) from an Isaac-Lab-style manager layer (observations, rewards, terminations, events, randomization and metrics). Existing MJCF is loaded into `MjSpec`, configured in Python, compiled to `MjModel`, then transferred to MuJoCo Warp for batched GPU stepping. RSL-RL handles training.
- What we adopt conceptually: MJCF remains the source for robot mechanics; task observations/rewards/terminations should become separate named components as the project grows beyond one task. Avoid adopting its GPU-specific runtime before the machine and a benchmark justify it.
- Benchmark only after the CPU baseline is understood: compare reset correctness, contact/actuator response, deterministic replay, sensor outputs and end-to-end training throughput. Treat its CUDA graph and thousand-world scale as possible benefits, not expected performance on this workstation.

## 6. Genesis World — broad multiphysics and renderer challenger

- Source: [Genesis World repository](https://github.com/Genesis-Embodied-AI/Genesis) and [official documentation](https://genesis-world.readthedocs.io/en/latest/). Its current repository describes a Python interface for MJCF/URDF/USD assets, parallel and heterogeneous environments, multiple physics solvers, sensor cameras, built-in GUI, and compilation across CUDA, ROCm, Metal, Vulkan, x86 and ARM64. The source repository is Apache-2.0; review bundled and optional backend licenses independently.
- Current host fit: the [latest installation documentation](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html) says Intel GPUs can drive viewing/offscreen rendering but have no Genesis physics backend, so physics runs on CPU. Older Genesis releases documented Intel GPU simulation; this has changed. On this machine Genesis would be a CPU physics challenger, not a way to accelerate simulation on the Arc GPU.
- Its breadth makes it interesting for future camera, tactile or differentiable modeling experiments, but a broad multi-physics stack is not automatically the best calibrated rigid-body/contact model for this humanoid. The project must reproduce known joint/contact traces and prove deterministic evaluation before being considered as a primary trainer.
- Keep it as a bounded challenger: import the same MJCF if supported, run the identical short balance/control benchmark, compare camera outputs and actuator semantics, and document unsupported features or conversion changes. Avoid changing model source just to fit a second engine.

## 7. Newton Physics — current solver/reference ecosystem

- Source: [Newton project organization](https://github.com/newton-physics), [main repository](https://github.com/newton-physics/newton), and [official project page](https://developer.nvidia.com/newton-physics).
- Newton is a Linux Foundation open project under Apache-2.0, initiated by Disney Research, Google DeepMind, and NVIDIA. Its current organization lists compatibility with MuJoCo Playground and Isaac Lab, MuJoCo-to-USD and URDF-to-USD converters, and a separate repository of permissively licensed example assets.
- Keep it distinct from the task framework. Newton can be a physics backend; Isaac Lab or mjlab can define the learning task, but solver choice still changes contacts, constraints, and supported features. Benchmark the same reset, step, sensor, and control traces before comparing policy returns.

## 8. Vosk — lightweight offline command-recognition reference

- Source: [alphacep/vosk-api](https://github.com/alphacep/vosk-api), [official install and audio-format guidance](https://alphacephei.com/vosk/install), and [official model catalog](https://alphacephei.com/vosk/models.html).
- The MVP's optional voice extra uses the Vosk Python API with the small US-English model and a constrained command grammar. Upstream lists the model at roughly 40 MB and Apache-2.0; the recognition library and model are separate downloads. A model citation/license record is written beside the downloaded files.
- The model catalog reports 9.85% WER on LibriSpeech test-clean and 10.38% on TED-LIUM for this model; these are not robot-command or this user's accent results. The current Windows-generated “stand” clip verifies format, offline inference, intent parsing and dispatch only. Evaluate actual speakers, background noise and false activations before relying on this recognizer.
- Vosk is one replaceable ASR adapter, not a claim that it is the best frontier recognizer. The command schema stays independent so later experiments can compare Vosk with Whisper-family local models or other streaming engines without changing skill dispatch.

## Source attribution rules for this repo

- Prefer upstream source and documentation links; pin the revision/commit in a decision record before copying an implementation pattern.
- Keep third-party models, meshes, textures, motion data and checkpoints separate from our original assets. Record the exact license and attribution beside every imported asset.
- Reimplement interfaces and task logic cleanly instead of vendoring large upstream files. If code is reused, retain copyright/license headers and add a `THIRD_PARTY_NOTICES.md` entry.
- Re-run upstream workflows on their supported robot before assuming their model/config is a valid Hitmonlee benchmark.
