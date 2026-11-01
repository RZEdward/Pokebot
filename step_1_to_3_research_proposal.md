# Project Hitmonlee: research proposal for Steps 1–3

**Research snapshot:** 1 November 2026  
**Purpose:** Choose and validate a simulation and robot-learning workflow, define a credible digital-twin model, and establish a staged plan for learning motion. This is a proposal, not a claim that any simulator is already selected.

## Executive recommendation

Treat simulation as a small toolchain with different jobs rather than choosing one all-purpose simulator in advance:

1. **Development target is now inventoried.** This is a Windows 11 laptop with Intel Core Ultra 5 125H (14 cores / 18 logical processors), Intel Arc graphics, 16 GiB RAM, and CPU-only PyTorch; CUDA is unavailable. Native MuJoCo CPU is the practical baseline. A remote NVIDIA workstation becomes relevant only if measured throughput warrants it.
2. **Model/control baseline:** use native MuJoCo on this host for model iteration, inspectable contact dynamics, and controller experiments. Compare Newton/Isaac Lab kitless, MuJoCo Warp and MJX/JAX on a supported NVIDIA host if throughput becomes a demonstrated bottleneck. Genesis can be a bounded CPU challenger; current docs say Intel graphics support rendering, not Genesis physics acceleration.
3. **Learning throughput:** benchmark candidates against the same robot, command contract, task, parameter cases, seeds, and stopping criteria. Retain whichever gives the best combination of model compatibility, reproducibility, useful throughput, and developer friction on the actual workstation. Do not assume backend parity simply because a framework exposes multiple engines.
4. **Perception and scene validation:** use Isaac Sim or another renderer only when realistic camera/depth/lighting and scene workflows are needed. Photorealism is not required to validate early joint mechanics or proprioceptive balance.
5. **Model matching:** establish a CAD-informed, measured digital twin with parameter provenance and uncertainty. Identify actuator/joint behavior on a bench before treating a learned whole-body policy as transferable.
6. **Control development:** first demonstrate model-based balance and safe low-level control; then use imitation/reference tracking and reinforcement learning for skills where they add value. Train with measured parameter distributions, deliberate perturbations, and more than one simulator as a robustness/evaluation method—not just broad, arbitrary randomization.

This is a deliberately conditional recommendation. Isaac Lab 3.0's current installation guide describes Newton as its default backend and it can run without Isaac Sim; Isaac Sim 6.2 is the current optional full scene/rendering stack. Native MuJoCo is broadly available, and its GPU paths have different development/runtime dependencies. The right choice depends on model behavior and measured workload on the actual computer, not on a 2024-era “MuJoCo versus Isaac” framing ([Isaac Lab installation](https://docs.nvidia.com/isaaclab/isaac-lab/installation), [MuJoCo binaries](https://github.com/google-deepmind/mujoco), [Isaac Sim 6.2 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)).

## What the current project files establish

The folder currently contains two documents and no source-code packages, robot assets, CAD, environment lockfiles, or tests:

- `initial_project_plan.txt` describes a long-range lifesize metallic Hitmonlee goal. It asks for research and a verbal proposal for Steps 1–3, but explicitly treats the initial simulator and workflow as open questions.
- `project_hitmonlee_build_plan_final.html` is a substantial engineering concept. It sketches a 1.2–1.4 m, approximately 25 kg humanoid, around 23–26 actuated joints, machined aluminium structure, ROS 2 and simulation-first work. It currently favors MuJoCo/MJX with Isaac Sim as a secondary tool, and recognizes actuator selection, locomotion, and calibration as major risks.

These numbers are planning hypotheses, not frozen requirements. The robot's mass, proportions, joint count, actuator technology, and target skills should be revisited as the actuator and safety envelopes become measurable. There is no existing code architecture to preserve; the repo layout below is a proposed starting structure.

## Research findings: what “frontier” means in practice

### 1. A digital twin is an executable, calibrated model

A mesh that looks like the robot is useful for visualization, but cannot answer whether a joint can accelerate a leg, whether the robot will fall, or whether a controller will survive current limits. Before any hardware exists, this is more accurately a **provisional robot simulation model**. It becomes a useful digital twin only as it is tied to the as-built robot through geometry revisions, measured actuation/sensing, and calibrated dynamics. The useful model joins geometry, inertial properties, joint limits, contact surfaces, actuation, sensing, controller timing, and measured calibration data. Its uncertainty matters as much as its nominal parameters.

Modern transfer work therefore combines several techniques:

- **Bottom-up identification:** measure motor/gearbox/drive behavior and joint friction, delays, saturation, and thermal/current constraints before whole-robot learning. Estimate inertial and contact parameters from CAD and physical measurements, then refine them from safe experiments.
- **Targeted system identification:** fit simulator parameters to observed trajectories. Recent work explores active excitation that chooses informative commands instead of relying on arbitrary data, and parameter fitting without assuming differentiable contact or direct torque sensing ([SPI-Active, 2025](https://arxiv.org/abs/2505.14266)).
- **Residual actuator/dynamics models:** retain a physics-grounded simulator, then model repeatable discrepancies such as actuator response or latency instead of asking a policy to absorb every modeling error. Current work also directly studies simulator adaptation against proprioceptive distributions ([Simulator Adaptation, 2026](https://arxiv.org/abs/2604.11090)).
- **Domain randomization with evidence:** randomize uncertain mass, friction, delay, motor strength, sensor noise, and pushes within measured or defensible ranges. Too-wide randomization can produce a weak policy; targeted identification and robustness testing should complement it.
- **Sim-to-sim and sim-to-real checks:** compare the same trajectories or policies across engine backends and, later, on hardware. Multi-simulator training is a promising research direction ([PolySim, 2025](https://arxiv.org/abs/2510.01708)), but it adds integration complexity and should be a later experiment, not a prerequisite.

These techniques move the project from “make the model plausible” to “measure discrepancies, record evidence, and update parameters deliberately.” The actuator bench and first single-joint build have high value because they identify the largest transfer uncertainty before expensive hardware is repeated across the robot.

### 2. The simulator landscape has changed

| Option | Best fit | Strengths | Costs and uncertainties | Proposed role |
|---|---|---|---|---|
| **MuJoCo native** | Model iteration, contact/control experiments, repeatable single-world evaluation | Compact MJCF, fast startup, mature articulated rigid-body/contact workflow, easy local inspection | Not the best route to huge GPU batches or photorealistic sensor scenes by itself; simulation quality still depends on model calibration | First working model and reference behavior |
| **MuJoCo Warp (MJWarp)** | Many MuJoCo worlds and batched camera rendering on NVIDIA GPUs | GPU-optimized MuJoCo backend and batched renderer; preserves a MuJoCo-centered model ([docs](https://mujoco.readthedocs.io/en/3.6.0/mjwarp/), [project](https://github.com/google-deepmind/mujoco_warp)) | NVIDIA GPU path; backend parity and supported features must be tested for the chosen model | Candidate for a later NVIDIA workstation or remote runner |
| **MJX (JAX)** | JAX-native batched physics, accelerator experiments, differentiable/JAX research | Parallelized identical scenes and integration with JAX learning code ([docs](https://mujoco.readthedocs.io/en/stable/mjx.html)); MuJoCo Playground provides robot-learning and sim-to-real examples ([paper](https://arxiv.org/abs/2502.08844), [code](https://github.com/google-deepmind/mujoco_playground)) | JAX introduces a separate programming/runtime path; backend feature and performance profile differs from native MuJoCo | Include in the bake-off if JAX is a plausible training choice |
| **Isaac Lab 3.x** | Scalable RL/imitation workflows | Open-source learning framework with multi-backend architecture; current docs describe Isaac Sim and standalone Newton support, with shared asset/sensor/scene APIs ([docs](https://docs.nvidia.com/isaaclab/isaac-lab), [framework paper](https://arxiv.org/abs/2511.04831)) | More system weight and a faster-moving stack; backend and version compatibility matter | Strong candidate for training; use Isaac Sim separately when sensor-rich SIL is needed; benchmark Newton and PhysX separately |
| **Newton** | GPU physics, extensible solvers, differentiable/system-identification experiments | Open-source Linux Foundation project built on Warp/OpenUSD with MuJoCo Warp integration and multiple solvers; supports URDF, MJCF, and USD ([project](https://github.com/newton-physics/newton), [current documentation](https://newton-physics.github.io/newton/)) | Fast-moving solver ecosystem; robot/contact/controller fidelity and reproducibility need direct evaluation | Evaluate through Isaac Lab or its own supported workflows; do not assume all advertised solver features suit this robot |
| **Isaac Sim** | Rendering, sensor simulation, USD scenes, synthetic data, ROS 2 SIL/HIL | Imports CAD/URDF and works with USD scenes; high-quality cameras and scene workflows ([overview](https://developer.nvidia.com/isaac/sim)) | Heavy workstation and version footprint; high visual fidelity does not guarantee correct contact or actuator dynamics | Add when camera/depth simulation or virtual commissioning is an actual need |
| **Genesis** | Emerging differentiable/multi-physics experimentation | Open-source physics platform with differentiable scenes and multiple solver/material goals ([docs](https://genesis-world.readthedocs.io/en/latest/), [project](https://github.com/Genesis-Embodied-AI/Genesis)) | Current install docs support Intel GPU viewing/rendering but not Intel GPU physics; physics would use CPU here. Compatibility and deterministic transfer evidence need validation. | Optional CPU challenger; do not expect Arc simulation acceleration |

Two established tools deserve explicit consideration, but for different jobs. **Gazebo Sim** has practical ROS 2 integration through `gz_ros2_control`, including custom simulated hardware components; it is a credible software-in-the-loop option for exercising the eventual ROS graph. It is not automatically the best massively parallel RL engine ([integration](https://control.ros.org/master/doc/gz_ros2_control/doc/index.html)). **Drake** emphasizes exposing multibody equations, gradients, contact, optimization and control structure; it is attractive for trajectory optimization, estimation and mechanics analysis, but it is not the default mass-parallel policy-training stack ([Drake overview](https://drake.mit.edu/)). Neither needs to be adopted as the main simulator to be useful.

The simulator shortlist should be role-based:

- **Training engine candidates:** Newton/Isaac Lab kitless, MuJoCo native, MJWarp, MJX/JAX. Test no more candidates than can be maintained.
- **Control and mechanics analysis:** MuJoCo native; Drake where its optimization/dynamics tools answer a specific question.
- **ROS integration/SIL:** `ros2_control` plus the MuJoCo plugin or Gazebo Sim; Isaac Sim also has a ROS integration but carries more platform and rendering dependencies.
- **Camera and synthetic-data work:** Isaac Sim only when needed; compare its image/latency pipeline against real sensor specifications and collected calibration data.
- **Exploratory only:** Genesis or other new backends, bounded by a concrete hypothesis and reproducible benchmark.

This separation avoids a false requirement that one engine must simultaneously be the fastest trainer, the easiest control debugger, the closest camera simulator, and the deployment runtime.

### Proposed bake-off scorecard

Score candidates 1–5, but publish raw measurements alongside scores; a weighted score must not conceal fatal mismatches. Suggested initial weights are judgment calls to be reviewed before the test:

| Criterion | Weight | Measurement |
|---|---:|---|
| Dynamics/model coverage | 25% | Correct joint transmissions, actuator saturations, contacts, sensing, reset semantics; list unsupported features |
| Transfer-relevant agreement | 20% | Error against reference single-joint and leg traces, using the same parameters and controller |
| Iteration/debug cost | 15% | Time to edit asset, inspect forces/state, reproduce and explain a failure |
| Learning throughput | 15% | Valid environment-steps per second at fixed scenario and fixed numerical timestep; include warm-up and rendering cost |
| Reproducibility | 10% | Variation across repeated seeded rollouts, physics and policy metrics, checkpoint replay |
| Integration path | 10% | Import/export effort, controller API, model provenance, eventual ROS/hardware adapter |
| Lifecycle/platform fit | 5% | Supported OS/GPU, pinned install, licenses, current docs/releases, expected upgrade work |

The weights reflect this project's current maturity: no actuators have been characterized, so a small improvement in samples/second is less valuable than knowing the physics is interpretable and the package can be installed. Once a calibrated model exists and training wall time dominates, throughput can receive a higher weight.

NVIDIA currently describes Isaac Lab as complementary to MuJoCo and supports several physics backends; Newton is no longer merely a concept. That changes the sensible decision from selecting one vendor stack today to measuring a small common workload across contenders ([Isaac Lab](https://developer.nvidia.com/isaac/lab), [Newton](https://developer.nvidia.com/newton-physics)).

### 3. Training methods are a portfolio, not “RL first”

Recent methods suggest several complementary tracks rather than one winning recipe:

| Method family | How it is used in current humanoid work | Fit for Hitmonlee / caveat |
|---|---|---|
| Reward-driven RL (often PPO; increasingly off-policy SAC/TD3 variants) | Learn tracking, balance and robustness from parallelized experience. FastSAC/FastTD3 reports a 15-minute training recipe for specific G1/T1 tasks and hardware; it does not imply a new custom robot will train in 15 minutes ([paper/code](https://arxiv.org/abs/2512.01996)). | Good once the task, action scale, actuator model and reset logic are sound. RL speed depends on parallelism and environment implementation. |
| Motion imitation / adversarial motion priors | Track retargeted motion capture or learned motion distributions; humanoid studies use these to create whole-body behavior beyond a hand-shaped walking reward ([Whole-body humanoid locomotion with human reference](https://arxiv.org/abs/2402.18294)). | Useful for a character-inspired expressive robot. Retargeting and torque feasibility must account for Hitmonlee's morphology and joint limits. |
| Teacher–student / privileged learning | A teacher sees simulator state or privileged contact/terrain variables; a deployable student gets only measurements. For visual tasks, distillation plus DAgger/behavior cloning transfers from privileged teacher to image policy ([VIRAL](https://arxiv.org/abs/2511.15200)). | Later, when visual loco-manipulation is a concrete target. It is wasteful for initial actuator/balance milestones. |
| System identification and dynamics adaptation | Fit physical parameters from real motion, or use active probes / residual actuator models, then update simulation or train an adaptive policy ([SPI-Active](https://arxiv.org/abs/2505.14266); [proprioceptive simulator adaptation](https://arxiv.org/abs/2604.11090)). | High priority. The design is new and the actuator transfer curve is likely unlike the commercial robots in published demos. |
| Structured robustness randomization | Perturb identified uncertain parameters; add torque-space/state-dependent perturbation or multiple simulators to cover unmodeled mismatch ([torque perturbation](https://arxiv.org/abs/2603.21853); [PolySim](https://arxiv.org/abs/2510.01708)). | Use only with documented ranges and holdouts. Untargeted, very broad randomization can hide bad nominal models and produce conservative or unstable behavior. |
| Model-based optimization / MPC / whole-body control | Use explicit dynamics and constraints to compute stance, balance, or feasible trajectories. Drake and other robotics toolkits are strong here. | Necessary baseline and diagnostic; does not exclude RL. Gives reference behavior and a route to test model errors before learning. |

Reinforcement learning has produced transferable legged and humanoid locomotion, including hardware demonstrations ([Humanoid-Gym](https://arxiv.org/abs/2404.05695), [MuJoCo Playground](https://arxiv.org/abs/2502.08844)). These results often depend on carefully chosen actuator abstractions, high-throughput GPU simulation, low-level motor control, strong safety engineering, and known robot models. They establish that the approach can work; they do not establish that a custom Hitmonlee model will transfer zero-shot. It is not a substitute for accurate low-level actuation, safety logic, or a well-defined task. For this project:

1. Implement joint-level position/velocity/torque interfaces and limits; verify conventions and signs.
2. Build a passive standing model and test static balance, contact, and state estimation assumptions.
3. Use model-based inverse kinematics, centroidal / whole-body control, or trajectory optimization for transparent baseline behaviors and safe reference motions.
4. Use motion capture or retargeted references for motion tracking when natural gesturing or human-like movement is desired. First verify the retargeted motion against kinematics and balance; then train a tracking controller from easy stable motions toward more dynamic ones, rather than optimizing “walk” from scratch immediately.
5. Apply RL for robust balance recovery, terrain adaptation, locomotion, and later expressive skills where it beats the baseline. Start with proprioception and a compact command space; add vision only after the motor policy is measurable.
6. Evaluate on held-out dynamics, perturbations, surfaces, and (if available) a second simulator backend. A second backend is a falsification test, not an automatic guarantee of sim-to-real. Record success, falls, tracking error, energy/current proxies, constraint violations, and compute cost.

### What not to infer from headline results

- A policy that walks on a Unitree G1, Go1 or Booster T1 validates a method on that robot, actuator and firmware. It is not evidence of generalization to a custom mass distribution, long-legged silhouette or self-designed joint modules.
- “Zero-shot transfer” generally still involves substantial real-to-sim work: robot CAD/URDF, actuator system ID, action-frequency matching, calibration, parameter selection, and a carefully constrained deployment test.
- More parallel environments only help if reset/physics/observation code keeps pace and rollout diversity remains useful. Compare samples per second and wall-clock time to a validated policy, not the engine's isolated physics benchmark.
- A photorealistic image is not proof of physical accuracy. Contact solver, actuator, timestep, sensing latency and control update semantics can dominate locomotion transfer.

### 4. Keep model formats interoperable, but do not pretend they are identical

The project will eventually need CAD, a ROS-facing robot description, and simulator-native models. A practical approach is to maintain a **canonical parameter and naming specification** (link/joint names, frames, dimensions, mass properties, limits, actuator data, sensor frames), then generate or validate formats from it:

- CAD is authoritative for manufactured geometry, dimensions, and revisioned part identity.
- URDF/Xacro is useful for ROS kinematics, frames, visualization, and ROS tooling.
- MJCF is likely the most direct source for detailed MuJoCo contacts, actuator models, sensors, and keyframes.
- USD is useful for Isaac scene composition and rendering/synthetic-data workflows.

No format should be presumed to round-trip perfectly. Automated consistency checks should compare joint names/axes/limits, transforms, inertias, and actuator interfaces between generated assets. The same control contract should run against simulated and physical hardware via an adapter. ROS 2 `ros2_control` is designed around controller/hardware separation, and the current MuJoCo integration exposes simulated actuators and sensors through a hardware interface ([ROS 2 Control](https://control.ros.org/jazzy/doc/ros2_control/doc/index.html), [MuJoCo integration docs](https://control.ros.org/jazzy/doc/mujoco_ros2_control/mujoco_ros2_control/docs/hardware_interface.html)).

## Proposed repo architecture

The repo should grow in small stages. Do not create every package before the first model exists. This target structure keeps authoritative data, simulator adapters, training experiments, and measured evidence distinct:

```text
Pokebot/
├── README.md                         # project status, setup entry point, current gate
├── docs/
│   ├── project/                       # intent, requirements, decision log, risk register
│   ├── research/                      # this proposal and dated research notes
│   ├── architecture/                  # interfaces, timing, model-format rules
│   └── verification/                  # calibration procedures and acceptance gates
├── robot_spec/                        # canonical names, frames, limits, units, versioned params
├── description/
│   ├── urdf/                          # ROS kinematic/visual description, generated as applicable
│   ├── mjcf/                          # MuJoCo dynamics, actuators, contacts, sensors
│   ├── usd/                           # optional Isaac scene/robot assets
│   └── meshes/                        # collision and visual meshes; source provenance recorded
├── cad/                               # source CAD or linked CAD release package, drawings, BOM
├── ros2_ws/src/
│   ├── hitmonlee_description/         # installable robot assets and launch descriptions
│   ├── hitmonlee_control/             # controller configs and shared command/state contracts
│   ├── hitmonlee_hardware/            # eventual real actuator/sensor adapters; initially stubbed
│   └── hitmonlee_bringup/              # simulation/hardware launch compositions
├── sim/
│   ├── mujoco/                        # model validation, headless runs, visual inspection
│   ├── isaac_lab/                     # optional benchmarked training tasks/backend config
│   └── benchmarks/                    # common scenarios, metrics, seeds, backend comparison
├── learning/
│   ├── tasks/                         # task specs and reward/observation definitions
│   ├── configs/                       # versioned training config, randomization ranges
│   ├── train/                         # RL and imitation entrypoints
│   ├── evaluate/                      # held-out and cross-simulator evaluation
│   └── policies/                      # metadata and export artifacts, not large checkpoints in git
├── calibration/
│   ├── protocols/                     # actuator/joint excitation and measurement procedures
│   ├── data/                          # raw data outside git or via LFS/DVC with provenance
│   └── models/                        # fitted parameter sets with uncertainty and date
├── tools/                             # model conversion, validation, plotting, report utilities
├── .github/workflows/                 # lint, model/schema checks, small smoke workflows
├── pyproject.toml / uv.lock           # Python tools and reproducible environment
└── .gitignore
```

### Architecture rules

- **One naming and unit convention:** SI units, explicit coordinate-frame convention, stable joint/link names. Document the convention before exporting assets.
- **Separate source from generated artifacts:** keep master CAD and canonical specification identifiable. Generated URDF/MJCF/USD assets carry a source revision and can be rebuilt.
- **Same control contract:** controllers consume state and emit commands through defined interfaces; simulation and hardware implement that contract separately. Avoid simulator-specific command logic leaking into the policy.
- **Experiment provenance:** every trained policy records the robot-model revision, simulator/backend/version, task/config, parameter distribution, random seeds, code revision, and evaluation results.
- **Version pairs, not floating latest:** record tested Isaac Lab / Isaac Sim / Newton / Warp versions as a compatible set. In particular, the Isaac Sim compatibility matrix says a matching release number alone does not guarantee backend/policy/ROS compatibility; keep bundled Newton dependencies together rather than independently upgrading them ([compatibility matrix](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/compatibility.html)).
- **Data discipline:** raw calibration measurements are immutable; fitted parameter sets include method, units, fit residuals, uncertainty, and the source dataset ID.
- **Safety by interface:** command saturation, joint limits, watchdog behavior, safe stop, and fall/overcurrent handling are outside the learned policy and validated independently.
- **Small automated checks:** verify descriptions compile, frames and names agree, inertias are valid, limits are consistent, and a short seeded simulation can initialize. These checks protect model changes without pretending to validate full-robot dynamics.

### Grow the repository only when the project earns each layer

The directory tree above is a destination, not day-one scaffolding. Start with a lean structure:

```text
Pokebot/
├── README.md
├── docs/
│   ├── research/step_1_to_3_research_proposal.md
│   ├── decisions/                       # dated ADRs, one per consequential choice
│   └── verification/                     # benchmark protocol and results
├── model/                               # first robot spec + one simulator asset
├── sim_bench/                           # reproducible joint/leg benchmark only
├── scripts/                             # validate model, run, and summarize traces
└── pyproject.toml + lockfile             # once a runnable Python workflow exists
```

Add `description/urdf`, ROS packages, `learning/`, and `calibration/` only when the corresponding deliverable exists. Avoid empty ROS workspaces or a framework abstraction before there are two real backends to compare. Keep CAD in a CAD-native source location/repo if file size and collaboration make Git awkward; commit neutral manufacturing exports and revision metadata when released.

## Step 1 — choose the toolchain by evidence

### Decision to make

Choose a **minimum viable stack** for design and model development, plus a **training/rendering path** only when needed. The intended result is a one-page ADR (architecture decision record) with selected versions, hardware needs, known limitations, and conditions for revisiting the choice.

### Proposed bake-off (2–3 weeks after environment setup)

Build one deliberately small model: a 1-DOF actuator/joint load with measured or provisional torque-speed, delay, saturation, and friction; then a simple leg/standing model if time permits. Run the same action sequence and a minimal balance/tracking workload in native MuJoCo and Newton through Isaac Lab 3.0 kitless. This is the default two-way comparison because both have plausible Windows paths and let us learn whether acceleration helps before taking on full Isaac Sim. Add MJWarp if the NVIDIA GPU is supported and the same model can be run; add MJX if JAX is strategically useful; add full Isaac Sim after its machine requirements are checked and a rendering/SIL task exists. Genesis is an optional challenger if it can reproduce the same benchmark and logs.

The Isaac Lab 3.0 installation guide says its standalone Newton path uses Python 3.12, supports Windows 11 x86-64, and requires a CUDA-capable NVIDIA GPU for GPU training; Isaac Sim is optional. Isaac Sim 6.2's workstation requirements are much higher than Newton-only: the current guide lists 32 GB RAM and a supported RTX GPU with 16 GB VRAM as minimum, with sensor-heavy workloads needing more. This distinction should shape setup order: test a compact Newton-only environment before considering full Isaac Sim ([Isaac Lab installation](https://docs.nvidia.com/isaaclab/isaac-lab/installation), [Isaac Sim 6.2 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)).

Score candidates on:

- Model expressiveness for the intended actuator/contact behavior and sensor noise.
- Difference from native MuJoCo reference trajectories: joint position/velocity, contact events, energy, and constraint behavior.
- Real-time factor or training samples/second on the actual workstation, including rendering costs if vision is tested.
- Reproducibility across repeated seeded runs and failure/debug visibility.
- Integration cost: install reliability, model import/export, ROS 2 connection, and policy export/deployment path.
- Maintenance and licensing risk, release cadence, documentation, and whether the workflow works without an expensive GPU.

Use two benchmark tiers so a large engine does not win by workload mismatch:

**Tier A — simulator correctness and iteration** (run on CPU where possible): compile the model, reset deterministically, apply the same joint command sequence, compare joint state and contact sequence, inspect force/torque values, replay a failure, and save traces. Fix the physics timestep and document solver settings.

**Tier B — learning throughput** (GPU if supported): hold robot/task, episode horizon, observation/action cadence, learning algorithm/version, reward, reset distribution, and total environment-steps constant. Report startup separately from steady-state steps per second, wall-clock time to a fixed evaluation criterion, hardware details, and success/fall/limit metrics over held-out seeds. Measure headless and camera modes separately. Compare each trained policy in a reference simulator; do not compare reward values across engines without equivalent task and physics definitions.

Pre-register the command traces, parameter set, timestep, action cadence, scenario seeds, logging resolution, and pass criteria before measuring. If engines require different model approximations, document the mismatch and compare them on the function they serve rather than claiming one global winner.

The initial stack need not maximize training FPS. The project has no validated actuator or full-body asset yet, so model transparency and iteration speed have higher immediate value. Revisit after the actual model and target workstation are known.

### Step 1 deliverables and exit gate

- Toolchain ADR plus pinned environment and a reproducible “hello robot” run.
- Model-format/naming/unit convention and a tiny asset conversion/check tool.
- Benchmark results with criteria and measured machine details.
- **Exit:** a new contributor can build the environment and reproduce the short benchmark from the repo; a simulator is selected for the next phase with explicit trade-offs.

### Platform and middleware implications

The existing build plan specifies Ubuntu 24.04 + ROS 2 Jazzy as its final stack, but that decision needs a dated update. ROS 2 Lyrical Luth was released in May 2026 as an LTS supported through May 2031; Ubuntu 26.04 and Windows 11 are Tier 1 platforms in ROS release materials ([release announcement](https://discourse.openrobotics.org/t/ros-2-lyrical-luth-released/55021), [release index](https://github.com/ros2/ros2/releases)). However, Isaac Sim 6.2's integration guide says Jazzy and Humble are tested; other distros on supported Ubuntu versions are experimental, and its native Windows Pixi walkthrough is for Jazzy ([ROS 2 install/integration matrix](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_ros.html)). Therefore: keep training independent of ROS; revisit the ROS distribution when the onboard compute and motor drivers are shortlisted. Prefer Lyrical if the physical Linux target and dependencies support it; use Jazzy for the documented Windows Isaac Sim path if that becomes necessary. Do not pin Jazzy into the entire project just to make a desktop simulation bridge work. The robot will likely use Linux onboard even if CAD, documentation, or some simulation happen on Windows.

Do not force the high-throughput learner to run through ROS 2 for every environment step. Train through the simulator's native/vectorized API; then build a software-in-the-loop path that exercises the same policy observation/action contract and hardware adapter. `ros2_control`'s value is the controller-to-hardware boundary and lifecycle; a custom real-time motor bus still needs its own validated implementation. MuJoCo and Gazebo integrations currently exist in ros2_control; Isaac Sim 6 has Windows ROS workflows with distribution constraints, so do not assume one latest ROS version works identically in every simulator ([simulator integrations](https://control.ros.org/master/doc/simulators/simulators.html), [Isaac Sim ROS migration note](https://docs.isaacsim.omniverse.nvidia.com/latest/migration_guides/isaac_sim_6_0/ros2_workspace_package_migration.html)).

## Step 2 — create the digital twin progressively

### Model ladder

1. **Kinematic skeleton:** proportions, frames, range of motion, joint axes, self-collision, cable/service volumes. Geometry can be coarse; correct frames matter first.
2. **Inertial skeleton:** link mass, center of mass, inertia tensor and total mass budget from CAD/material estimates. Cross-check units, positive-definite inertias, and plausible whole-body COM.
3. **Actuator/joint model:** torque/current and speed limits, gearbox reduction/backlash, friction, compliance where relevant, servo loop behavior, update delay, communication latency, saturation, thermal derating. Use measured module data, not an ideal torque source.
4. **Contact and environment:** foot collision geometry, friction range, restitution only if justified, floor/terrain variation, self-contact and fall behavior. Simplify collision meshes separately from visual meshes.
5. **Sensor/control timing:** IMU, joint encoder, foot/contact or force sensing, camera frames when needed; include bias/noise, dropout, delay, rates, and clock effects appropriate to the control problem.
6. **Versioned calibration:** measure one actuator, one assembled joint, then one leg/lower-body rig. Fit parameters and store uncertainty distributions and residual plots. Update the twin only with traceable evidence.

**The policy action space must match the eventual drive mode.** If the chosen module accepts position commands and closes its own current/velocity loops internally, a torque-commanded simulator policy can look successful while being unusable on hardware. Record whether the drive accepts position, velocity, current/torque, or impedance commands; the policy update rate; the drive's internal loop rates/gains; and any filtering/limits. Train and evaluate the same command semantics end-to-end. A higher-level policy may output joint targets while a low-level controller handles torque, but the simulated low-level controller must represent the hardware loop with measured delay and saturation. Freeze this contract only after the actuator bake-off.

### Identification plan: measure what the controller will actually command

| Quantity | Bench/data method | Twin use | Common trap |
|---|---|---|---|
| Command-to-motion delay and update jitter | Timestamp command issue, drive acknowledgement, encoder samples; use a safe step and periodic command at several operating rates | Transport and controller delay, action-hold semantics | Measuring only average network latency; tail jitter and stale frames destabilize a biped |
| Closed-loop position/velocity response | Small steps, chirps and sine sweeps under unloaded and known inertial load; log command, encoder, velocity estimate, bus voltage and current/temperature | Embedded drive loop, gains/bandwidth, saturation, rate limits | A datasheet peak-torque number does not define the high-level policy's torque-to-motion response |
| Torque/current-speed envelope | Characterize safely with an appropriate fixture/load cell or brake and thermal monitoring, within vendor limits; include supply voltage and duty cycle | Torque-speed limit, current clipping, voltage and thermal derating | Using stall torque as continuous torque; ignoring battery sag and gearbox heating |
| Friction, backlash, compliance and deadband | Slow bidirectional ramps, reversals, repeat at different loads; inspect hysteresis and repeatability | Coulomb/viscous friction, backlash zone, stiffness/damping or residual model | Folding all hysteresis into a single viscous coefficient |
| Reflected inertia and load response | Repeat dynamic sweeps with known loads; compare no-load and loaded response, use known rotor/gear data where credible | Rotor/gear inertia and motor-load coupling | Treating a geared servo as an ideal force source |
| Link mass/COM/inertia | CAD/material estimate, weighing, balance/pendulum fixture for selected links, then whole-subassembly mass/COM check | Rigid-body dynamics | Accepting CAD density defaults or invalid inertia tensors after mesh conversion |
| IMU bias/noise and frame alignment | Stationary logs at multiple orientations and temperature; compare gravity direction against mechanical frame | Bias, noise, orientation, rates and estimator model | Assuming a noise-free perfect orientation in training and deploying a filtered delayed estimate |
| Foot-ground interaction | Instrumented push/slide tests on candidate sole/floor pairs; measure repeatability and load distribution | Friction/contact parameter distributions and touchdown behavior | Treating a simulator's numeric friction coefficient as portable across solvers/material pairs |

For black-box motor drives, identify a usable input-output model even if the electrical constants are unavailable. Use isolated, low-energy, mechanically restrained experiments designed around the exact controller mode intended for the robot. Store raw logs; do not extrapolate beyond tested voltage, load, temperature, or duty cycle. Record uncertainty intervals instead of hiding them behind a single best-fit number.

This follows the broader system-identification lesson: physics-based and learned models can be combined, but parameters are only identifiable when the excitation and measurements make them observable. See [Robot Model Identification and Learning: A Modern Perspective](https://doi.org/10.1146/annurev-control-061523-102310); a recent transfer-gap review notes that parameter fitting cannot fix a fundamentally wrong model class, such as omitting compliance where it matters ([The Reality Gap in Robotics](https://doi.org/10.1146/annurev-control-031924-100130)).

Model fidelity should follow the next question. Early walking control needs inertias, actuator response, feet, contacts, and delays more than detailed face meshes or photorealistic lighting. Vision tasks later need calibrated intrinsics/extrinsics, lighting, materials, and scene variability.

### Twin validation ladder

- **Static:** mass totals, COM, joint limits, gravity direction, no unexplained penetrations or unstable inertias.
- **Kinematic:** commanded pose and frame transforms match CAD/URDF/MJCF expectations.
- **Single joint:** step/chirp/sine sweeps compare measured and simulated position, velocity, current/torque proxy, rise time, overshoot, deadband, and saturation.
- **Unloaded leg and lower body:** compare quasi-static poses, inertial response, foot contact timing, and safe fall states.
- **Closed-loop behaviors:** repeat the exact safe reference/controller sequence in sim and hardware; compare time-aligned traces and quantify errors.

### Step 2 exit gate

The digital twin is ready for learning only when a calibrated single joint and a representative leg/lower-body setup reproduce measured motions within pre-agreed tolerances, and parameter uncertainty is understood. This is a gate to train useful candidate policies, not proof the robot is safe to walk untethered.

## Step 3 — train and validate motion in increasing risk

### Learning sequence

1. **Simulation-only fundamentals:** actuator response, joint trajectories, posture, fall detection, stand-up feasibility, and balance in a simplified model.
2. **Stable reference motions:** offline trajectory optimization, inverse kinematics/whole-body control, or motion retargeting to generate physically feasible poses and transitions.
3. **Tracking and balance policy:** imitate/track references; train a compact controller with command velocity/heading and proprioceptive observations. Use privileged state only for training if the deployed estimator cannot provide it; distill to deployable observations.
4. **Robustness:** curriculum from flat floor to small disturbances and bounded terrain changes; randomize within measured/calibrated ranges, then add stress tests outside those ranges. Include delays, payload/COM shifts, friction, actuator weakness, pushes, observation noise, and episode resets.
5. **Generalization and evaluation:** held-out seeds, parameter corners, separate simulation backend, and hardware-in-loop when available. Log falls, command tracking, joint-limit violations, peak effort/current proxy, work/energy, and recovery behavior—not reward alone.
6. **Transfer gate:** export a policy with model/config metadata; first use it in a physically constrained setup (e.g. test stand/harness, low speed, emergency stop, supervised operator). Require stable bounded tests before increasing freedom or speed.

### Training approaches to compare

- **Model-based controller / trajectory optimization:** interpretable baseline, useful for validating model and providing references; less general under unmodeled disturbances.
- **Imitation or motion tracking:** efficient way to teach a target gait or expressive movement from references; depends on reference quality and retargeting.
- **PPO or SAC-style RL:** robust behavior through repeated interaction and randomization; costly to tune and sensitive to dynamics/reward/observation choices.
- **Residual/adaptive approaches:** learn correction or infer dynamics online around a known safe controller; may improve robustness but adds state-estimation and validation complexity.

Do not begin with camera-based end-to-end locomotion or a language-to-action policy. Those solve different perception/behavior problems and make it harder to isolate the mechanics and low-level control risks.

### Step 3 exit gate

First milestone: reproducible simulated standing and controlled stepping across held-out disturbances with no limit violations in the evaluation envelope, followed by a documented hardware-transfer protocol. Actual hardware transfer starts only after the actuator, structure, sensing, power, emergency stop, and test fixture are independently ready.

## Immediate work plan

| Order | Work | Evidence produced |
|---|---|---|
| 1 | Hardware inventory completed: Windows 11, Core Ultra 5 125H, Intel Arc, 16 GiB RAM, no CUDA runtime | Host facts recorded in README, ADR and this proposal |
| 2 | Native MuJoCo CPU environment and decision log implemented; provisional planar biped is a study model | Reproducible training/evaluation and model audit |
| 3 | Defer accelerated bake-off until a CUDA host exists or CPU throughput becomes the measured bottleneck | Same-task protocol and reference workflows are documented |
| 4 | Define canonical names/frames/units and description validation checks | Model contract and first skeleton asset |
| 5 | Define actuator bench measurements before selecting motors or freezing geometry | Measurement protocol and fit schema |
| 6 | Add a 1-DOF parameterized actuator model, then a leg; no locomotion reward tuning until the model compiles and responds plausibly | Calibratable twin and staged gate |
| 7 | Implement baseline controller, then reference tracking and RL comparison | Recorded policy evaluations with provenance |

### Suggested first 90 days, conditional on part-time availability

These are work packages, not calendar promises; supplier lead times and actuator access are unknown.

- **Weeks 1–2: platform and decision log.** Host inventory and native MuJoCo CPU baseline are complete. Next, profile the training workload and retain the hardware facts with experiment provenance; only stage a remote NVIDIA comparison if run time blocks useful iteration.
- **Weeks 3–4: common benchmark.** Reproduce a known simple joint/pendulum model and then a simplified leg in native MuJoCo. Capture contact/actuator traces, reset/replay behavior, and training wall time. Run a second backend only on supported hardware and after defining a same-model parity check.
- **Weeks 5–8: kinematics and mass budget.** Build a coarse skeleton from silhouette and constraints; choose provisional joint axes/frames without freezing actuator geometry. Validate conventions, self-collision, and mass/COM/inertia assumptions. List missing actuator data.
- **Weeks 9–12: model and test plan.** Add configurable actuator behavior, safe controller limits, baseline stand/balance and fall reset. Create an actuator measurement protocol and joint-module test specification. Decide what evidence is required before motor selection and lower-body CAD release.

**Day-90 intended result:** a runnable, versioned provisional twin, a measured toolchain decision, an uncertainty-aware preliminary mass/inertia budget, and a concrete actuator/joint test protocol. This is not a claim that a calibrated whole-body twin or transferable walking policy can be completed before the physical actuator module exists.

## Decisions intentionally left open

- Final height/mass, morphology, number of actuators, hands/grippers, and manipulation goals.
- Motor type, gearing, drive electronics, bus protocol, safety factor, and workstation/edge-compute hardware.
- ROS 2 distribution and the exact division between ROS control loops and a learned policy runtime.
- Which accelerated simulator/backend wins the bake-off; which rendering/synthetic-data tool is needed.
- Whether CAD master files live in this repo, a CAD collaboration system, or released neutral files plus revision links.
- Which movement goals matter first: standing, slow walking, disturbance recovery, gestures, or object interaction.

The first actual actuator and safety constraints should reduce these choices. Do not freeze a full body around an aesthetic target before the peak torque, speed, thermal, battery, and impact constraints are understood.

## Sources and reading list

Primary docs, code, and research accessed for this proposal on 1 November 2026:

- NVIDIA, [Isaac Lab overview](https://developer.nvidia.com/isaac/lab); [Isaac Lab 3.0 docs](https://docs.nvidia.com/isaaclab/isaac-lab); [framework paper](https://arxiv.org/abs/2511.04831).
- NVIDIA, Google DeepMind, Disney Research / Linux Foundation, [Newton project](https://github.com/newton-physics/newton); [current Newton documentation](https://newton-physics.github.io/newton/); [NVIDIA overview](https://developer.nvidia.com/newton-physics).
- Google DeepMind, [MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/); [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html); [MuJoCo Warp](https://mujoco.readthedocs.io/en/3.6.0/mjwarp/); [MuJoCo Playground paper](https://arxiv.org/abs/2502.08844) and [implementation](https://github.com/google-deepmind/mujoco_playground).
- NVIDIA, [Isaac Sim overview](https://developer.nvidia.com/isaac/sim).
- ROS 2 Control, [framework documentation](https://control.ros.org/jazzy/doc/ros2_control/doc/index.html); [MuJoCo hardware integration](https://control.ros.org/jazzy/doc/mujoco_ros2_control/mujoco_ros2_control/docs/hardware_interface.html).
- N. Sobanbabu et al., [SPI-Active: Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning](https://arxiv.org/abs/2505.14266), 2025.
- [PolySim: Bridging the Sim-to-Real Gap for Humanoid Control via Multi-Simulator Dynamics Randomization](https://arxiv.org/abs/2510.01708), 2025.
- [Simulator Adaptation for Sim-to-Real Learning of Legged Locomotion via Proprioceptive Distribution Matching](https://arxiv.org/abs/2604.11090), 2026.
- [Sim-to-Real of Humanoid Locomotion Policies via Joint Torque Space Perturbation Injection](https://arxiv.org/abs/2603.21853), 2026; [VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation](https://arxiv.org/abs/2511.15200), 2025; [FastSAC/FastTD3 humanoid recipe](https://arxiv.org/abs/2512.01996), 2025.
- [Validating Robotics Simulators on Real-World Impacts](https://arxiv.org/abs/2110.00541), 2021. Though older, this direct real-world contact comparison is a useful warning against assuming different simulators agree simply because they accept similar robot models.
- [Genesis documentation](https://genesis-world.readthedocs.io/en/latest/) and [source project](https://github.com/Genesis-Embodied-AI/Genesis).

Research note: publication claims and vendor performance descriptions should be treated as hypotheses until reproduced on the project's chosen robot, computer, and task. The proposal prioritizes primary documentation, released code, and papers; it does not rely on vendor speed claims as decision evidence.
