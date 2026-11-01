# Body design and motion validation

Status: engineering plan for the `HITMONLEE-BODY-LAYOUT-01` concept; this is not a released robot design.

## How the geometry, motion, and appearance fit together

We do not create a CAD assembly for every possible limb pose. CAD describes the robot's rigid parts, joint locations, interfaces, and geometry. A kinematic model describes each joint axis and allowed range. The simulator then moves those links through joint configurations and checks contact, self-collision, actuator response, and task performance. A saved pose is useful for review, but it is not a separate design.

The downloaded GLB is not an engineering model. It is a small, unrigged, two-mesh appearance asset with unknown units. It is useful as a silhouette and visual proportion reference only. Its 1,350 mm normalized height is a project assumption, not source evidence. Its triangles do not define link boundaries, shell thickness, joints, material, inertia, or tolerances. The native CAD assembly therefore uses independently created concept envelopes and shell proxies. Those proxies are not yet judged to fit the surface: the mesh comparison is only a global bounding-box check, and the preview reveals unresolved limb/shell proportions. The source model should not constrain detailed engineering until dimensions and requirements are independently established.

## Validation approach: configurations rather than pose-specific CAD

1. **Define frames and joints.** Give every link a local frame and every joint an origin, axis, type, range, velocity/effort limit, and actuator mapping. Keep the transforms in a reviewed robot specification and export the same conventions to MJCF and, when needed, URDF. A joint is not validated just because two CAD solids appear connected.
2. **Validate static extreme configurations.** Check neutral pose, each joint's individual limits, coupled combinations that tighten clearances, self-collision, ground contact, cable/service envelopes, and sensor fields of view. Store failed configurations with the exact model/config hash so they can be replayed.
3. **Sample the reachable space.** Use deterministic boundary/corner cases and seeded joint-space samples. A raw Cartesian product grows exponentially and is not a useful exhaustive test for a many-jointed robot. Use collision checking and targeted sampling to find near-collisions, then retain representative worst cases.
4. **Validate trajectories in simulation.** For the selected tasks, test the time-varying swept volume, foot-ground contact, actuator torque/speed/thermal limits, joint limit margins, balance, and recovery. Randomize uncertain mass, friction, delay, and actuator parameters only within defensible ranges; record the ranges and keep a fixed held-out evaluation set.
5. **Close the loop with hardware.** Replace envelope values with measured vendor interfaces and robot mass properties, compare simulated joint/contact traces with bench and real-robot measurements, and update the model. A digital twin is a calibrated, versioned model plus measured evidence, not just CAD or a successful policy run.

This catches geometric clearance issues and motion/dynamics failures at different levels. Collision-free motion is not proof of structural strength, thermal safety, wiring reliability, or stable walking.

## Repository architecture and source-of-truth boundaries

The repo currently has three distinct model layers:

- `design/cad/body-layout-v1.json` is the parameter source for the current FreeCAD-generated envelope assembly. `tools/cad/build_hitmonlee_body_v1.FCMacro` builds the native `.FCStd`, concept STEP, preview images, axis references, and validation manifest. This is a regenerable script-driven FreeCAD document, not a native PartDesign feature tree or a manufacturing release.
- `assets/reference_models/` contains the attributed third-party GLB and a normalized OBJ used only to visualize the source silhouette in FreeCAD. Keep it separate from original design geometry and from exports intended for suppliers.
- `src/hitmonlee_sim/models/` contains the current planar MuJoCo MJCF used by the runnable PPO standing-balance example. `src/hitmonlee_sim/` owns task, training/evaluation, telemetry, and visualization code. This 2D model is a training-loop foundation, not a simulation of the 3D CAD concept.

These layers are not yet generated from one authoritative robot specification: the v1 CAD is a fit-study while the MJCF is an independent planar test model. Do not imply that policy behavior in the planar MJCF validates the CAD. The next 3D gate is to freeze the joint/frame/mechanics schema and establish tested exports to MJCF and optional URDF; CAD and simulator should share link dimensions, joint frames, limits, inertial assumptions, and version/hash metadata. Prefer simulator-native MJCF for MuJoCo-specific contacts, actuators, and sensors. ROS/URDF is an interoperability and future hardware-control boundary, not a lossless replacement for MJCF.

## Next implementation gates

### A. Improve and qualify this concept export

- The current assembly contains 87 concept solids and its full STEP import preserves all 87 valid solids; overall bounds and total volume are checked in the validation report. Preserve that report and per-part manifest when changing geometry.
- Every exported solid in the native FCStd now stores both its dotted JSON source path and the resolved configuration value as inspectable properties. The generator rejects unresolved paths, and the report records that value beside each part. Geometry remains regenerated by the FreeCAD macro from JSON; the individual solid shapes are not live native feature-tree expressions.
- Thirty front/rear hollow shell-panel prototypes now export separately as STEP and STL. Their nominal radial inset is 3 mm; this is a modeling parameter, not a guaranteed minimum wall thickness. The STL has 285,718 facets. Shell-only STEP round-trip imports all 30 valid features and bounds agree within 0.014 mm, but its relative total-volume delta is 0.000122, narrowly above the current 0.0001 acceptance threshold, so that shell export is explicitly not marked verified.
- These panels are only packaging/appearance prototypes: no flanges, fasteners, inserts, tolerances, supplier process limits, structural/thermal validation, cable/service access, or fit checks are defined. Neither shell export is a manufacturing file.
- The JSON and FreeCAD generator now define a 28-joint provisional chain with parent/child link names, robot-frame origins and axes, and broad study ranges. The current saved FCStd/report still contain the earlier 14 axis references; rebuild and inspect them before claiming the new chain is present. The study ranges are for layout/motion exploration only; multi-axis actuator stack offsets, joint signs, cable routes, singularities, and hardware limits remain unverified.
- The two foot plate proxies touch the CAD ground datum at `z=0`. This establishes only a geometric datum contact, not a validated contact patch, sole, friction model, or stable support polygon.
- Maintain the editable source as FCStd + JSON + generator and improve shell proportions/fit using measured design requirements rather than undocumented source-mesh assumptions. The current FCStd remains a script-generated shape assembly, not a native PartDesign feature tree.
- Improve proxy proportions and verify actual surface deviation against the reference mesh (for visualization only); remove overlap/clearance assumptions from the claims until collision checks exist.
- Maintain a clear `NOT FOR MANUFACTURE` status until each part has a selected process/material, tolerances, interfaces, and verified load case.

### B. Establish a mechanics model before exterior detailing

- Choose requirements for height, payload, task, speed, impact limits, and serviceability; obtain actual actuator/gearbox/bearing/sensor candidates and vendor CAD/spec sheets.
- Decide DoFs from intended behavior and hardware constraints; define joint axes/ranges and build a 3D kinematic chain.
- Set measured/provisional link masses, centers of mass, and inertia tensors. Run static packaging, collision, ground-clearance, torque/speed, and stability checks.
- Design the structure, replaceable printed shells, infill, inserts, fasteners, wiring paths, and access covers around those constraints. Create manufacturing drawings and neutral exports only per released part.

### C. Create a 3D simulation/training foundation

- Build a minimal model with one validated leg or a reduced full-body model before a complex complete humanoid. Verify joint sign/range, reset contacts, actuator response, timestep/contact convergence, and deterministic replay against known traces.
- Add a simple meaningful task (for example, commanded weight shift or squat-to-stand) and a baseline controller before RL; then evaluate a policy against fixed seeds, perturbations, and a non-learning baseline.
- Save the exact MJCF, config, code revision, seed, dependency versions, and metrics with each run. Use the MuJoCo viewer for live inspection, telemetry/dashboard for training status, and deterministic replay/MP4 for review. Training performance and real-time rendering are separate modes because live viewing reduces throughput.
- Benchmark an alternative backend only with the same robot model, reset, action, contact, and evaluation checks. Existing repo research compares MuJoCo Playground, Isaac Lab/Newton, mjlab, Genesis, and ROS-control integration; follow upstream examples and licenses but avoid copying their robot-specific assumptions.

## Open-source architecture references

These are useful examples, not a mandate to adopt all their frameworks:

- [MuJoCo documentation](https://mujoco.readthedocs.io/) and [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground): simulator-native mechanics, reproducible tasks, training/evaluation, domain randomization, and trajectory visualization.
- [ROS 2 URDF tutorials](https://docs.ros.org/en/rolling/Tutorials/Intermediate/URDF/URDF-Main.html) and [ros2_control](https://control.ros.org/): link/joint descriptions and a hardware abstraction for later simulation-to-hardware controller reuse. Keep simulator-specific physics features explicit.
- [MoveIt Planning Scene](https://moveit.picknik.ai/main/doc/concepts/planning_scene/planning_scene.html): collision geometry, robot/environment collision checking, and allowed-collision management as a model-validation pattern. It is not a substitute for dynamics simulation.
- [Drake multibody plant documentation](https://drake.mit.edu/doxygen_cxx/group__multibody__plant.html): explicit multibody model, frames, joints, actuation, and contact as a cross-checking/mechanics reference.
- The repository's [open-source workflow review](../research/open-source-reference-workflows.md) records current tool tradeoffs, licenses, host constraints, and which patterns are being adopted or deferred.

## Current CAD verification status

The latest successfully verified v1 rebuild records 87 valid source OCC solids and 87 valid imported STEP features. Its JSON report includes resolved parameter values for all exported solids and the earlier 14 nominal axis records. The current JSON/generator edits add the proposed 28-joint study chain, but the native file has not yet been regenerated to include it. The separate shell STEP imports 30 valid panel features but misses the configured total-volume threshold; the shell STL was generated with 285,718 facets. These are CAD and packaging checks only, not a qualified deliverable. Surface fit, swept collision checks, final kinematics, loads, mass properties, material, tolerance, and motion remain unvalidated. Do not send these files to a printer or aluminium supplier as build-ready geometry.
