# Phase 1–4 execution plan

This plan governs the current simulation-first foundation work. Each gate is recorded separately so an attractive render or a long PPO run cannot be mistaken for validated hardware behavior.

## Operating principles

- Keep robot-independent simulation, task, experiment, and visualization interfaces separate from a robot embodiment's geometry and limits.
- Use the current CAD envelope and the proposed 1.35 m / 25 kg context as a design case only. The existing CAD marks actuator selection and mass properties as unapproved. Never label a placeholder as measured, vendor-specified, or physically calibrated.
- Track every physical parameter with provenance, units, and uncertainty. Preserve multiple mass/center-of-mass cases for robustness experiments.
- Keep visual shell geometry, collision geometry, and inertial geometry as separate layers. Do not use the downloaded appearance mesh for collision or mass properties.
- Save each experiment's model/config hashes, software versions, seed, metrics, checkpoint, and replay/video under its run directory. Preserve failed or weak policies as evidence.
- Design sensor mounts and stable frame IDs now, but defer camera/audio policy inputs until the mechanics and proprioceptive control gates pass.
- Use upstream patterns from MuJoCo Playground and mjlab for explicit task composition and evaluation. Benchmark another backend only with an equivalent model, task, and replay; retain Isaac Lab/Newton, Genesis, and ROS 2 integrations as optional seams rather than runtime assumptions.

## Gate 1 — model and code audit

1. Inspect the saved FreeCAD assembly, its generator/config, packaged MJCF, training code, evaluation code, and generated artifacts.
2. Record the differences among CAD size, physics-model size, total mass, joints, collision shapes, and current training task.
3. Identify body-frame and joint-axis errors before adding skills. Verify asset/license provenance and generated-file ignore behavior.
4. Define the shared embodiment/config contract and migration path so a second robot can use the same simulator/task/experiment APIs without copying Hitmonlee constants.

**Exit evidence:** an audit note lists verified values, assumptions, broken/misaligned items, and open hardware decisions; baseline commit and run artifact remain reproducible.

## Gate 2 — body and sensing-ready design

1. Correct the body coordinate convention and human-resting arm geometry; validate every joint origin/axis against the kinematic chain.
2. Keep the CAD source editable and regenerable from a versioned parameter file. Add mass-budget records per link/component with `PLACEHOLDER`, `CAD_ESTIMATE`, `VENDOR_SPEC`, `MEASURED`, or `FITTED` authority and uncertainty.
3. Add explicit forward stereo/eye, side, rear, downward, and upward camera frames plus synchronized microphone-array and speaker mounting sites. Record intended fields of view, time bases, calibration transforms, and acoustic feedback concerns. Hardware, lens, microphone count, and speaker remain unselected until component review.
4. Build a 3D collision/dynamics model from segmented links, not the unsegmented character mesh. Keep removable shell panels visual-only unless their material, thickness, fasteners, and mass are known.
5. Check model scale, mass sum, center of mass, support contacts, joint limits, actuator bounds, and representative self-clearance poses. Preserve detailed CAD validation outputs.

**Exit evidence:** editable source + generator + native CAD + neutral export; 3D MJCF with named links/joints/sensor frames; machine-readable validation and explicit mass-property provenance. No supplier-ready claim until selected parts, material, tolerances, load cases, and structural analysis exist.

## Gate 3 — task and training ladder

Training sequence (do not bundle every behavior into one reward or one checkpoint):

1. Establish a non-learning stand controller and deterministic replay on a nominal level plane.
2. Train/evaluate proprioceptive balance under a fixed grid of friction, slope, and contact-compliance cases, plus held-out combinations and bounded disturbances. Report per-condition success, falls, recovery time, actuator-limit events, and uncertainty; include an untrained/baseline comparison.
3. Add command-conditioned stepping and walking: forward/reverse, lateral translation, and heading turns. A first planar velocity-tracking PPO environment is now implemented and trained experimentally for translation commands; it is not a validated gait yet. The current translation gate is a mean tracking score of at least 0.70 in each command over three held-out seeds, with no falls in the six-second episodes. Progress from flat ground to surface variations only after held-out gait stability and command tracking are repeatable.
4. Add crouch, sit/stand, supine/prone lying and recovery as explicit pose-transition tasks with contact and self-collision checks.
5. Treat jump, kick, and punch as impact/dynamic skills requiring actuator torque-speed/thermal limits, structural and contact assumptions, and harnessed hardware review. Sim results alone do not approve hardware trials.

Use reference motion/trajectory tracking or model-based control as a baseline and curriculum; use RL for residual adaptation or clearly bounded tasks. Save a separate policy/config per skill or an explicitly conditioned multi-skill policy. Evaluate on held-out seeds and parameter cases, and compare against a deterministic baseline. Run budget is set by measured CPU throughput and does not substitute for acceptance criteria.

**Exit evidence:** trained checkpoints only for tasks that meet a prewritten held-out criterion; non-passing tasks remain marked untrained/experimental with reproducible run metadata and failure examples.

## Gate 4 — developer observability and sensor seams

1. Keep live MuJoCo view optional and non-blocking for headless throughput.
2. Provide one run index linking config, model revision/hash, training/evaluation curves, checkpoint, per-episode metrics, and replay video.
3. Provide calibrated frame IDs for camera views and a future audio interface: synchronized multi-channel samples, waveform/spectrogram, VAD/ASR/intent events, latency, and confidence. Do not fabricate audio or camera policy success before real/sim sensor pipelines are tested.
4. Verify the UI can inspect the active run and replay an evaluation independently of a training process.

**Exit evidence:** a reproducible run can be launched, watched live, inspected after completion, and replayed from saved artifacts; camera/audio panels show their defined input/output contracts and remain clearly separate from control authority.

## Scope boundary for this work session

The first success is a credible, inspectable 3D design/model and a reproducible balance experiment across a bounded synthetic surface matrix. Command-conditioned planar locomotion is now an experimental task; it must pass held-out command-by-command evaluation before being called walking. Sitting/get-up, prone/supine recovery, crouching, jump, kick, and punch remain separate curriculum items. No training duration, policy score, CAD envelope, or camera/microphone mount can establish real-world transfer without chosen components, measured masses/inertias, actuator characterization, sensor calibration, and safeguarded physical validation.
