# Provisional model card: planar biped v0.1

This MJCF is an executable mechanics example. It is intentionally **not** a life-size or calibrated digital twin.

## Scope

- Planar sagittal dynamics: pelvis/root x and z translation plus torso pitch.
- Six position-actuated joints: bilateral hip, knee, ankle.
- Two rigid feet with explicit contact/friction settings.
- A torso/head camera frame and fixed red visual target for perception-interface work.
- Nominal model height is approximately 1.6 m; summed link/actuator mass is approximately 17 kg before cables, batteries, compute, shell, hands, and real actuator modules. It is a study morphology, not the earlier 25 kg product target.

## Explicit assumptions to replace with evidence

- Link capsule/box geometry is a collision/visual simplification, not final CAD.
- Link masses are provisional; inertias are generated from collision geometry and mass by MuJoCo's compiler.
- Servo gains (`kp=75`, `kv=7`), force range (±55 N·m), armature, joint damping, and torque saturation are placeholders.
- Sole friction 0.9 is a starting simulation value, not a measured material coefficient.
- Root is constrained to the sagittal plane. The model cannot fall sideways, balance laterally, or test real 3D biped stability.
- No battery voltage, current/thermal behavior, CAN timing, backlash, structural compliance, tendon/cable loads, or sensor noise has been calibrated.

## Validation before extending

1. Run `python -m hitmonlee_sim.inspect_model --output runs/model-audit.json`; inspect the MJCF SHA-256, required joints/actuators/camera frames, timestep, mass and reset contact checks.
2. Confirm contact feet begin near the ground and the passive model falls plausibly.
3. Plot joint ranges and ensure action targets stay within the declared limits.
4. Compare a known joint step in native MuJoCo with the selected physical actuator once measured data exists.
5. Add lateral DOF, shoulders/arms, and realistic as-built mass/inertias only when there is a validated question for each addition.
