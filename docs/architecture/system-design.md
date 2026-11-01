# System design: simulation-first robot software

## Design goals

1. Train a control skill without coupling each physics step to ROS 2 middleware.
2. Preserve one typed joint/observation/action contract across simulation, replay, and eventual hardware.
3. Keep learned behaviors behind deterministic limits and a safety supervisor.
4. Give every experiment reproducible model, environment, seed, and policy metadata.
5. Make model fidelity evidence-based: distinguish CAD estimates, vendor specifications, measured values, and fitted values.

## Runtime boundaries

```text
Sensors / drives → timestamped state estimator → safety supervisor → skill controller → command limiter → actuator transport
                         ↑                          ↑                    ↑
                   perception models         behavior intents       learned policy
```

The simulation backend owns rigid-body/contact stepping, sensor generation, and actuators. The Gymnasium adapter exposes a task-specific observation and action tensor. The trainer speaks directly to that adapter for throughput. The eventual ROS/hardware bridge is a separate execution adapter; it consumes a deployment artifact and the same semantic joint/action schema. ROS topics are not the policy's inner-loop transport contract.

## Current task contract

- **Embodiment:** planar biped with a root constrained to sagittal translation/rotation, six actuated leg joints, two feet, and a torso-mounted camera frame.
- **Policy observation:** root height, root pitch as sine/cosine, root linear/angular velocity, each leg joint angle and velocity, and previous normalized action. This is intentionally proprioceptive.
- **Policy action:** six normalized desired joint positions. The simulator applies a position-servo model with finite gain, damping, position limits, and force saturation. The action space is not asserted to match a selected physical actuator.
- **Reward:** remain upright and near nominal height, penalize lateral/forward drift, excessive joint motion, and abrupt commands. Episode termination represents a fall or leaving the safe region.
- **Evaluation:** deterministic rollouts, survival duration, root height/pitch, joint-limit/effort events, and seed-wise success. Mean reward alone is insufficient.

## Model data authority

CAD is authoritative for manufactured dimensions and part revisions. A robot specification records semantic names, frames, units, joint limits, and sources for mass properties. MJCF carries MuJoCo-specific actuators, contacts, sensors, and options. URDF is generated/maintained for ROS kinematics and tooling. USD is optional for Isaac scenes. Cross-format checks compare names, axes, limits, transforms, masses, centers of mass, and inertias; the XML/URDF/USD files are not presumed to round-trip losslessly.

Every numeric model parameter should carry a source label:

- `CAD_ESTIMATE`: derived from geometry/material assumptions.
- `VENDOR_SPEC`: supplied by component maker; note test conditions.
- `MEASURED`: raw bench measurement with data file and protocol ID.
- `FITTED`: identified from measured trajectories with method, residual, and uncertainty.
- `PLACEHOLDER`: provisional value that must not be used to claim transfer fidelity.

## Training lifecycle

1. Load a pinned model revision and task config.
2. Seed Python, NumPy, Torch, and simulator state.
3. Train headless by default; optional human rendering deliberately slows training.
4. Atomically publish checkpoint, telemetry and run metadata.
5. Evaluate on held-out initial conditions and seeds.
6. Replay a deterministic episode and optionally record camera frames to video.

## Step 4 prototype and extension boundary

The prototype renders a head-camera image, passes a timestamped RGB frame through a red-dominant blob detector, and maps speech transcripts to an allowlisted typed intent. An optional Vosk adapter can capture a single microphone utterance or transcribe a 16 kHz mono PCM WAV locally, using a constrained command vocabulary. A small dispatch gate resolves `STAND` to the registered balance skill, rejects unavailable walking skills and low-confidence transcripts, and turns `STOP` into a supervisor safe-stop request. This is a narrow command recognizer, not general dictation or speaker verification. A mature perception node should return calibrated observations (camera timestamp, frame ID, intrinsics/extrinsics, detections/pose and confidence). A behavior manager may select only approved skills. No foundation model or speech recognizer may directly emit joint commands. Sensor delay, dropped frames and uncertainty must be represented in simulation before vision/audio policy transfer is attempted.

## Current non-goals

- A high-fidelity 3D full-body digital twin.
- Claims that the learned policy transfers to hardware.
- Camera-based locomotion, voice recognition, manipulation, or autonomous recovery.
- A real motor-bus driver or physical robot safety implementation.
- Choosing actuator hardware, final mass/height, or ROS distribution before target compute and component support are known.
