# ADR 0001 — runnable simulation and learning baseline

- **Status:** accepted as the first development baseline; host inventory is complete, revisit after actuator selection or a throughput bottleneck.
- **Date:** 1 November 2026

## Decision

Use native MuJoCo on CPU as the canonical first simulator, Gymnasium as the task interface, and Stable-Baselines3 PPO for a deliberately small balance policy. The run path supports Windows 11 and Python 3.11–3.13. The model is a custom planar biped, so it can be executed and tested without relying on a GPU or an external humanoid asset.

Provide two ways to inspect results: the MuJoCo passive viewer during training, and a local-only web monitor reading structured telemetry. A deterministic evaluator records a rollout to MP4 using MuJoCo's renderer.

## Evidence and constraints

- The project directory is on Windows 11; Python 3.13 is installed. The laptop has an Intel Core Ultra 5 125H (14 cores / 18 logical processors), Intel Arc graphics, and 16 GiB RAM.
- Installed PyTorch is CPU-only (`torch.cuda.is_available()` is false, CUDA build is absent, device count is zero); `nvidia-smi` and `nvcc` are unavailable. The current machine has no usable CUDA device for the proposed NVIDIA-Warp training paths.
- The full robot and actuator are not defined or measured, so neither GPU throughput nor a specific physical motor model can yet be optimized.
- Native MuJoCo provides the smallest path to a inspectable, physically simulated multi-body model on the current host. Its public Python bindings support Windows and Python 3.11+.
- Isaac Lab 3 / Newton and MuJoCo Warp are meaningful candidates on a later supported NVIDIA host, if throughput justifies it. Current Genesis docs state Intel GPU rendering support but no Intel physics backend, so it is not an Arc acceleration path. Full Isaac Sim is deferred until sensor rendering or scene validation is needed and suitable hardware is available.

## Consequences

- The first learner is CPU-capable but not intended to train a large visual policy at scale.
- The planar model and position-servo gains are placeholders. The setup demonstrates the engineering loop, not sim-to-real transfer.
- When actuator candidates are available, re-run the tool bake-off with measured joint data and a common 1-DOF/leg benchmark. Preserve the same task/observation/action schemas when adding a second backend.
- Revisit Python pinning if Isaac Lab's Python 3.12 environment becomes the primary training platform. Do not force all packages into one interpreter merely to avoid maintaining a second environment.

## Revisit triggers

1. GPU inventory shows a supported NVIDIA device and expected RL wall time becomes a bottleneck.
2. A measured actuator module provides a transfer-relevant test trace.
3. Camera/depth training, synthetic-data generation, or USD scene authoring becomes an active task.
4. The hardware target and motor SDK make a ROS 2 distribution/runtime decision necessary.
