"""Compare a locomotion PPO policy with the inspectable gait reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import PPO

from robot_twin.controllers import ankle_upright_pd_action
from robot_twin.env import CONFIG_PATH, load_config
from robot_twin.locomotion import phase_gait_action
from robot_twin.locomotion_env import CommandedLocomotionEnv

COMMAND_NAMES = ("forward", "reverse", "left", "right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--randomization-intensity", type=float, default=1.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Checkpoint not found: {args.model}")
    if args.seconds <= 0 or args.episodes < 1:
        parser.error("--seconds and --episodes must be positive")
    if not 0.0 <= args.randomization_intensity <= 1.0:
        parser.error("--randomization-intensity must be within [0, 1]")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    config_path = args.model.parent / "robot_config_snapshot.json"
    if not config_path.is_file():
        config_path = CONFIG_PATH
    run_config = load_config(config_path)
    manifest_path = args.model.parent / "run.json"
    run_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    reference_mode = run_manifest.get("reference_mode", "ankle_pd")
    policy = PPO.load(args.model, device="cpu")
    env = CommandedLocomotionEnv(
        config_path=config_path,
        randomize=True,
        render_mode="rgb_array",
        reference_mode=reference_mode,
    )
    baseline_settings = run_config["evaluation_baselines"]["ankle_upright_pd"]
    gait_settings = run_config["locomotion_reference"]
    commands = dict(zip(COMMAND_NAMES, run_config["locomotion_training"]["commands"], strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics or args.output.with_suffix(".json")
    render_every = max(1, round((1.0 / env.dt) / 10.0))
    rows: list[dict] = []

    try:
        with imageio.get_writer(args.output, fps=10, codec="libx264", macro_block_size=2) as video:
            for name, command in commands.items():
                for episode in range(args.episodes):
                    seed = 12000 + COMMAND_NAMES.index(name) * 100 + episode
                    for controller in ("phase_gait_reference", "ppo"):
                        env.set_randomization_intensity(args.randomization_intensity)
                        observation, _ = env.reset(seed=seed, options={"command": command})
                        initial_position = env.data.qpos[:3].copy()
                        initial_state = {
                            "root_position_m": initial_position.astype(float).tolist(),
                            "root_quaternion_wxyz": env.data.qpos[3:7].astype(float).tolist(),
                            "root_linear_velocity_m_s": env.data.qvel[:3].astype(float).tolist(),
                            "root_angular_velocity_rad_s": env.data.qvel[3:6]
                            .astype(float)
                            .tolist(),
                            "joint_positions_rad": env.data.qpos[env.base.qpos_ids]
                            .astype(float)
                            .tolist(),
                            "joint_velocities_rad_s": env.data.qvel[env.base.qvel_ids]
                            .astype(float)
                            .tolist(),
                            "gait_phase_offset_rad": env.gait_phase_offset,
                        }
                        tracking_scores = []
                        body_velocities = []
                        upright_values = []
                        steps = 0
                        fell = False
                        step_limit = min(env.max_steps, int(args.seconds / env.dt))
                        while steps < step_limit:
                            if controller == "ppo":
                                action, _ = policy.predict(observation, deterministic=True)
                                transition = env.step(action)
                            else:
                                command_dict = {
                                    "vx_m_s": command[0],
                                    "vy_m_s": command[1],
                                    "yaw_rate_rad_s": command[2],
                                }
                                gait = phase_gait_action(
                                    env.base,
                                    command_dict,
                                    env.data.time,
                                    gait_settings,
                                    env.gait_phase_offset,
                                )
                                stabilizer = ankle_upright_pd_action(env.base, baseline_settings)
                                transition = env.step_reference(
                                    np.clip(gait + stabilizer, -1.0, 1.0)
                                )
                            observation, _, terminated, truncated, info = transition
                            tracking_scores.append(info["velocity_tracking_score"])
                            body_velocities.append(info["body_velocity_m_s"])
                            upright_values.append(info["upright"])
                            if controller == "ppo" and steps % render_every == 0:
                                video.append_data(env.render())
                            steps += 1
                            if terminated:
                                fell = True
                                break
                            if truncated:
                                break
                        rows.append(
                            {
                                "controller": controller,
                                "command": name,
                                "command_body_m_s": list(command),
                                "seed": seed,
                                "initial_state": initial_state,
                                "duration_s": steps * env.dt,
                                "fell": fell,
                                "randomization": info["randomization"],
                                "mean_tracking_score": float(np.mean(tracking_scores)),
                                "mean_body_velocity_m_s": np.mean(body_velocities, axis=0).tolist(),
                                "min_upright": float(np.min(upright_values)),
                                "displacement_world_m": (env.data.qpos[:3] - initial_position)
                                .astype(float)
                                .tolist(),
                            }
                        )
                        print(json.dumps(rows[-1]))
    finally:
        env.close()

    grouped = {}
    for controller in ("phase_gait_reference", "ppo"):
        samples = [row for row in rows if row["controller"] == controller]
        grouped[controller] = {
            "episodes": len(samples),
            "no_fall_fraction": float(np.mean([not row["fell"] for row in samples])),
            "mean_tracking_score": float(np.mean([row["mean_tracking_score"] for row in samples])),
            "mean_survival_s": float(np.mean([row["duration_s"] for row in samples])),
        }
    model_path = Path(run_config["model_path"])
    if not model_path.is_absolute():
        snapshot_candidate = config_path.parent / model_path
        model_path = snapshot_candidate if snapshot_candidate.is_file() else Path.cwd() / model_path
    payload = {
        "schema_version": 1,
        "status": "EXPERIMENTAL_HELD_OUT_COMMAND_EVALUATION",
        "task": "command_conditioned_velocity_tracking",
        "episodes_requested_per_command_controller": args.episodes,
        "episode_seconds": args.seconds,
        "robot_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "controller_summary": grouped,
        "episodes": rows,
        "success_criterion": "No fall and mean tracking score >= 0.70 in every command condition.",
        "limitations": [
            "Simulation only, with provisional masses, inertias, contact and actuator parameters.",
            "Held-out episodes vary seeds only; physical hardware transfer is not evaluated.",
        ],
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved evaluation video: {args.output}")
    print(f"Saved evaluation metrics: {metrics_path}")


if __name__ == "__main__":
    main()
