"""Replay the provisional low-speed gait reference on matched, unrandomized commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from robot_twin.controllers import ankle_upright_pd_action
from robot_twin.env import CONFIG_PATH, HumanoidBalanceEnv, load_config
from robot_twin.locomotion import phase_gait_action

COMMANDS = {
    "forward": {"vx_m_s": 0.15, "vy_m_s": 0.0, "yaw_rate_rad_s": 0.0},
    "reverse": {"vx_m_s": -0.15, "vy_m_s": 0.0, "yaw_rate_rad_s": 0.0},
    "left": {"vx_m_s": 0.0, "vy_m_s": 0.12, "yaw_rate_rad_s": 0.0},
    "right": {"vx_m_s": 0.0, "vy_m_s": -0.12, "yaw_rate_rad_s": 0.0},
    "turn_left": {"vx_m_s": 0.0, "vy_m_s": 0.0, "yaw_rate_rad_s": 0.3},
    "turn_right": {"vx_m_s": 0.0, "vy_m_s": 0.0, "yaw_rate_rad_s": -0.3},
}


def root_yaw(quaternion: np.ndarray) -> float:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    rotation = matrix.reshape(3, 3)
    return float(math.atan2(rotation[1, 0], rotation[0, 0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path")
    parser.add_argument("--metrics", type=Path, default=None, help="Output JSON metrics path")
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    config = load_config()
    env = HumanoidBalanceEnv(config_path=CONFIG_PATH, render_mode="rgb_array", randomize=False)
    env.control_mode = "direct"
    baseline = config["evaluation_baselines"]["ankle_upright_pd"]
    gait = config["locomotion_reference"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics or args.output.with_suffix(".json")
    render_every = max(1, round((1.0 / env.dt) / 10.0))
    rows: list[dict] = []
    with imageio.get_writer(args.output, fps=10, codec="libx264", macro_block_size=2) as video:
        for index, (name, command) in enumerate(COMMANDS.items()):
            env.randomize = False
            env.reset(seed=7100 + index)
            initial_position = env.data.qpos[:3].copy()
            initial_yaw = root_yaw(env.data.qpos[3:7])
            min_upright = 1.0
            fell = False
            steps = 0
            limit = min(env.max_steps, int(args.seconds / env.dt))
            while steps < limit:
                gait_action = phase_gait_action(env, command, env.data.time, gait)
                stabilizer = ankle_upright_pd_action(env, baseline)
                _, _, terminated, truncated, info = env.step(
                    np.clip(gait_action + stabilizer, -1.0, 1.0)
                )
                if steps % render_every == 0:
                    video.append_data(env.render())
                min_upright = min(min_upright, info["upright"])
                steps += 1
                if terminated:
                    fell = True
                    break
                if truncated:
                    break
            displacement = env.data.qpos[:3] - initial_position
            rows.append(
                {
                    "command": name,
                    "command_body_frame": command,
                    "seed": 7100 + index,
                    "duration_s": steps * env.dt,
                    "fell": fell,
                    "min_upright": min_upright,
                    "displacement_world_m": displacement.astype(float).tolist(),
                    "yaw_change_rad": root_yaw(env.data.qpos[3:7]) - initial_yaw,
                    "final_root_position_m": env.data.qpos[:3].astype(float).tolist(),
                }
            )
            print(json.dumps(rows[-1]))
    model_path = env.model_path
    env.close()
    payload = {
        "schema_version": 1,
        "status": "EXPERIMENTAL_REFERENCE_ROLLOUTS",
        "task": "phase_gait_reference_with_ankle_pd",
        "physics_backend": "MuJoCo",
        "robot_config": str(CONFIG_PATH),
        "robot_model": str(model_path),
        "robot_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "gait_settings": gait,
        "episode_seconds_requested": args.seconds,
        "episodes": rows,
        "limitations": [
            "No PPO locomotion training is included in this baseline evaluation.",
            "The gait is an unvalidated phase trajectory without contact-phase feedback.",
            "The mass, inertias, actuator response, and ground parameters are provisional.",
        ],
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved video: {args.output}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
