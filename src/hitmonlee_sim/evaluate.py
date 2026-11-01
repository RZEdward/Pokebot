"""Evaluate a trained policy and optionally record a MuJoCo rollout video."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from hitmonlee_sim.env import MODEL_PATH, PlanarBipedBalanceEnv


def _wilson_interval(
    successes: int, trials: int, z: float = 1.959963984540054
) -> list[float] | None:
    """Return a two-sided 95% Wilson interval for a binomial success rate."""
    if trials <= 0:
        return None
    rate = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (rate + z_squared / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z_squared / (4.0 * trials * trials))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="PPO checkpoint (.zip)")
    parser.add_argument(
        "--robot-model",
        type=Path,
        default=MODEL_PATH,
        help="Robot MJCF path; defaults to packaged model",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--render", action="store_true", help="Show the native MuJoCo viewer")
    parser.add_argument(
        "--record", type=Path, default=None, help="Write an MP4 from the overview camera"
    )
    parser.add_argument("--camera", choices=("overview", "head_camera"), default="overview")
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be a positive integer.")
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    if not np.isfinite(args.episode_seconds) or args.episode_seconds <= 0.0:
        parser.error("--episode-seconds must be finite and greater than zero.")
    if not args.model.is_file():
        parser.error(f"--model does not exist or is not a file: {args.model}")
    if not args.robot_model.is_file():
        parser.error(f"--robot-model does not exist or is not a file: {args.robot_model}")
    return args


def main() -> None:
    args = _parse_args()
    robot_model_path = args.robot_model.resolve()
    render_mode = "rgb_array" if args.record else "human" if args.render else None
    env = PlanarBipedBalanceEnv(
        render_mode=render_mode,
        episode_seconds=args.episode_seconds,
        model_path=robot_model_path,
    )
    model = PPO.load(args.model, device="cpu")
    results = []
    writer = None
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.record, fps=50, codec="libx264", macro_block_size=2)
    try:
        for episode in range(args.episodes):
            observation, reset_info = env.reset(seed=args.seed + episode)
            reset_state = {
                key: reset_info[key]
                for key in (
                    "root_x",
                    "root_z",
                    "root_pitch",
                    "root_x_velocity",
                    "root_z_velocity",
                    "root_pitch_velocity",
                    "joint_angles",
                    "joint_velocities",
                )
            }
            total_reward = 0.0
            terminated = truncated = False
            heights = []
            pitches = []
            limit_margins = []
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                heights.append(info["root_z"])
                pitches.append(info["root_pitch"])
                limit_margins.append(info["joint_limit_margin"])
                if writer is not None:
                    writer.append_data(env.render(camera_name=args.camera))
            result = {
                "episode": episode,
                "seed": args.seed + episode,
                "initial_state": reset_state,
                "return": total_reward,
                "steps": int(info["episode_step"]),
                "simulated_seconds": float(info["simulation_time"]),
                "fell": bool(terminated),
                "mean_root_height_m": float(np.mean(heights)),
                "max_abs_pitch_rad": float(np.max(np.abs(pitches))),
                "minimum_joint_limit_margin_rad": float(np.min(limit_margins)),
            }
            results.append(result)
            print(json.dumps(result))
    finally:
        if writer is not None:
            writer.close()
        env.close()

    successes = sum(not row["fell"] for row in results)
    report = {
        "model": str(args.model.resolve()),
        "robot_model": str(robot_model_path),
        "robot_model_sha256": hashlib.sha256(robot_model_path.read_bytes()).hexdigest(),
        "episodes": results,
        "success_rate_no_fall": float(np.mean([not row["fell"] for row in results])),
        "success_rate_wilson_95_interval": _wilson_interval(successes, len(results)),
        "mean_return": float(np.mean([row["return"] for row in results])),
        "mean_steps": float(np.mean([row["steps"] for row in results])),
        "video": str(args.record.resolve()) if args.record else None,
    }
    report_path = (
        args.record.with_suffix(".evaluation.json")
        if args.record
        else args.model.with_suffix(".evaluation.json")
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluation report: {report_path.resolve()}")
    if args.record:
        print(f"Video: {args.record.resolve()}")


if __name__ == "__main__":
    main()
