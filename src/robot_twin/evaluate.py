"""Evaluate a PPO controller over named contact conditions and save an MP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from stable_baselines3 import PPO

from robot_twin.env import CONFIG_PATH, HumanoidBalanceEnv, load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Robot config snapshot; defaults to the checkpoint's run snapshot if present",
    )
    ap.add_argument("--output", type=Path, default=None, help="Optional MP4 replay path")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument(
        "--mass-scales",
        type=float,
        nargs="+",
        default=[0.8, 1.0, 1.2],
        help="Whole-robot mass/inertia scale for the evaluation matrix",
    )
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    if a.episodes < 1:
        ap.error("--episodes must be positive")
    if not a.mass_scales or any(scale <= 0 for scale in a.mass_scales):
        ap.error("--mass-scales must contain positive values")
    if a.live and a.output:
        ap.error("Use either --live or --output for an evaluation run, not both.")
    if not a.model.is_file():
        ap.error(f"Checkpoint does not exist: {a.model}")
    policy = PPO.load(a.model, device="cpu")
    config_path = a.config or (
        a.model.parent / "robot_config_snapshot.json"
        if (a.model.parent / "robot_config_snapshot.json").is_file()
        else CONFIG_PATH
    )
    cfg = load_config(config_path)
    rows = []
    render_mode = "rgb_array" if a.output else "human" if a.live else None
    env = HumanoidBalanceEnv(config_path=config_path, render_mode=render_mode)
    conditions = list(cfg["surfaces"].items())
    mass_bounds = cfg["randomization"]["mass_scale"]
    if any(scale < mass_bounds[0] or scale > mass_bounds[1] for scale in a.mass_scales):
        ap.error(f"--mass-scales must be within configured training bounds {mass_bounds}")
    writer = None
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(a.output, fps=50, codec="libx264", macro_block_size=2)
    try:
        for idx, (name, surface) in enumerate(conditions):
            for mass_index, mass_scale in enumerate(a.mass_scales):
                for episode in range(a.episodes):
                    env.randomize = False
                    seed = 2000 + idx * 10000 + mass_index * 1000 + episode
                    for controller in ("neutral_hold", "ppo"):
                        obs, _ = env.reset(seed=seed)
                        rng = env.np_random
                        env.data.qpos[3:7] = env._tilt_quat(
                            rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02)
                        )
                        env.data.qpos[0:2] = rng.uniform(-0.01, 0.01, size=2)
                        for joint_index, (low, high) in enumerate(env.joint_ranges):
                            qpos_id = env.qpos_ids[joint_index]
                            env.data.qpos[qpos_id] = np.clip(
                                env.data.qpos[qpos_id] + rng.uniform(-0.01, 0.01),
                                low + 0.005,
                                high - 0.005,
                            )
                        env.data.qvel[:] = rng.normal(0.0, 0.01, size=env.model.nv)
                        # Apply held-out mass, contact and slope after the seeded reset.
                        distribution_bounds = cfg["randomization"][
                            "episode_mass_redistribution_scale"
                        ]
                        redistribution = rng.uniform(
                            distribution_bounds[0],
                            distribution_bounds[1],
                            size=env.model.nbody,
                        )
                        redistribution[0] = 1.0
                        redistributed_mass = env.base_mass * redistribution
                        target_mass = float(env.base_mass.sum() * mass_scale)
                        redistributed_mass *= target_mass / redistributed_mass.sum()
                        env.model.body_mass[:] = redistributed_mass
                        link_ratio = np.divide(
                            redistributed_mass,
                            env.base_mass,
                            out=np.ones_like(env.base_mass),
                            where=env.base_mass > 0,
                        )
                        env.model.body_inertia[:] = env.base_inertia * link_ratio[:, None]
                        geom = env.floor_id
                        env.model.geom_friction[geom, 0] = surface["friction"]
                        slope_x, slope_y = surface["slope_x_deg"], surface["slope_y_deg"]
                        env.model.opt.gravity[:] = [
                            9.81 * math.sin(math.radians(slope_x)),
                            -9.81 * math.sin(math.radians(slope_y)),
                            -9.81
                            * math.cos(math.radians(slope_x))
                            * math.cos(math.radians(slope_y)),
                        ]
                        env.model.geom_solref[geom, 0] = math.sqrt(surface["compliance_m"] / 9.81)
                        mujoco.mj_setConst(env.model, env.data)
                        mujoco.mj_forward(env.model, env.data)
                        positive_body_masses = env.model.body_mass[env.model.body_mass > 0]
                        obs = env._obs()
                        total = 0.0
                        fell = False
                        steps = 0
                        min_upright = 1.0
                        while steps < env.max_steps:
                            if controller == "ppo":
                                action, _ = policy.predict(obs, deterministic=True)
                            else:
                                action = np.zeros(env.model.nu, dtype=np.float32)
                            obs, reward, terminated, truncated, info = env.step(action)
                            total += reward
                            steps += 1
                            min_upright = min(min_upright, info["upright"])
                            if writer is not None and controller == "ppo":
                                writer.append_data(env.render())
                            if terminated:
                                fell = True
                                break
                            if truncated:
                                break
                        rows.append(
                            {
                                "controller_type": controller,
                                "surface": name,
                                "episode": episode,
                                "seed": seed,
                                "mass_scale": mass_scale,
                                "mass_kg": float(env.model.body_mass.sum()),
                                "mass_distribution_cv": float(
                                    np.std(positive_body_masses)
                                    / max(np.mean(positive_body_masses), 1e-9)
                                ),
                                "steps": steps,
                                "return": total,
                                "fell": fell,
                                "min_upright": min_upright,
                                "surface_parameters": surface,
                            }
                        )
                        print(json.dumps(rows[-1]))
    finally:
        if writer:
            writer.close()
        env.close()
    surface_metrics = {}
    for name, _surface in conditions:
        surface_metrics[name] = {}
        for mass_scale in a.mass_scales:
            surface_metrics[name][str(mass_scale)] = {}
            for controller in ("neutral_hold", "ppo"):
                samples = [
                    row
                    for row in rows
                    if row["surface"] == name
                    and row["mass_scale"] == mass_scale
                    and row["controller_type"] == controller
                ]
                surface_metrics[name][str(mass_scale)][controller] = {
                    "episodes": len(samples),
                    "no_fall_fraction": float(np.mean([not row["fell"] for row in samples])),
                    "mean_survival_seconds": float(
                        np.mean([row["steps"] for row in samples]) * env.dt
                    ),
                }
    report = {
        "controller": str(a.model.resolve()),
        "robot_config": str(config_path.resolve()),
        "robot_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "mass_status": cfg["mass_authority"],
        "mass_cases_are_synthetic": True,
        "mass_scales_tested": a.mass_scales,
        "episodes": rows,
        "per_surface": surface_metrics,
        "controller_summary": {
            controller: {
                "episodes": sum(row["controller_type"] == controller for row in rows),
                "no_fall_fraction": float(
                    np.mean(
                        [not row["fell"] for row in rows if row["controller_type"] == controller]
                    )
                ),
                "mean_survival_seconds": float(
                    np.mean([row["steps"] for row in rows if row["controller_type"] == controller])
                    * env.dt
                ),
            }
            for controller in ("neutral_hold", "ppo")
        },
        "video": str(a.output.resolve()) if a.output else None,
    }
    path = (
        a.output.with_suffix(".evaluation.json")
        if a.output
        else a.model.with_suffix(".evaluation.json")
    )
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("REPORT=" + str(path.resolve()))


if __name__ == "__main__":
    main()
