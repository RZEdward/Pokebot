"""Train and checkpoint a PPO proprioceptive standing controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from robot_twin.env import CONFIG_PATH, ROOT, HumanoidBalanceEnv, load_config
from robot_twin.experiments import RunTelemetry, write_json_atomic


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--resume", type=Path, help="Resume from a saved Stable-Baselines3 checkpoint")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    ap.add_argument("--torch-threads", type=int, default=2, help="Limit PyTorch CPU threads")
    ap.add_argument("--gamma", type=float, default=0.995, help="PPO discount factor")
    ap.add_argument(
        "--control-mode",
        choices=("direct", "ankle_pd_residual"),
        default="ankle_pd_residual",
        help="Learn direct targets or bounded residuals over the ankle-PD stabilizer",
    )
    ap.add_argument(
        "--residual-scale",
        type=float,
        default=None,
        help="Override residual amplitude from the robot config for this run",
    )
    a = ap.parse_args()
    if a.torch_threads < 1:
        ap.error("--torch-threads must be positive")
    if a.residual_scale is not None and not 0.0 <= a.residual_scale <= 1.0:
        ap.error("--residual-scale must be in [0, 1]")
    if not 0.0 < a.gamma <= 1.0:
        ap.error("--gamma must be in (0, 1]")
    if a.resume and not a.resume.is_file():
        ap.error(f"Resume checkpoint does not exist: {a.resume}")
    if a.resume:
        source_manifest = a.resume.parent / "run.json"
        if source_manifest.is_file():
            source_mode = json.loads(source_manifest.read_text(encoding="utf-8")).get(
                "control_mode", "direct"
            )
            if source_mode != a.control_mode:
                ap.error(f"Resume control mode {source_mode!r} does not match {a.control_mode!r}")
    torch.set_num_threads(a.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    run = a.run_dir or ROOT / "runs" / datetime.now(UTC).strftime("humanoid-balance-%Y%m%d-%H%M%S")
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    robot = load_config()
    if a.residual_scale is not None:
        robot["evaluation_baselines"]["ankle_upright_pd"]["residual_scale"] = a.residual_scale
    model_source = ROOT / robot["model_path"]
    shutil.copy2(model_source, run / "robot_model.xml")
    run_config = dict(robot)
    run_config["model_path"] = "robot_model.xml"
    write_json_atomic(run / "robot_config_snapshot.json", run_config)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    def make_env():
        return Monitor(
            HumanoidBalanceEnv(
                render_mode="human" if a.live else "rgb_array",
                config_path=run / "robot_config_snapshot.json",
                control_mode=a.control_mode,
            )
        )

    env = DummyVecEnv([make_env])
    ppo = {
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 5,
        "gamma": a.gamma,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.005,
        "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "log_std_init": -0.7},
    }
    model = (
        PPO.load(a.resume, env=env, device=a.device)
        if a.resume
        else PPO("MlpPolicy", env, seed=a.seed, device=a.device, verbose=1, **ppo)
    )
    starting_timesteps = int(model.num_timesteps)
    meta = {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "seed": a.seed,
        "requested_steps": a.steps,
        "algorithm": "PPO",
        "device": str(model.device),
        "torch_threads": a.torch_threads,
        "task": "proprioceptive_stand",
        "control_mode": a.control_mode,
        "residual_scale_override": a.residual_scale,
        "resumed_from": str(a.resume.resolve()) if a.resume else None,
        "resume_source_timesteps": starting_timesteps if a.resume else 0,
        "model_path": "src/robot_twin/models/humanoid_28dof.xml",
        "robot_config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "nominal_total_mass_kg": 25.0,
        "mass_authority": "PLACEHOLDER_DESIGN_ALLOCATION",
        "reference_com_m": robot["reference_mass_com_m"],
        "joint_groups": robot["joint_groups"],
        "domain_randomization": robot["randomization"],
        "surface_profiles": robot["surfaces"],
        "mjcf_sha256": hashlib.sha256(
            (ROOT / "src/robot_twin/models/humanoid_28dof.xml").read_bytes()
        ).hexdigest(),
        "robot_config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "environment_sha256": hashlib.sha256(
            Path(__file__).with_name("env.py").read_bytes()
        ).hexdigest(),
        "action_interface": (
            "normalized target position around zero-radian neutral, 0.35 joint half-range, "
            "finite gain/torque saturation"
        ),
        "domain_randomization_schedule": "linear nominal-to-full-ranges over first 60% of PPO steps",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mujoco": version("mujoco"),
        "gymnasium": version("gymnasium"),
        "stable_baselines3": version("stable-baselines3"),
        "train_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_config_snapshot_sha256": hashlib.sha256(
            (run / "robot_config_snapshot.json").read_bytes()
        ).hexdigest(),
        "run_model_sha256": hashlib.sha256((run / "robot_model.xml").read_bytes()).hexdigest(),
        "ppo": {k: v for k, v in ppo.items() if k != "policy_kwargs"},
    }
    write_json_atomic(run / "run.json", meta)
    started = time.perf_counter()
    try:
        model.learn(
            total_timesteps=a.steps,
            callback=RunTelemetry(run),
            progress_bar=False,
            reset_num_timesteps=True,
        )
        model.save(run / "policy")
        meta.update(
            status="completed",
            actual_steps=int(model.num_timesteps),
            wall_seconds=time.perf_counter() - started,
            completed_utc=datetime.now(UTC).isoformat(),
        )
    except BaseException as e:
        meta.update(
            status="failed", error=f"{type(e).__name__}: {e}", actual_steps=int(model.num_timesteps)
        )
        raise
    finally:
        write_json_atomic(run / "run.json", meta)
        env.close()
    print(f"RUN_DIR={run}")


if __name__ == "__main__":
    main()
