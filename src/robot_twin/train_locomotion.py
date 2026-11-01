"""Train a command-conditioned PPO residual for planar humanoid locomotion."""

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
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from robot_twin.env import ROOT, load_config
from robot_twin.experiments import RunTelemetry, write_json_atomic
from robot_twin.locomotion_env import CommandedLocomotionEnv


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Continue from an SB3 locomotion checkpoint")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--reference-mode",
        choices=("ankle_pd", "phase_gait_plus_ankle_pd"),
        default="ankle_pd",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.torch_threads < 1:
        parser.error("--steps and --torch-threads must be positive")
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"Checkpoint not found: {args.resume}")
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    config = load_config()
    model_source = ROOT / config["model_path"]
    shutil.copy2(model_source, run_dir / "robot_model.xml")
    run_config = dict(config)
    run_config["model_path"] = "robot_model.xml"
    write_json_atomic(run_dir / "robot_config_snapshot.json", run_config)
    config_snapshot = run_dir / "robot_config_snapshot.json"
    if args.resume is not None:
        prior_config = args.resume.parent / "robot_config_snapshot.json"
        prior_model = args.resume.parent / "robot_model.xml"
        if not prior_config.is_file() or not prior_model.is_file():
            parser.error(
                "Resume checkpoint must be accompanied by robot model and config snapshots"
            )
        if sha256_file(prior_config) != sha256_file(config_snapshot):
            parser.error("Resume checkpoint config does not match this run's robot config")
        if sha256_file(prior_model) != sha256_file(run_dir / "robot_model.xml"):
            parser.error("Resume checkpoint model does not match this run's robot model")
        prior_manifest = args.resume.parent / "run.json"
        if prior_manifest.is_file():
            prior_mode = json.loads(prior_manifest.read_text(encoding="utf-8")).get(
                "reference_mode", "ankle_pd"
            )
            if prior_mode != args.reference_mode:
                parser.error("Resume checkpoint reference mode does not match this run")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def make_env():
        return Monitor(
            CommandedLocomotionEnv(
                config_path=run_dir / "robot_config_snapshot.json",
                render_mode="rgb_array",
                randomize=True,
                reference_mode=args.reference_mode,
            )
        )

    env = DummyVecEnv([make_env])
    hyperparameters = {
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 128,
        "n_epochs": 5,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.005,
        "policy_kwargs": {
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            "log_std_init": -1.0,
        },
    }
    policy = (
        PPO.load(args.resume, env=env, device=args.device)
        if args.resume is not None
        else PPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            device=args.device,
            verbose=0,
            **hyperparameters,
        )
    )
    starting_timesteps = int(policy.num_timesteps)
    manifest = {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "requested_steps": args.steps,
        "resume_checkpoint": str(args.resume.resolve()) if args.resume is not None else None,
        "resume_source_timesteps": starting_timesteps if args.resume is not None else 0,
        "algorithm": "PPO",
        "device": str(policy.device),
        "torch_threads": args.torch_threads,
        "task": "command_conditioned_velocity_tracking",
        "control_mode": "ankle_pd_plus_whole_body_residual",
        "reference_mode": args.reference_mode,
        "observation_features": "proprioception, planar velocity command, sine/cosine gait phase",
        "robot_config_snapshot": "robot_config_snapshot.json",
        "robot_model": "robot_model.xml",
        "robot_model_sha256": sha256_file(model_source),
        "robot_config_sha256": sha256_file(config_snapshot),
        "randomization_schedule": "0 to 1 linearly by 60% of planned steps; sampled episode values are logged",
        "source_sha256": {
            path.name: sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("locomotion_env.py"),
                Path(__file__).with_name("locomotion.py"),
                Path(__file__).with_name("controllers.py"),
                Path(__file__).with_name("env.py"),
                Path(__file__).with_name("experiments.py"),
            )
        },
        "software": {
            name: version(name)
            for name in ("gymnasium", "mujoco", "numpy", "stable-baselines3", "torch")
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hyperparameters": hyperparameters,
    }
    write_json_atomic(run_dir / "run.json", manifest)
    started = time.monotonic()
    try:
        policy.learn(
            total_timesteps=args.steps,
            callback=CallbackList([RunTelemetry(run_dir)]),
            reset_num_timesteps=args.resume is None,
        )
        policy.save(run_dir / "policy.zip")
        manifest.update(
            {
                "status": "complete",
                "finished_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "actual_steps": int(policy.num_timesteps),
            }
        )
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "finished_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        raise
    finally:
        write_json_atomic(run_dir / "run.json", manifest)
        env.close()
    print(f"Saved locomotion policy and run manifest to {run_dir}")


if __name__ == "__main__":
    main()
