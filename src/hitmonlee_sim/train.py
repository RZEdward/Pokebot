"""Train a PPO standing-balance skill and publish live run telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import tempfile
import time
import warnings
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from hitmonlee_sim.env import MODEL_PATH, PlanarBipedBalanceEnv


def _write_json_atomic(
    destination: Path, payload: dict[str, Any], *, indent: int | None = None
) -> bool:
    """Publish JSON atomically, tolerating short Windows reader locks."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=".telemetry-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(json.dumps(payload, indent=indent))

        for attempt in range(8):
            try:
                os.replace(temporary_path, destination)
                temporary_path = None
                return True
            except PermissionError:
                if attempt == 7:
                    warnings.warn(
                        f"Skipped telemetry update because {destination.name} stayed locked.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return False
                time.sleep(0.001 * (2**attempt))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return False


class TelemetryCallback(BaseCallback):
    """Atomically publish compact state/reward telemetry for the local dashboard."""

    def __init__(self, run_dir: Path, every_steps: int = 8, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.run_dir = run_dir
        self.every_steps = max(1, every_steps)
        self.recent_rewards: list[float] = []

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.recent_rewards.append(float(info["episode"]["r"]))
                self.recent_rewards = self.recent_rewards[-80:]
        if self.n_calls % self.every_steps == 0 and infos:
            info = infos[0]
            payload = {
                "step": int(self.num_timesteps),
                "simulation_time": float(info.get("simulation_time", 0.0)),
                "reward": float(info.get("reward", 0.0)),
                "root_x": float(info.get("root_x", 0.0)),
                "root_z": float(info.get("root_z", 0.0)),
                "root_pitch": float(info.get("root_pitch", 0.0)),
                "joint_angles": info.get("joint_angles", [0.0] * 6),
                "joint_velocities": info.get("joint_velocities", [0.0] * 6),
                "episode_step": int(info.get("episode_step", 0)),
                "episode_rewards": self.recent_rewards,
                "episode_reward_mean": float(np.mean(self.recent_rewards))
                if self.recent_rewards
                else None,
                "fall": bool(info.get("fall", False)),
            }
            destination = self.run_dir / "telemetry.json"
            _write_json_atomic(destination, payload)
        return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Robot MJCF path; defaults to packaged model",
    )
    parser.add_argument(
        "--live", action="store_true", help="Show the MuJoCo viewer during training"
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--n-steps",
        type=int,
        default=256,
        help="PPO rollout length; smaller updates the monitor more often",
    )
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be a positive integer.")
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    if args.n_steps < 2:
        parser.error("--n-steps must be at least 2.")
    batch_size = min(64, args.n_steps)
    if args.n_steps % batch_size:
        parser.error("--n-steps must be divisible by its PPO minibatch size (min(64, n_steps)).")
    if args.eval_every <= 0:
        parser.error("--eval-every must be a positive integer.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("--learning-rate must be finite and greater than zero.")
    if not args.model_path.is_file():
        parser.error(f"--model-path does not exist or is not a file: {args.model_path}")
    return args


def main() -> None:
    args = _parse_args()
    model_path = args.model_path.resolve()
    run_dir = args.run_dir or Path("runs") / datetime.now(UTC).strftime("balance-%Y%m%d-%H%M%S")
    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        raise FileExistsError(f"Run directory must be empty; refusing to overwrite: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def make_train_env() -> Monitor:
        env = PlanarBipedBalanceEnv(
            render_mode="human" if args.live else None, model_path=model_path
        )
        return Monitor(env, filename=str(run_dir / "progress"))

    train_env = DummyVecEnv([make_train_env])
    eval_env = make_vec_env(
        lambda: Monitor(PlanarBipedBalanceEnv(randomize_reset=True, model_path=model_path)),
        n_envs=1,
        seed=args.seed + 10_000,
    )
    eval_frequency = max(args.eval_every, args.n_steps)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "evaluation"),
        eval_freq=eval_frequency,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callback = CallbackList([TelemetryCallback(run_dir), eval_callback])

    ppo_config = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": min(64, args.n_steps),
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.003,
        "policy_architecture": {"pi": [64, 64], "vf": [64, 64]},
    }
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "seed": args.seed,
        "requested_steps": args.steps,
        "algorithm": "PPO",
        "device_requested": args.device,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gymnasium": version("gymnasium"),
            "mujoco": version("mujoco"),
            "numpy": version("numpy"),
            "stable_baselines3": version("stable-baselines3"),
            "torch": torch.__version__,
        },
        "model_path": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "source_sha256": {
            "train.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "env.py": hashlib.sha256(Path(__file__).with_name("env.py").read_bytes()).hexdigest(),
        },
        "task": "planar_biped_proprioceptive_standing_balance",
        "environment": {
            "episode_seconds": 8.0,
            "frame_skip": 4,
            "physics_timestep_seconds": 0.005,
            "training_reset_randomization": True,
            "evaluation_seed": args.seed + 10_000,
        },
        "ppo_config": ppo_config,
        "evaluation": {
            "frequency_steps": eval_frequency,
            "episodes_per_evaluation": 5,
            "deterministic": True,
        },
        "render_live": args.live,
        "policy_action": "normalized desired joint positions through finite-gain saturated position actuators",
    }
    print(f"Run directory: {run_dir.resolve()}")
    print(f"Model: {model_path}")
    print(f"Device: requested={args.device}, torch_cuda={torch.cuda.is_available()}")
    model = PPO(
        "MlpPolicy",
        train_env,
        seed=args.seed,
        **{
            key: ppo_config[key]
            for key in (
                "learning_rate",
                "n_steps",
                "batch_size",
                "n_epochs",
                "gamma",
                "gae_lambda",
                "clip_range",
                "ent_coef",
            )
        },
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        device=args.device,
        verbose=1,
    )
    metadata["device_resolved"] = str(model.device)
    _write_json_atomic(run_dir / "run.json", metadata, indent=2)
    try:
        model.learn(total_timesteps=args.steps, callback=callback, progress_bar=False)
        model.save(run_dir / "final_model")
        metadata["status"] = "completed"
        metadata["completed_utc"] = datetime.now(UTC).isoformat()
        metadata["actual_timesteps"] = int(model.num_timesteps)
        evaluation_archive = run_dir / "evaluation" / "evaluations.npz"
        if evaluation_archive.is_file():
            with np.load(evaluation_archive) as evaluations:
                timesteps = evaluations["timesteps"]
                rewards = evaluations["results"].mean(axis=1)
            if len(timesteps):
                best_index = int(np.argmax(rewards))
                metadata["evaluation"].update(
                    {
                        "best_checkpoint_timestep": int(timesteps[best_index]),
                        "best_mean_reward": float(rewards[best_index]),
                        "results_path": str(evaluation_archive.resolve()),
                        "best_checkpoint_path": str(
                            (run_dir / "best" / "best_model.zip").resolve()
                        ),
                    }
                )
        _write_json_atomic(run_dir / "run.json", metadata, indent=2)
        print(f"Saved policy: {(run_dir / 'final_model.zip').resolve()}")
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["completed_utc"] = datetime.now(UTC).isoformat()
        metadata["actual_timesteps"] = int(model.num_timesteps)
        metadata["error"] = {"type": type(error).__name__, "message": str(error)}
        _write_json_atomic(run_dir / "run.json", metadata, indent=2)
        raise
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
