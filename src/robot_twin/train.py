"""Train and checkpoint a PPO proprioceptive standing controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from robot_twin.env import CONFIG_PATH, ROOT, HumanoidBalanceEnv, load_config


def write_json_atomic(path: Path, payload: dict) -> None:
    """Keep readers of the local live monitor from seeing a partial JSON document."""
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".state-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class RunTelemetry(BaseCallback):
    def __init__(self, run_dir: Path):
        super().__init__()
        self.run_dir = run_dir
        self.returns = []
        self.last_checkpoint_step = 0

    def _on_training_start(self) -> None:
        self.training_env.envs[0].unwrapped.set_randomization_intensity(0.0)

    def _on_step(self):
        env = self.training_env.envs[0].unwrapped
        scale = min(1.0, self.num_timesteps / max(1, int(self.model._total_timesteps * 0.6)))
        env.set_randomization_intensity(scale)
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.returns.append(float(info["episode"]["r"]))
        if self.n_calls % 50 == 0:
            info = (self.locals.get("infos") or [{}])[0]
            payload = {
                "step": int(self.num_timesteps),
                "reward": float(info.get("reward", 0)),
                "position": info.get("root_position", [0, 0, 0]),
                "upright": info.get("upright", 0),
                "episode_step": info.get("episode_step", 0),
                "joint_angles": info.get("joint_angles", []),
                "randomization": info.get("randomization", {}),
                "episode_returns": self.returns,
            }
            write_json_atomic(self.run_dir / "telemetry.json", payload)
        if self.num_timesteps - self.last_checkpoint_step >= 10_000:
            checkpoint = self.run_dir / "policy-checkpoint.zip"
            temporary = self.run_dir / f".policy-checkpoint-{self.num_timesteps}.zip"
            self.model.save(temporary)
            os.replace(temporary, checkpoint)
            write_json_atomic(
                self.run_dir / "checkpoint.json",
                {"step": int(self.num_timesteps), "path": checkpoint.name},
            )
            self.last_checkpoint_step = self.num_timesteps
        if self.n_calls % 500 == 0:
            frame = env.render()
            if frame is not None:
                temporary = self.run_dir / "latest.tmp.png"
                imageio.imwrite(temporary, frame)
                temporary.replace(self.run_dir / "latest.png")
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    ap.add_argument("--gamma", type=float, default=0.995, help="PPO discount factor")
    a = ap.parse_args()
    if not 0.0 < a.gamma <= 1.0:
        ap.error("--gamma must be in (0, 1]")
    run = a.run_dir or ROOT / "runs" / datetime.now(UTC).strftime("humanoid-balance-%Y%m%d-%H%M%S")
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    robot = load_config()
    model_source = ROOT / robot["model_path"]
    shutil.copy2(model_source, run / "robot_model.xml")
    run_config = dict(robot)
    run_config["model_path"] = "robot_model.xml"
    write_json_atomic(run / "robot_config_snapshot.json", run_config)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    def make_env():
        return Monitor(HumanoidBalanceEnv(render_mode="human" if a.live else "rgb_array"))

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
    model = PPO("MlpPolicy", env, seed=a.seed, device=a.device, verbose=1, **ppo)
    meta = {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "seed": a.seed,
        "requested_steps": a.steps,
        "algorithm": "PPO",
        "device": str(model.device),
        "task": "proprioceptive_stand",
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
        model.learn(total_timesteps=a.steps, callback=RunTelemetry(run), progress_bar=False)
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
