"""Shared atomic run storage and lightweight live-training telemetry."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import imageio.v2 as imageio
from stable_baselines3.common.callbacks import BaseCallback


def replace_with_retry(source: Path, destination: Path) -> None:
    """Retry transient Windows sharing violations from dashboard readers."""
    for attempt in range(5):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.01 * (2**attempt))


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write a complete JSON snapshot before replacing the dashboard's current file."""
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".state-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(Path(temporary), path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class RunTelemetry(BaseCallback):
    """Capture generic environment state and recoverable checkpoints for local runs."""

    def __init__(self, run_dir: Path):
        super().__init__()
        self.run_dir = run_dir
        self.returns: list[float] = []
        self.last_checkpoint_step = 0

    def _on_training_start(self) -> None:
        env = self.training_env.envs[0].unwrapped
        if hasattr(env, "set_randomization_intensity"):
            env.set_randomization_intensity(0.0)

    def _on_step(self) -> bool:
        env = self.training_env.envs[0].unwrapped
        randomization_scale = min(
            1.0,
            self.num_timesteps / max(1, int(self.model._total_timesteps * 0.6)),
        )
        if hasattr(env, "set_randomization_intensity"):
            env.set_randomization_intensity(randomization_scale)
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.returns.append(float(info["episode"]["r"]))
        if self.n_calls % 50 == 0:
            info = (infos or [{}])[0]
            write_json_atomic(
                self.run_dir / "telemetry.json",
                {
                    "step": int(self.num_timesteps),
                    "reward": float(info.get("reward", 0)),
                    "position": info.get("root_position", [0, 0, 0]),
                    "upright": info.get("upright", 0),
                    "episode_step": info.get("episode_step", 0),
                    "joint_angles": info.get("joint_angles", []),
                    "randomization": info.get("randomization", {}),
                    "command_body_m_s": info.get("command_body_m_s"),
                    "body_velocity_m_s": info.get("body_velocity_m_s"),
                    "velocity_tracking_score": info.get("velocity_tracking_score"),
                    "fall": info.get("fall", False),
                    "gait_phase_offset_rad": info.get("gait_phase_offset_rad"),
                    "episode_returns": self.returns,
                },
            )
        if self.num_timesteps - self.last_checkpoint_step >= 10_000:
            checkpoint = self.run_dir / "policy-checkpoint.zip"
            temporary = self.run_dir / f".policy-checkpoint-{self.num_timesteps}.zip"
            self.model.save(temporary)
            replace_with_retry(temporary, checkpoint)
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
                replace_with_retry(temporary, self.run_dir / "latest.png")
        return True
