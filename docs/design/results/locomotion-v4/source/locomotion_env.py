"""Command-conditioned velocity tracking task for articulated humanoid models."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from robot_twin.controllers import ankle_upright_pd_action
from robot_twin.env import CONFIG_PATH, HumanoidBalanceEnv, load_config
from robot_twin.locomotion import phase_gait_action


class CommandedLocomotionEnv(gym.Env[np.ndarray, np.ndarray]):
    """Track planar body-frame velocity commands with proprioceptive PPO residuals.

    This is a research locomotion task on a flat, provisional MuJoCo model. The
    ankle-PD controller provides bounded attitude stabilization; PPO supplies joint
    target offsets across the whole body. It is not a hardware control law.
    """

    metadata: ClassVar[dict[str, Any]] = HumanoidBalanceEnv.metadata

    def __init__(
        self,
        config_path=CONFIG_PATH,
        render_mode: str | None = None,
        randomize: bool = True,
        reference_mode: str = "ankle_pd",
    ) -> None:
        super().__init__()
        self.config = load_config(config_path)
        self.task_config = self.config["locomotion_training"]
        if reference_mode not in {"ankle_pd", "phase_gait_plus_ankle_pd"}:
            raise ValueError(f"Unsupported locomotion reference mode: {reference_mode}")
        self.reference_mode = reference_mode
        self.base = HumanoidBalanceEnv(
            config_path=config_path,
            render_mode=render_mode,
            randomize=randomize,
            control_mode="direct",
        )
        self.action_space = self.base.action_space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.base.observation_space.shape[0] + 5,),
            dtype=np.float32,
        )
        self.command = np.zeros(3, dtype=np.float32)
        self.last_policy_action = np.zeros(self.base.model.nu, dtype=np.float32)
        self.gait_phase_offset = 0.0

    @property
    def model(self):
        return self.base.model

    @property
    def data(self):
        return self.base.data

    @property
    def dt(self) -> float:
        return self.base.dt

    @property
    def max_steps(self) -> int:
        return self.base.max_steps

    @property
    def model_path(self):
        return self.base.model_path

    def set_randomization_intensity(self, intensity: float) -> None:
        self.base.set_randomization_intensity(intensity)

    def _observation(self) -> np.ndarray:
        phase = (
            2.0 * np.pi * self.config["locomotion_reference"]["frequency_hz"] * self.data.time
            + self.gait_phase_offset
        )
        gait_phase = np.asarray([np.sin(phase), np.cos(phase)], dtype=np.float32)
        return np.concatenate((self.base._obs(), self.command, gait_phase)).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.base.reset(seed=seed, options=options)
        requested_command = (options or {}).get("command")
        if requested_command is not None:
            command = requested_command
        elif self.base.randomize:
            index = int(self.np_random.integers(0, len(self.task_config["commands"])))
            command = self.task_config["commands"][index]
        else:
            command = self.task_config["commands"][0]
        self.command[:] = command
        self.gait_phase_offset = (
            float(self.np_random.uniform(0.0, 2.0 * np.pi)) if self.base.randomize else 0.0
        )
        self.last_policy_action[:] = 0
        info = self._info(0.0, 0.0)
        info["command_body_m_s"] = self.command.astype(float).tolist()
        info["body_velocity_m_s"] = self._body_velocity().astype(float).tolist()
        info["gait_phase_offset_rad"] = self.gait_phase_offset
        return self._observation(), info

    def _body_velocity(self) -> np.ndarray:
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        matrix = rotation.reshape(3, 3)
        return matrix.T @ self.data.qvel[:3]

    def step(self, action: np.ndarray):
        return self._step(action, residual=True)

    def step_reference(self, target_action: np.ndarray):
        """Advance with an explicit normalized joint-target reference trajectory."""
        return self._step(target_action, residual=False)

    def _step(self, action: np.ndarray, *, residual: bool):
        policy_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if policy_action.shape != (self.model.nu,) or not np.all(np.isfinite(policy_action)):
            raise ValueError(f"Expected {self.model.nu} finite normalized actions")
        baseline = ankle_upright_pd_action(
            self.base, self.config["evaluation_baselines"]["ankle_upright_pd"]
        )
        if self.reference_mode == "phase_gait_plus_ankle_pd":
            command = {
                "vx_m_s": float(self.command[0]),
                "vy_m_s": float(self.command[1]),
                "yaw_rate_rad_s": float(self.command[2]),
            }
            baseline = np.clip(
                phase_gait_action(
                    self.base,
                    command,
                    self.data.time,
                    self.config["locomotion_reference"],
                    self.gait_phase_offset,
                )
                + baseline,
                -1.0,
                1.0,
            )
        combined = (
            np.clip(baseline + self.task_config["residual_scale"] * policy_action, -1.0, 1.0)
            if residual
            else policy_action
        )
        _, _, terminated, truncated, base_info = self.base.step(combined)
        body_velocity = self._body_velocity()
        velocity_error = body_velocity[:2] - self.command[:2]
        tracking = float(
            np.exp(
                -float(np.dot(velocity_error, velocity_error))
                / self.task_config["tracking_sigma_m_s"] ** 2
            )
        )
        height_error = (self.data.qpos[2] - self.config["base_height_m"]) / 0.18
        action_cost = float(np.mean(policy_action**2))
        smooth_cost = float(np.mean((policy_action - self.last_policy_action) ** 2))
        upright = float(base_info["upright"])
        reward = (
            self.task_config["tracking_reward_weight"] * tracking
            + self.task_config["upright_reward_weight"] * upright
            + self.task_config["height_reward_weight"] * np.exp(-(height_error**2))
            - self.task_config["action_cost_weight"] * action_cost
            - self.task_config["smooth_cost_weight"] * smooth_cost
        )
        if terminated:
            reward -= self.task_config["fall_penalty"]
        self.last_policy_action[:] = policy_action
        info = self._info(float(reward), tracking)
        info.update(base_info)
        info["reward"] = float(reward)
        info["command_body_m_s"] = self.command.astype(float).tolist()
        info["body_velocity_m_s"] = body_velocity.astype(float).tolist()
        info["velocity_tracking_score"] = tracking
        info["gait_phase_offset_rad"] = self.gait_phase_offset
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    def _info(self, reward: float, tracking: float) -> dict[str, Any]:
        return {
            "reward": reward,
            "root_position": self.data.qpos[:3].astype(float).tolist(),
            "upright": float(self.base._info(0.0)["upright"]),
            "episode_step": int(self.base.steps),
            "joint_angles": self.data.qpos[self.base.qpos_ids].astype(float).tolist(),
            "randomization": self.base.last_randomization.copy(),
            "velocity_tracking_score": tracking,
        }

    def render(self, camera_name: str = "overview"):
        return self.base.render(camera_name)

    def close(self):
        self.base.close()
