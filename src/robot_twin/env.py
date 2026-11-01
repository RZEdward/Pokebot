"""Robot-neutral MuJoCo stand-balance environment for articulated humanoid MJCFs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from robot_twin.model_validation import validate_model

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("robot_config.json")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class HumanoidBalanceEnv(gym.Env[np.ndarray, np.ndarray]):
    """Learn standing balance from proprioception with seeded dynamics variation.

    This task uses a position-target action interface backed by saturated finite-gain
    MuJoCo actuators. It does not claim to model a selected motor/drive train.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        config_path: Path = CONFIG_PATH,
        render_mode: str | None = None,
        episode_seconds: float | None = None,
        randomize: bool = True,
    ) -> None:
        super().__init__()
        self.config_path = config_path.resolve()
        self.config = load_config(config_path)
        configured_model = Path(self.config["model_path"])
        run_local_model = self.config_path.parent / configured_model
        self.model_path = (
            run_local_model.resolve()
            if run_local_model.is_file()
            else (ROOT / configured_model).resolve()
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.render_mode = render_mode
        self.randomize = randomize
        self.randomization_intensity = 1.0
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.validation = validate_model(self.model, self.config)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * self.config["control_decimation"]
        self.max_steps = int((episode_seconds or self.config["episode_seconds"]) / self.dt)
        self.joint_names = [n for group in self.config["joint_groups"].values() for n in group]
        if self.model.nu != len(self.joint_names):
            raise ValueError(
                f"Model actuators ({self.model.nu}) != configured joints ({len(self.joint_names)})"
            )
        self.qpos_ids: list[int] = []
        self.qvel_ids: list[int] = []
        self.joint_ranges: list[np.ndarray] = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Configured joint missing from MJCF: {name}")
            self.qpos_ids.append(int(self.model.jnt_qposadr[jid]))
            self.qvel_ids.append(int(self.model.jnt_dofadr[jid]))
            self.joint_ranges.append(self.model.jnt_range[jid].copy())
        self.actuator_targets = np.asarray(self.model.actuator_ctrlrange, dtype=np.float64)
        # Zero-angle neutral posture is not generally the midpoint of asymmetric joint limits.
        self.neutral_targets = np.clip(
            np.zeros(self.model.nu), self.actuator_targets[:, 0], self.actuator_targets[:, 1]
        )
        self.target_half_ranges = (self.actuator_targets[:, 1] - self.actuator_targets[:, 0]) / 2
        self.action_spans = self.target_half_ranges * 0.35
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        # root height, body-frame gravity, global root velocity, joint position/velocity,
        # previous action. No idealized vision/audio observations are fed to policy.
        dim = 1 + 3 + 6 + 3 * self.model.nu
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        self.base_mass = self.model.body_mass.copy()
        self.base_inertia = self.model.body_inertia.copy()
        self.base_friction = self.model.geom_friction.copy()
        self.base_geom_solref = self.model.geom_solref.copy()
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.steps = 0
        self.last_action = np.zeros(self.model.nu)
        self.last_randomization: dict[str, float] = {}
        self._renderer = None
        self._viewer = None

    def set_randomization_intensity(self, intensity: float) -> None:
        """Set curriculum progress in [0, 1]; zero is the nominal design case."""
        if not np.isfinite(intensity):
            raise ValueError("Randomization intensity must be finite")
        self.randomization_intensity = float(np.clip(intensity, 0.0, 1.0))

    def _sample_range(
        self, bounds: list[float], baseline: float, size: int | tuple[int, ...] | None = None
    ):
        scale = self.randomization_intensity
        low = baseline + (bounds[0] - baseline) * scale
        high = baseline + (bounds[1] - baseline) * scale
        if size is None:
            return self.np_random.uniform(low, high)
        return self.np_random.uniform(low, high, size=size)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        rng = self.np_random
        randomization = self.config["randomization"]
        if self.randomize:
            nominal_surface = self.config["surfaces"]["nominal"]
            mass_scale = float(self._sample_range(randomization["mass_scale"], 1.0))
            friction = float(
                self._sample_range(randomization["friction"], nominal_surface["friction"])
            )
            slope_x, slope_y = self._sample_range(randomization["slope_deg"], 0.0, size=2)
            redistribute = self._sample_range(
                randomization["episode_mass_redistribution_scale"],
                1.0,
                size=self.model.nbody,
            )
        else:
            mass_scale, friction, slope_x, slope_y = (
                1.0,
                self.config["surfaces"]["nominal"]["friction"],
                0.0,
                0.0,
            )
            redistribute = np.ones(self.model.nbody)
        redistribute[0] = 1.0
        redistributed_mass = self.base_mass * redistribute
        total_ratio = mass_scale * self.base_mass.sum() / max(redistributed_mass.sum(), 1e-9)
        self.model.body_mass[:] = redistributed_mass * total_ratio
        link_ratio = np.divide(
            self.model.body_mass,
            self.base_mass,
            out=np.ones_like(self.base_mass),
            where=self.base_mass > 0,
        )
        self.model.body_inertia[:] = self.base_inertia * link_ratio[:, None]
        self.model.geom_friction[:] = self.base_friction
        self.model.geom_solref[:] = self.base_geom_solref
        self.model.geom_friction[self.floor_id, 0] = friction
        compliance = (
            float(
                self._sample_range(
                    randomization["compliance_m"],
                    self.config["surfaces"]["nominal"]["compliance_m"],
                )
            )
            if self.randomize
            else self.config["surfaces"]["nominal"]["compliance_m"]
        )
        self.model.geom_solref[self.floor_id, 0] = math.sqrt(compliance / 9.81)
        self.model.opt.gravity[:] = [
            9.81 * math.sin(math.radians(slope_x)),
            -9.81 * math.sin(math.radians(slope_y)),
            -9.81 * math.cos(math.radians(slope_x)) * math.cos(math.radians(slope_y)),
        ]
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[:3] = [0, 0, self.config["base_height_m"]]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        if self.randomize:
            self.data.qpos[3:7] = self._tilt_quat(
                rng.uniform(-0.055, 0.055) * self.randomization_intensity,
                rng.uniform(-0.055, 0.055) * self.randomization_intensity,
            )
            self.data.qpos[0:2] = rng.uniform(-0.015, 0.015, size=2) * self.randomization_intensity
            perturbation = (
                rng.uniform(-0.025, 0.025, size=self.model.nu) * self.randomization_intensity
            )
            for index, (low, high) in enumerate(self.joint_ranges):
                margin = min(0.01, 0.02 * (high - low))
                qpos_id = self.qpos_ids[index]
                self.data.qpos[qpos_id] = np.clip(
                    self.data.qpos[qpos_id] + perturbation[index], low + margin, high - margin
                )
            self.data.qvel[:] = (
                rng.normal(0, 0.015, size=self.model.nv) * self.randomization_intensity
            )
        self.data.ctrl[:] = self.neutral_targets
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.last_action[:] = 0
        self.last_randomization = {
            "mass_scale": mass_scale,
            "friction": friction,
            "slope_x_deg": float(slope_x),
            "slope_y_deg": float(slope_y),
            "compliance_m": compliance,
            "randomization_intensity": self.randomization_intensity if self.randomize else 0.0,
        }
        return self._obs(), self._info(0.0)

    @staticmethod
    def _tilt_quat(roll: float, pitch: float) -> np.ndarray:
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        return np.asarray([cr * cp, sr * cp, cr * sp, -sr * sp])

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        if a.shape != (self.model.nu,) or not np.all(np.isfinite(a)):
            raise ValueError(f"Expected {self.model.nu} finite normalized actions")
        self.data.ctrl[:] = self.neutral_targets + self.action_spans * a
        for _ in range(self.config["control_decimation"]):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        quat = self.data.qpos[3:7]
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, quat)
        gravity_body = rot.reshape(3, 3).T @ np.asarray([0, 0, -1.0])
        height = float(self.data.qpos[2])
        upright = float(np.clip(-gravity_body[2], -1, 1))
        xy_speed = float(np.linalg.norm(self.data.qvel[:2]))
        action_cost = float(np.mean(a * a))
        action_smooth = float(np.mean((a - self.last_action) ** 2))
        reward = (
            1.0
            + 2.0 * upright
            + 1.2 * np.exp(-(((height - self.config["base_height_m"]) / 0.20) ** 2))
        )
        reward += 0.8 * np.exp(-((np.linalg.norm(self.data.qpos[:2]) / 0.25) ** 2))
        reward -= 0.06 * action_cost + 0.04 * action_smooth + 0.08 * min(xy_speed, 5.0)
        self.last_action[:] = a
        fallen = (
            height < 0.42
            or upright < 0.35
            or abs(self.data.qpos[0]) > 0.8
            or abs(self.data.qpos[1]) > 0.8
        )
        if fallen:
            reward -= 2.0
        truncated = self.steps >= self.max_steps
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), bool(fallen), bool(truncated), self._info(float(reward))

    def _obs(self) -> np.ndarray:
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, self.data.qpos[3:7])
        gravity_body = rot.reshape(3, 3).T @ np.asarray([0, 0, -1.0])
        return np.concatenate(
            (
                [self.data.qpos[2]],
                gravity_body,
                self.data.qvel[:6],
                self.data.qpos[self.qpos_ids],
                self.data.qvel[self.qvel_ids],
                self.last_action,
            )
        ).astype(np.float32)

    def _info(self, reward: float) -> dict[str, Any]:
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, self.data.qpos[3:7])
        upright = float(rot.reshape(3, 3)[2, 2])
        return {
            "reward": reward,
            "root_position": self.data.qpos[:3].astype(float).tolist(),
            "upright": upright,
            "simulation_time": float(self.data.time),
            "episode_step": self.steps,
            "joint_angles": self.data.qpos[self.qpos_ids].astype(float).tolist(),
            "randomization": self.last_randomization.copy(),
            "fall": bool(self.data.qpos[2] < 0.42 or upright < 0.35),
        }

    def render(self, camera_name: str = "overview"):
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer

                self._viewer = viewer.launch_passive(self.model, self.data)
            if self._viewer.is_running():
                self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=720, width=960)
            self._renderer.update_scene(self.data, camera=camera_name)
            return self._renderer.render()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
