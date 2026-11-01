"""Gymnasium environment for the provisional planar biped."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

MODEL_PATH = Path(__file__).resolve().parent / "models" / "hitmonlee_planar.xml"
LEG_JOINTS = (
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
)
NOMINAL_TARGETS = np.array([0.0, -0.10, 0.0, 0.0, -0.10, 0.0], dtype=np.float64)
ACTION_SPANS = np.array([0.45, 0.45, 0.35, 0.45, 0.45, 0.35], dtype=np.float64)


class PlanarBipedBalanceEnv(gym.Env[np.ndarray, np.ndarray]):
    """Learn a proprioceptive standing-balance skill in a 2D biped.

    Policy actions are normalized desired joint positions. The MJCF position
    actuators apply finite stiffness, damping, limits, and saturation; they are
    not ideal torque sources. This interface is provisional until a physical
    actuator module and its drive mode are measured.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        episode_seconds: float = 8.0,
        frame_skip: int = 4,
        randomize_reset: bool = True,
        model_path: Path = MODEL_PATH,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if not np.isfinite(episode_seconds) or episode_seconds <= 0.0:
            raise ValueError("episode_seconds must be finite and greater than zero")
        if frame_skip <= 0:
            raise ValueError("frame_skip must be a positive integer")
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.randomize_reset = randomize_reset
        self.model_path = Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.max_steps = max(1, round(episode_seconds / (self.model.opt.timestep * frame_skip)))
        self._joint_qpos = {
            name: self._address(mujoco.mjtObj.mjOBJ_JOINT, name, "qpos") for name in LEG_JOINTS
        }
        self._joint_qvel = {
            name: self._address(mujoco.mjtObj.mjOBJ_JOINT, name, "qvel") for name in LEG_JOINTS
        }
        self._root_x = self._address(mujoco.mjtObj.mjOBJ_JOINT, "root_x", "qpos")
        self._root_z = self._address(mujoco.mjtObj.mjOBJ_JOINT, "root_z", "qpos")
        self._root_pitch = self._address(mujoco.mjtObj.mjOBJ_JOINT, "root_pitch", "qpos")
        self._root_x_vel = self._address(mujoco.mjtObj.mjOBJ_JOINT, "root_x", "qvel")
        self._root_z_vel = self._address(mujoco.mjtObj.mjOBJ_JOINT, "root_z", "qvel")
        self._root_pitch_vel = self._address(mujoco.mjtObj.mjOBJ_JOINT, "root_pitch", "qvel")
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(LEG_JOINTS),), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float32)
        self._steps = 0
        self._last_action = np.zeros(len(LEG_JOINTS), dtype=np.float64)
        self._viewer: Any = None
        self._renderer: mujoco.Renderer | None = None

    def _address(self, object_type: mujoco.mjtObj, name: str, field: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Model is missing named {object_type.name}: {name}")
        address_array = self.model.jnt_qposadr if field == "qpos" else self.model.jnt_dofadr
        return int(address_array[object_id])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._steps = 0
        self._last_action.fill(0.0)
        self.data.qpos[self._root_x] = 0.0
        # With the pelvis body at the world origin, the stacked leg geometry
        # puts the soles at z=0 when the root is approximately 0.94 m high.
        self.data.qpos[self._root_z] = 0.94
        self.data.qpos[self._root_pitch] = 0.0
        for name, target in zip(LEG_JOINTS, NOMINAL_TARGETS, strict=True):
            self.data.qpos[self._joint_qpos[name]] = target
        if self.randomize_reset:
            self.data.qpos[self._root_pitch] += self.np_random.uniform(-0.04, 0.04)
            self.data.qpos[self._root_z] += self.np_random.uniform(-0.008, 0.008)
            self.data.qpos[self._root_x] += self.np_random.uniform(-0.025, 0.025)
            for name in LEG_JOINTS:
                self.data.qpos[self._joint_qpos[name]] += self.np_random.uniform(-0.045, 0.045)
            self.data.qvel[self._root_pitch_vel] = self.np_random.uniform(-0.04, 0.04)
            self.data.qvel[self._root_x_vel] = self.np_random.uniform(-0.03, 0.03)
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), self._info(reward=0.0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        step_wall_start = time.perf_counter() if self.render_mode == "human" else None
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action_array.shape}"
            )
        if not np.all(np.isfinite(action_array)):
            raise ValueError("Actions must contain only finite values")
        clipped_action = np.clip(action_array, -1.0, 1.0)
        self.data.ctrl[:] = NOMINAL_TARGETS + ACTION_SPANS * clipped_action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        pitch = float(self.data.qpos[self._root_pitch])
        root_z = float(self.data.qpos[self._root_z])
        root_x = float(self.data.qpos[self._root_x])
        joint_angles = self._joint_angles()
        smoothness = float(np.square(clipped_action - self._last_action).mean())
        control_cost = float(np.square(self.data.actuator_force).mean())
        self._last_action[:] = clipped_action

        upright = np.exp(-((pitch / 0.28) ** 2))
        height = np.exp(-(((root_z - 0.94) / 0.18) ** 2))
        centered = np.exp(-((root_x / 0.35) ** 2))
        joint_cost = float(np.square(joint_angles - NOMINAL_TARGETS).mean())
        reward = float(
            1.0
            + 2.0 * upright
            + 1.2 * height
            + 0.6 * centered
            - 0.08 * joint_cost
            - 0.025 * smoothness
            - 0.000015 * control_cost
        )

        terminated = root_z < 0.57 or abs(pitch) > 1.05 or abs(root_x) > 0.85
        truncated = self._steps >= self.max_steps
        if terminated:
            reward -= 2.0
        if self.render_mode == "human":
            self.render()
            assert step_wall_start is not None
            remaining_wall_time = self.frame_skip * self.model.opt.timestep - (
                time.perf_counter() - step_wall_start
            )
            if remaining_wall_time > 0.0:
                time.sleep(remaining_wall_time)
        info = self._info(reward=reward)
        info["fall"] = bool(terminated)
        info["joint_limit_margin"] = self._minimum_joint_limit_margin()
        return self._observation(), reward, bool(terminated), bool(truncated), info

    def _joint_angles(self) -> np.ndarray:
        return np.asarray([self.data.qpos[self._joint_qpos[name]] for name in LEG_JOINTS])

    def _joint_velocities(self) -> np.ndarray:
        return np.asarray([self.data.qvel[self._joint_qvel[name]] for name in LEG_JOINTS])

    def _observation(self) -> np.ndarray:
        pitch = float(self.data.qpos[self._root_pitch])
        root_velocity = np.array(
            [
                self.data.qvel[self._root_x_vel],
                self.data.qvel[self._root_z_vel],
                self.data.qvel[self._root_pitch_vel],
            ],
            dtype=np.float64,
        )
        observation = np.concatenate(
            (
                np.array(
                    [
                        self.data.qpos[self._root_z],
                        np.sin(pitch),
                        np.cos(pitch),
                        self.data.qpos[self._root_x],
                    ],
                    dtype=np.float64,
                ),
                root_velocity,
                self._joint_angles(),
                self._joint_velocities(),
                self._last_action,
            )
        )
        return observation.astype(np.float32)

    def _minimum_joint_limit_margin(self) -> float:
        margins = []
        for name in LEG_JOINTS:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            low, high = self.model.jnt_range[joint_id]
            angle = self.data.qpos[self._joint_qpos[name]]
            margins.append(min(angle - low, high - angle))
        return float(min(margins))

    def _info(self, reward: float) -> dict[str, Any]:
        pitch = float(self.data.qpos[self._root_pitch])
        root_z = float(self.data.qpos[self._root_z])
        root_x = float(self.data.qpos[self._root_x])
        return {
            "reward": float(reward),
            "root_x": root_x,
            "root_z": root_z,
            "root_pitch": pitch,
            "root_x_velocity": float(self.data.qvel[self._root_x_vel]),
            "root_z_velocity": float(self.data.qvel[self._root_z_vel]),
            "root_pitch_velocity": float(self.data.qvel[self._root_pitch_vel]),
            "joint_angles": self._joint_angles().astype(float).tolist(),
            "joint_velocities": self._joint_velocities().astype(float).tolist(),
            "episode_step": int(self._steps),
            "simulation_time": float(self.data.time),
        }

    def render(self, camera_name: str = "overview") -> np.ndarray | None:
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

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
