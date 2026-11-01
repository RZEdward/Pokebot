"""Simple model-based controllers used as inspectable task baselines."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


def ankle_upright_pd_action(env: Any, settings: dict[str, Any]) -> np.ndarray:
    """Return a four-ankle PD correction from body-frame gravity and base angular rate.

    This is a compact reference baseline, not a hardware controller. Gains and joint
    mappings are embodiment configuration values and must be retuned after calibration.
    """
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, env.data.qpos[3:7])
    gravity_body = rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    pitch_target = (
        settings["proportional_gain"] * gravity_body[0]
        - settings["angular_damping"] * env.data.qvel[4]
    )
    roll_target = (
        settings["proportional_gain"] * gravity_body[1]
        - settings["angular_damping"] * env.data.qvel[3]
    )
    joint_index = {name: index for index, name in enumerate(env.joint_names)}
    action = np.zeros(env.model.nu, dtype=np.float64)
    for name in settings["pitch_ankle_joints"]:
        index = joint_index[name]
        action[index] = pitch_target / env.action_spans[index]
    for name in settings["roll_ankle_joints"]:
        index = joint_index[name]
        action[index] = roll_target / env.action_spans[index]
    return np.clip(action, -1.0, 1.0).astype(np.float32)
