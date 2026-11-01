"""Inspectable phase-based locomotion reference for low-speed gait experiments.

The trajectory is a research baseline, not a balance controller or a hardware gait.
Joint names and action scaling come from the selected robot embodiment configuration.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def phase_gait_action(
    env: Any,
    command: dict[str, float],
    simulation_time: float,
    settings: dict[str, Any],
    phase_offset_rad: float = 0.0,
) -> np.ndarray:
    """Map a bounded body-frame velocity command to normalized joint target offsets.

    This simple alternating-leg trajectory makes no claim of ZMP, capture-point, or
    contact-phase feedback. Evaluate it under a harness before using it as a training
    reference; residual learning can be layered over it only after the baseline moves.
    """
    vx = float(
        np.clip(command.get("vx_m_s", 0.0), -settings["max_speed_m_s"], settings["max_speed_m_s"])
    )
    vy = float(
        np.clip(command.get("vy_m_s", 0.0), -settings["max_speed_m_s"], settings["max_speed_m_s"])
    )
    yaw_rate = float(
        np.clip(
            command.get("yaw_rate_rad_s", 0.0),
            -settings["max_yaw_rate_rad_s"],
            settings["max_yaw_rate_rad_s"],
        )
    )
    frequency = float(settings["frequency_hz"])
    phase = 2.0 * np.pi * frequency * simulation_time + phase_offset_rad
    stride_length = min(
        abs(vx) / frequency,
        settings["max_stride_length_m"],
    )
    side_stride = min(abs(vy) / settings["max_speed_m_s"], 1.0)
    turn = yaw_rate / settings["max_yaw_rate_rad_s"]
    ramp = float(np.clip(simulation_time / settings["startup_ramp_s"], 0.0, 1.0))
    blend = ramp * ramp * (3.0 - 2.0 * ramp)
    targets = np.zeros(env.model.nu, dtype=np.float64)
    joint_index = {name: index for index, name in enumerate(env.joint_names)}

    def set_target(name: str, value: float) -> None:
        index = joint_index.get(name)
        if index is not None:
            targets[index] = np.clip(blend * value / max(env.action_spans[index], 1e-8), -1.0, 1.0)

    length_thigh, length_shank = settings["leg_segment_lengths_m"]
    hip_height = settings["hip_to_ground_nominal_m"]
    lift = settings["swing_clearance_m"]
    for side, label in ((1.0, "left"), (-1.0, "right")):
        leg_phase = (phase + (0.0 if side > 0 else np.pi)) % (2.0 * np.pi)
        cycle = leg_phase / (2.0 * np.pi)
        stance_fraction = settings["stance_fraction"]
        if cycle < stance_fraction:
            progress = cycle / stance_fraction
            foot_x = (0.5 - progress) * stride_length
            foot_z_down = hip_height
        else:
            progress = (cycle - stance_fraction) / (1.0 - stance_fraction)
            foot_x = (-0.5 + progress) * stride_length
            foot_z_down = hip_height - lift * np.sin(np.pi * progress)
        foot_x *= np.sign(vx)
        distance = float(np.hypot(foot_x, foot_z_down))
        distance = min(distance, length_thigh + length_shank - 1e-5)
        knee_cos = (distance**2 - length_thigh**2 - length_shank**2) / (
            2.0 * length_thigh * length_shank
        )
        knee_pitch = float(np.arccos(np.clip(knee_cos, -1.0, 1.0)))
        foot_angle = float(np.arctan2(foot_x, foot_z_down))
        leg_offset = float(
            np.arctan2(
                length_shank * np.sin(knee_pitch),
                length_thigh + length_shank * np.cos(knee_pitch),
            )
        )
        hip_pitch = -foot_angle - leg_offset
        ankle_pitch = -hip_pitch - knee_pitch
        # These lateral and turning terms only bias joint pose; they are not a
        # validated side-step or heading controller.
        hip_roll = -side * np.sign(vy) * settings["hip_roll_amplitude_rad"] * side_stride
        hip_yaw = side * turn * settings["hip_yaw_turn_amplitude_rad"]
        set_target(f"{label}_hip_pitch", hip_pitch)
        set_target(f"{label}_knee_pitch", knee_pitch)
        set_target(f"{label}_ankle_pitch", ankle_pitch)
        set_target(f"{label}_hip_roll", hip_roll)
        set_target(f"{label}_hip_yaw", hip_yaw)
        # Counter-swing arms to reduce angular-momentum demand on the trunk.
        set_target(f"{label}_shoulder_pitch", settings["arm_counter_swing"] * hip_pitch)
        swing_blend = max(0.0, np.sin(2.0 * np.pi * progress))
        set_target(
            f"{label}_elbow_pitch",
            settings["elbow_flex_amplitude_rad"] * swing_blend * side_stride,
        )

    set_target("waist_yaw", turn * settings["waist_yaw_turn_amplitude_rad"])
    return targets.astype(np.float32)
