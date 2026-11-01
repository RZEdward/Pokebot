"""Cross-check a compiled MJCF against its robot config and CAD reference frames."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


def validate_model(model: mujoco.MjModel, config: dict[str, Any]) -> dict[str, Any]:
    """Fail fast on mismatched actuators, mass case, kinematic frames, or sensors."""
    expected_joints = config["joint_frames"]
    names = [row["name"] for row in expected_joints]
    grouped_names = [name for group in config["joint_groups"].values() for name in group]
    if len(names) != len(set(names)) or set(names) != set(grouped_names):
        raise ValueError("joint_frames must enumerate each configured joint exactly once")
    if model.nu != len(names):
        raise ValueError(f"MJCF has {model.nu} actuators but config has {len(names)} joints")
    if not np.isclose(model.opt.timestep, config["timestep_s"], atol=1e-9):
        raise ValueError("MJCF and robot config physics timesteps differ")

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frame_errors = []
    for row in expected_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, row["name"])
        if jid < 0:
            raise ValueError(f"Configured joint is missing from MJCF: {row['name']}")
        anchor_error = float(
            np.linalg.norm(data.xanchor[jid] - np.asarray(row["origin_m"], dtype=float))
        )
        axis_error = float(
            np.linalg.norm(data.xaxis[jid] - np.asarray(row["axis_robot"], dtype=float))
        )
        range_error = float(
            np.max(np.abs(model.jnt_range[jid] - np.asarray(row["range_rad"], dtype=float)))
        )
        frame_errors.append(
            {
                "name": row["name"],
                "anchor_error_m": anchor_error,
                "axis_error": axis_error,
                "range_error_rad": range_error,
            }
        )
    max_anchor = max(row["anchor_error_m"] for row in frame_errors)
    max_axis = max(row["axis_error"] for row in frame_errors)
    max_range = max(row["range_error_rad"] for row in frame_errors)
    if max_anchor > 0.002 or max_axis > 1e-4 or max_range > 0.001:
        raise ValueError(
            f"MJCF/CAD joint mismatch: anchor={max_anchor:.6f}m, "
            f"axis={max_axis:.6f}, range={max_range:.6f}rad"
        )

    actual_mass = float(mujoco.mj_getTotalmass(model))
    expected_mass = float(config["nominal_mass_kg"])
    if abs(actual_mass - expected_mass) > 0.005:
        raise ValueError(
            f"Compiled model mass {actual_mass:.4f}kg != configured {expected_mass:.4f}kg"
        )
    actual_com = data.subtree_com[1].copy()
    expected_com = np.asarray(config["reference_mass_com_m"], dtype=float)
    com_error = float(np.linalg.norm(actual_com - expected_com))
    if com_error > 0.02:
        raise ValueError(
            f"Compiled neutral COM differs from placeholder ledger by {com_error:.4f}m"
        )

    camera_names = config["sensors"]["cameras"]
    for name in camera_names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
            raise ValueError(f"Configured camera is missing from MJCF: {name}")
    for name in config["sensors"]["microphones"] + [config["sensors"]["speaker"]]:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) < 0:
            raise ValueError(f"Configured acoustic site is missing from MJCF: {name}")

    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "body_count": int(model.nbody),
        "camera_count": int(model.ncam),
        "nominal_mass_kg": actual_mass,
        "mass_authority": config["mass_authority"],
        "neutral_com_m": actual_com.tolist(),
        "reference_com_m": expected_com.tolist(),
        "com_delta_m": com_error,
        "max_joint_anchor_error_m": max_anchor,
        "max_joint_axis_error": max_axis,
        "max_joint_range_error_rad": max_range,
        "joint_frames": frame_errors,
    }
