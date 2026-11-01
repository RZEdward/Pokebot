"""Audit a Hitmonlee MJCF model and report its mechanics and reset contacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco

from hitmonlee_sim.env import LEG_JOINTS, MODEL_PATH, PlanarBipedBalanceEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="MJCF file to inspect")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"--model does not exist or is not a file: {args.model}")
    return args


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str | None:
    result = mujoco.mj_id2name(model, object_type, object_id)
    return str(result) if result is not None else None


def inspect_model(model_path: Path) -> dict[str, Any]:
    model_path = model_path.resolve()
    with PlanarBipedBalanceEnv(model_path=model_path) as env:
        model = env.model
        data = env.data
        _, reset_info = env.reset(seed=0)

        joints: list[dict[str, Any]] = []
        required_joints_present = True
        all_joints_actuated = True
        for joint_name in LEG_JOINTS:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                required_joints_present = False
                all_joints_actuated = False
                joints.append({"name": joint_name, "present": False, "actuator": None})
                continue

            actuator_ids = [
                actuator_id
                for actuator_id in range(model.nu)
                if int(model.actuator_trnid[actuator_id, 0]) == joint_id
            ]
            all_joints_actuated &= len(actuator_ids) == 1
            actuator_id = actuator_ids[0] if actuator_ids else -1
            limited = bool(model.jnt_limited[joint_id])
            joint_row: dict[str, Any] = {
                "name": joint_name,
                "present": True,
                "type": mujoco.mjtJoint(model.jnt_type[joint_id]).name,
                "limited": limited,
                "range_rad": model.jnt_range[joint_id].astype(float).tolist() if limited else None,
                "damping": float(model.dof_damping[model.jnt_dofadr[joint_id]]),
                "actuator": None,
            }
            if actuator_id >= 0:
                joint_row["actuator"] = {
                    "name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id),
                    "type": mujoco.mjtTrn(model.actuator_trntype[actuator_id]).name,
                    "control_range": model.actuator_ctrlrange[actuator_id].astype(float).tolist()
                    if model.actuator_ctrllimited[actuator_id]
                    else None,
                    "force_limited": bool(model.actuator_forcelimited[actuator_id]),
                    "force_range": model.actuator_forcerange[actuator_id].astype(float).tolist()
                    if model.actuator_forcelimited[actuator_id]
                    else None,
                    "position_gain": float(model.actuator_gainprm[actuator_id, 0]),
                    "velocity_gain": float(-model.actuator_biasprm[actuator_id, 2]),
                }
            joints.append(joint_row)

        contact_pairs: list[dict[str, str | None]] = []
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            contact_pairs.append(
                {
                    "geom1": _name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
                    "geom2": _name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
                }
            )
        left_foot_contact = any(
            {pair["geom1"], pair["geom2"]} == {"floor", "left_foot_geom"} for pair in contact_pairs
        )
        right_foot_contact = any(
            {pair["geom1"], pair["geom2"]} == {"floor", "right_foot_geom"} for pair in contact_pairs
        )
        cameras = [
            _name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id) for camera_id in range(model.ncam)
        ]
        sensor_names = [
            _name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
            for sensor_id in range(model.nsensor)
        ]
        total_mass = float(model.body_mass[1:].sum())
        checks = {
            "required_leg_joints_present": required_joints_present,
            "each_leg_joint_has_one_actuator": all_joints_actuated,
            "positive_total_mass": total_mass > 0.0,
            "both_feet_contact_floor_at_reset": left_foot_contact and right_foot_contact,
            "head_camera_present": "head_camera" in cameras,
            "positive_physics_timestep": float(model.opt.timestep) > 0.0,
        }

        return {
            "model_path": str(model_path),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "mujoco_version": mujoco.__version__,
            "compiled_dimensions": {
                "bodies": int(model.nbody - 1),
                "joints": int(model.njnt),
                "positions": int(model.nq),
                "velocities": int(model.nv),
                "actuators": int(model.nu),
                "sensors": int(model.nsensor),
            },
            "total_mass_kg": total_mass,
            "physics": {
                "timestep_seconds": float(model.opt.timestep),
                "gravity_m_s2": model.opt.gravity.astype(float).tolist(),
                "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
            },
            "joints": joints,
            "cameras": cameras,
            "sensors": sensor_names,
            "reset_state": reset_info,
            "reset_contact_pairs": contact_pairs,
            "checks": checks,
            "valid": all(checks.values()),
            "fidelity_note": "Mechanics values are provisional; see docs/architecture/provisional-model.md.",
        }


def main() -> None:
    args = _parse_args()
    report = inspect_model(args.model)
    serialized = json.dumps(report, indent=2)
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
        print(f"Model audit written: {output_path}")
    print(serialized)
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
