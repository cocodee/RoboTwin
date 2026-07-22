#!/usr/bin/env python3
"""Validate RJ2506 RoboTwin gripper-frame transforms."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import transforms3d as t3d
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "assets" / "embodiments" / "RJ2506" / "config.yml"
TASK_CONFIG = ROOT / "task_config" / "demo_rj2506.yml"
EMBODIMENT_CONFIG = ROOT / "task_config" / "_embodiment_config.yml"
ANGLE_TOLERANCE_DEG = 0.05


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def validate_delta_matrix(config: dict) -> None:
    delta = np.array(config["delta_matrix"], dtype=float)
    if delta.shape != (2, 3, 3):
        raise AssertionError(f"delta_matrix shape must be (2, 3, 3), got {delta.shape}")

    for idx, matrix in enumerate(delta):
        det = np.linalg.det(matrix)
        orth_err = np.linalg.norm(matrix.T @ matrix - np.eye(3))
        print(f"matrix[{idx}] det={det:.8f} orth_err={orth_err:.3e}")
        if not np.isclose(det, 1.0, atol=1e-6):
            raise AssertionError(f"delta_matrix[{idx}] determinant is {det}")
        if orth_err > 1e-6:
            raise AssertionError(f"delta_matrix[{idx}] is not orthonormal: {orth_err}")


def build_env():
    from envs.stack_blocks_two import stack_blocks_two

    with open(TASK_CONFIG, encoding="utf-8") as f:
        args = yaml.safe_load(f)
    with open(EMBODIMENT_CONFIG, encoding="utf-8") as f:
        embodiment_config = yaml.safe_load(f)

    robot_file = embodiment_config[args["embodiment"][0]]["file_path"]
    with open(os.path.join(robot_file, "config.yml"), encoding="utf-8") as f:
        robot_config = yaml.safe_load(f)

    args.update(
        {
            "task_name": "stack_blocks_two",
            "left_robot_file": robot_file,
            "right_robot_file": robot_file,
            "dual_arm_embodied": True,
            "left_embodiment_config": robot_config,
            "right_embodiment_config": robot_config,
            "now_ep_num": 0,
            "seed": 13,
            "save_data": False,
        }
    )

    env = stack_blocks_two()
    env.setup_demo(**args)
    return env


def validate_runtime_frame(env) -> None:
    for side in ("left", "right"):
        raw_ee = (env.robot.left_ee if side == "left" else env.robot.right_ee).child_link
        base = env.robot.left_entity.find_link_by_name(f"{side}_hand_base").get_entity_pose()
        tcp = env.robot.left_entity.find_link_by_name(f"{side}_hand_tcp").get_entity_pose()
        gripper_pose = np.array(
            env.robot.get_left_ee_pose() if side == "left" else env.robot.get_right_ee_pose()
        )

        physical = np.asarray(tcp.p) - np.asarray(base.p)
        gripper_x = t3d.quaternions.quat2mat(gripper_pose[-4:])[:, 0]
        endlink = env.robot._trans_from_gripper_to_endlink(gripper_pose, arm_tag=side)

        raw_local = t3d.quaternions.quat2mat(raw_ee.get_entity_pose().q).T @ (
            physical / np.linalg.norm(physical)
        )
        angle = angle_deg(gripper_x, physical)
        roundtrip_pos_err = np.linalg.norm(np.asarray(endlink.p) - np.asarray(tcp.p))
        roundtrip_rot_err = angle_deg(
            t3d.quaternions.quat2mat(endlink.q)[:, 0],
            t3d.quaternions.quat2mat(tcp.q)[:, 0],
        )
        print(
            f"{side}: angle={angle:.6f} deg "
            f"roundtrip_pos_err={roundtrip_pos_err:.3e} "
            f"roundtrip_rot_x_err={roundtrip_rot_err:.6f} deg "
            f"raw_local={raw_local.tolist()}"
        )

        if angle > ANGLE_TOLERANCE_DEG:
            raise AssertionError(f"{side} gripper +X is not aligned with TCP direction: {angle}")
        if roundtrip_pos_err > 1e-9:
            raise AssertionError(f"{side} gripper/endlink roundtrip position error: {roundtrip_pos_err}")
        if roundtrip_rot_err > ANGLE_TOLERANCE_DEG:
            raise AssertionError(f"{side} gripper/endlink roundtrip rotation error: {roundtrip_rot_err}")


def main() -> None:
    with open(CONFIG, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_delta_matrix(config)
    validate_runtime_frame(build_env())
    print("RJ2506 gripper frame calibration: PASS")


if __name__ == "__main__":
    main()
