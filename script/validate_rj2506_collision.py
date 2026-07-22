#!/usr/bin/env python3
"""Validate RJ2506 CuRobo collision configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from curobo.types.robot import JointState

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASK_CONFIG = ROOT / "task_config" / "demo_rj2506.yml"
EMBODIMENT_CONFIG = ROOT / "task_config" / "_embodiment_config.yml"


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


def validate_yml(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    kinematics = config["robot_cfg"]["kinematics"]
    ignore = kinematics["self_collision_ignore"]
    planner = config["planner"]
    if ignore.get("body_link1") != ["body_link2"]:
        raise AssertionError(f"{path} missing body_link1/body_link2 self-collision ignore")
    if planner.get("skip_warmup") is not True:
        raise AssertionError(f"{path} must set planner.skip_warmup=true for RJ2506")


def validate_start_state(env, side: str) -> None:
    planner = env.robot.left_planner if side == "left" else env.robot.right_planner
    entity = env.robot.left_entity if side == "left" else env.robot.right_entity
    qpos = entity.get_qpos()
    joint_indices = [planner.all_joints.index(name) for name in planner.active_joints_name]
    joint_angles = [round(float(qpos[index]), 5) for index in joint_indices]
    start = JointState.from_position(
        torch.tensor(joint_angles, device="cuda").reshape(1, -1),
        joint_names=planner.active_joints_name,
    )
    valid, status = planner.motion_gen.check_start_state(start)
    print(f"{side}: start_state_valid={valid} status={status} joints={joint_angles}")
    if not valid:
        raise AssertionError(f"{side} start state invalid: {status}")

    gripper_pose = np.array(
        env.robot.get_left_ee_pose() if side == "left" else env.robot.get_right_ee_pose(),
        dtype=float,
    )
    result = env.robot.left_plan_path(gripper_pose.tolist()) if side == "left" else env.robot.right_plan_path(
        gripper_pose.tolist()
    )
    print(f"{side}: noop_plan_status={result.get('status')}")
    if result.get("status") != "Success":
        raise AssertionError(f"{side} no-op plan failed: {result.get('error', result)}")


def main() -> None:
    validate_yml(str(ROOT / "assets" / "embodiments" / "RJ2506" / "curobo_left.yml"))
    validate_yml(str(ROOT / "assets" / "embodiments" / "RJ2506" / "curobo_right.yml"))
    env = build_env()
    validate_start_state(env, "left")
    validate_start_state(env, "right")
    print("RJ2506 collision configuration: PASS")


if __name__ == "__main__":
    main()
