#!/usr/bin/env python3
"""Validate RJ2506 joint ordering between SAPIEN, RoboTwin config, and CuRobo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

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
    return env, robot_config


def validate_side(env, config: dict, side: str, index: int) -> None:
    planner = env.robot.left_planner if side == "left" else env.robot.right_planner
    active_names = [joint.get_name() for joint in env.robot.left_entity.get_active_joints()]
    config_names = config["arm_joints_name"][index]

    missing = [name for name in config_names if name not in active_names]
    if missing:
        raise AssertionError(f"{side} arm joints missing from SAPIEN active joints: {missing}")

    if planner.active_joints_name != config_names:
        raise AssertionError(
            f"{side} planner active joints do not match config: "
            f"{planner.active_joints_name} != {config_names}"
        )

    with open(planner.yml_path, encoding="utf-8") as f:
        curobo_config = yaml.safe_load(f)
    curobo_names = curobo_config["robot_cfg"]["kinematics"]["cspace"]["joint_names"]
    if curobo_names != config_names:
        raise AssertionError(f"{side} CuRobo cspace order mismatch: {curobo_names} != {config_names}")

    indices = [active_names.index(name) for name in config_names]
    print(f"{side}: joint_order={config_names} active_indices={indices}")


def main() -> None:
    env, config = build_env()
    validate_side(env, config, "left", 0)
    validate_side(env, config, "right", 1)
    print("RJ2506 joint ordering: PASS")


if __name__ == "__main__":
    main()
