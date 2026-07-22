#!/usr/bin/env python3
import math
from pathlib import Path

import numpy as np
import sapien.core as sapien
import torch
import yaml
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml


ROOT = Path(__file__).resolve().parents[1]
EMBODIMENT = ROOT / "assets" / "embodiments" / "RJ2506"
URDF = EMBODIMENT / "urdf" / "relpath_RJ2506_fixed.urdf"
CONFIG = EMBODIMENT / "config.yml"
POSITION_TOLERANCE_M = 5e-3
ROTATION_TOLERANCE_DEG = 3.0


def quat_angle_deg(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = abs(float(np.dot(a, b)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def make_scene_robot():
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    scene = engine.create_scene()
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(str(URDF))
    return scene, robot


def get_pose(link):
    if hasattr(link, "get_entity_pose"):
        return link.get_entity_pose()
    return link.get_pose()


def main():
    with open(CONFIG, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    scene, sapien_robot = make_scene_robot()
    active_names = [joint.get_name() for joint in sapien_robot.get_active_joints()]

    samples = [
        ("zero", {}),
        (
            "home",
            {
                name: value
                for side in range(2)
                for name, value in zip(config["arm_joints_name"][side], config["homestate"][side])
            },
        ),
        (
            "offset",
            {
                "left_arm_joint0": -0.2,
                "left_arm_joint1": 0.5,
                "left_arm_joint2": 0.1,
                "left_arm_joint3": 0.8,
                "left_arm_joint4": -0.1,
                "left_arm_joint5": 0.2,
                "right_arm_joint0": 0.2,
                "right_arm_joint1": 0.5,
                "right_arm_joint2": -0.1,
                "right_arm_joint3": 0.8,
                "right_arm_joint4": 0.1,
                "right_arm_joint5": -0.2,
            },
        ),
    ]

    tensor_args = TensorDeviceType()
    curobo_models = {}
    for side in ("left", "right"):
        yml = load_yaml(str(EMBODIMENT / f"curobo_{side}.yml"))
        robot_cfg = RobotConfig.from_dict(yml["robot_cfg"], tensor_args)
        curobo_models[side] = (robot_cfg.kinematics.cspace.joint_names, CudaRobotModel(robot_cfg.kinematics))

    max_pos_err = 0.0
    max_rot_err = 0.0
    for sample_name, sample in samples:
        qpos = np.zeros(len(active_names), dtype=np.float32)
        for joint_name, value in sample.items():
            qpos[active_names.index(joint_name)] = value
        sapien_robot.set_qpos(qpos)

        for side in ("left", "right"):
            joint_names, model = curobo_models[side]
            q = torch.tensor(
                [float(qpos[active_names.index(name)]) for name in joint_names],
                **tensor_args.as_torch_dict(),
            ).view(1, -1)
            state = model.get_state(q)
            curobo_p = state.ee_position[0].detach().cpu().numpy()
            curobo_q = state.ee_quaternion[0].detach().cpu().numpy()
            sapien_pose = get_pose(sapien_robot.find_link_by_name(f"{side}_hand_tcp"))
            pos_err = float(np.linalg.norm(np.asarray(sapien_pose.p) - curobo_p))
            rot_err = quat_angle_deg(sapien_pose.q, curobo_q)
            max_pos_err = max(max_pos_err, pos_err)
            max_rot_err = max(max_rot_err, rot_err)
            print(
                f"{sample_name:>6} {side:>5}: "
                f"position_error_m={pos_err:.6f} rotation_error_deg={rot_err:.6f}"
            )

    print(f"max_position_error_m={max_pos_err:.6f}")
    print(f"max_rotation_error_deg={max_rot_err:.6f}")
    if max_pos_err > POSITION_TOLERANCE_M or max_rot_err > ROTATION_TOLERANCE_DEG:
        raise SystemExit(1)
    print("RJ2506 EE frame alignment: PASS")


if __name__ == "__main__":
    main()
