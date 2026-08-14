import argparse
import importlib
import os
import sys

sys.path.append("./")

import numpy as np
import transforms3d as t3d
import yaml

from envs import CONFIGS_PATH
from envs.utils import ArmTag


def load_task(task_name):
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


def rotate_delta_matrix(robot_config, roll):
    if roll is None:
        return
    rot = t3d.euler.euler2mat(float(roll), 0.0, 0.0, axes="sxyz")
    robot_config["delta_matrix"] = [
        (np.asarray(matrix, dtype=float) @ rot).tolist()
        for matrix in robot_config["delta_matrix"]
    ]


def load_args(task_name, task_config, angle, delta_roll, gripper_bias):
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    embodiment = args["embodiment"]
    robot_file = embodiment_types[embodiment[0]]["file_path"]
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        robot_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    robot_config["rotate_lim"] = [float(angle), float(angle)]
    if gripper_bias is not None:
        robot_config["gripper_bias"] = float(gripper_bias)
    rotate_delta_matrix(robot_config, delta_roll)
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = robot_config
    args["right_embodiment_config"] = robot_config
    args["embodiment_name"] = str(embodiment[0])
    args["task_config"] = task_config
    args["save_path"] = os.path.join(args["save_path"], task_name, task_config)
    args["need_plan"] = True
    args["save_data"] = False
    args["render_freq"] = 0
    return args


def hammer_z(env):
    return float(env.hammer.get_pose().p[2])


def run_one(args, angle, delta_roll):
    task_args = load_args(args.task_name, args.task_config, angle, delta_roll, args.gripper_bias)
    env = load_task(args.task_name)
    try:
        env.setup_demo(now_ep_num=0, seed=args.seed, **task_args)
        block_pose = env.block.get_functional_point(0, "pose").p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
        z0 = hammer_z(env)
        result = {
            "angle": float(angle),
            "angle_deg": float(np.degrees(angle)),
            "delta_roll": None if delta_roll is None else float(delta_roll),
            "delta_roll_deg": None if delta_roll is None else float(np.degrees(delta_roll)),
            "gripper_bias": args.gripper_bias,
            "arm": str(arm_tag),
            "z0": z0,
            "grasp_ok": False,
            "lift_ok": False,
            "z_after_grasp": None,
            "z_after_lift": None,
            "lift_delta": None,
            "error": None,
        }

        grasp_actions = env.grasp_actor(
            env.hammer,
            arm_tag=arm_tag,
            pre_grasp_dis=args.pre_grasp_dis,
            grasp_dis=args.grasp_dis,
        )
        result["grasp_ok"] = bool(env.move(grasp_actions, save_freq=1))
        result["z_after_grasp"] = hammer_z(env)
        if env.plan_success:
            result["lift_ok"] = bool(env.move(env.move_by_displacement(arm_tag, z=args.lift_z, move_axis="arm"), save_freq=1))
            result["z_after_lift"] = hammer_z(env)
            result["lift_delta"] = result["z_after_lift"] - z0
        return result
    except Exception as exc:
        return {
            "angle": float(angle),
            "angle_deg": float(np.degrees(angle)),
            "delta_roll": None if delta_roll is None else float(delta_roll),
            "delta_roll_deg": None if delta_roll is None else float(np.degrees(delta_roll)),
            "gripper_bias": args.gripper_bias,
            "arm": None,
            "z0": None,
            "grasp_ok": False,
            "lift_ok": False,
            "z_after_grasp": None,
            "z_after_lift": None,
            "lift_delta": None,
            "error": str(exc),
        }
    finally:
        try:
            env.close_env()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default="beat_block_hammer")
    parser.add_argument("--task-config", default="demo_rj2506")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--angles", default="0,0.25,0.5,0.75,1.0,1.25,1.57079633")
    parser.add_argument("--pre-grasp-dis", type=float, default=0.12)
    parser.add_argument("--grasp-dis", type=float, default=0.04)
    parser.add_argument("--lift-z", type=float, default=0.07)
    parser.add_argument("--delta-roll", type=float, default=None)
    parser.add_argument("--gripper-bias", type=float, default=None)
    args = parser.parse_args()

    angles = [float(item) for item in args.angles.split(",") if item.strip()]
    print("angle_rad,angle_deg,delta_roll_rad,delta_roll_deg,gripper_bias,arm,grasp_ok,lift_ok,z0,z_after_grasp,z_after_lift,lift_delta,error")
    for angle in angles:
        result = run_one(args, angle, args.delta_roll)
        print(
            f"{result['angle']:.8f},"
            f"{result['angle_deg']:.2f},"
            f"{result['delta_roll']},"
            f"{result['delta_roll_deg']},"
            f"{result['gripper_bias']},"
            f"{result['arm']},"
            f"{result['grasp_ok']},"
            f"{result['lift_ok']},"
            f"{result['z0']},"
            f"{result['z_after_grasp']},"
            f"{result['z_after_lift']},"
            f"{result['lift_delta']},"
            f"{result['error']}"
        )


if __name__ == "__main__":
    main()
