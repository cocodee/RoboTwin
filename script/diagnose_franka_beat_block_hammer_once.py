import importlib
import os
import sys

sys.path.append("./")

import numpy as np
import yaml

from envs import CONFIGS_PATH
from envs.utils import ArmTag


def get_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def load_args(task_name, task_config):
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    embodiment = args["embodiment"]
    robot_file = embodiment_types[embodiment[0]]["file_path"]
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = get_embodiment_config(robot_file)
    args["right_embodiment_config"] = get_embodiment_config(robot_file)
    args["embodiment_name"] = str(embodiment[0])
    args["task_config"] = task_config
    args["save_path"] = os.path.join(args["save_path"], task_name, task_config)
    args["need_plan"] = True
    args["save_data"] = False
    args["render_freq"] = 0
    return args


def print_state(env, label):
    hammer_pose = env.hammer.get_pose().p
    hammer_target = env.hammer.get_functional_point(0, "pose").p
    block_target = env.block.get_functional_point(1, "pose").p
    print(
        label,
        {
            "plan_success": bool(env.plan_success),
            "hammer_xyz": np.round(hammer_pose, 4).tolist(),
            "hammer_target_xyz": np.round(hammer_target, 4).tolist(),
            "block_target_xyz": np.round(block_target, 4).tolist(),
            "xyz_error": np.round(hammer_target - block_target, 4).tolist(),
            "hammer_block_contact": bool(env.check_actors_contact(env.hammer.get_name(), env.block.get_name())),
            "check_success": bool(env.check_success()),
        },
        flush=True,
    )


def main():
    task_name = "beat_block_hammer"
    task_config = "demo_franka_1ep"
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    module = importlib.import_module(f"envs.{task_name}")
    env = getattr(module, task_name)()
    args = load_args(task_name, task_config)

    try:
        print(f"setup seed={seed}", flush=True)
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        print_state(env, "initial")

        block_pose = env.block.get_functional_point(0, "pose").p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
        print(f"arm_tag={arm_tag}", flush=True)

        print("stage=grasp", flush=True)
        grasp_dis = 0.04 if args["embodiment_name"] == "RJ2506" else 0.0
        grasp_ok = env.move(env.grasp_actor(env.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=grasp_dis))
        print(f"grasp_ok={grasp_ok}", flush=True)
        print_state(env, "after_grasp")

        if env.plan_success:
            print("stage=lift", flush=True)
            lift_ok = env.move(env.move_by_displacement(arm_tag, z=0.07, move_axis="arm"))
            print(f"lift_ok={lift_ok}", flush=True)
            print_state(env, "after_lift")

        if env.plan_success:
            print("stage=world_align_place", flush=True)
            hammer_target = env.hammer.get_functional_point(0, "pose").p
            block_target = env.block.get_functional_point(1, "pose").p
            xy_delta = block_target[:2] - hammer_target[:2]
            align_ok = env.move(env.move_by_displacement(arm_tag, x=xy_delta[0], y=xy_delta[1], move_axis="world"))
            print(f"align_ok={align_ok}", flush=True)
            print_state(env, "after_world_xy_align")

        if env.plan_success:
            hammer_target = env.hammer.get_functional_point(0, "pose").p
            block_target = env.block.get_functional_point(1, "pose").p
            z_delta = block_target[2] - hammer_target[2] - 0.015
            lower_ok = env.move(env.move_by_displacement(arm_tag, z=z_delta, move_axis="world"))
            print(f"lower_ok={lower_ok}", flush=True)
            print_state(env, "after_world_lower")
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
