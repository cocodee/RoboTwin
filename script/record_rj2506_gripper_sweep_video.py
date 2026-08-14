import argparse
import importlib
import os
import sys
from pathlib import Path

sys.path.append("./")

import imageio.v2 as imageio
import numpy as np
import sapien.core as sapien
import yaml
from PIL import Image, ImageDraw, ImageFont

from envs import CONFIGS_PATH


def load_args(task_name, task_config):
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    robot_file = embodiment_types[args["embodiment"][0]]["file_path"]

    def get_embodiment_config(path):
        with open(os.path.join(path, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = get_embodiment_config(robot_file)
    args["right_embodiment_config"] = get_embodiment_config(robot_file)
    args["embodiment_name"] = str(args["embodiment"][0])
    args["task_config"] = task_config
    args["save_path"] = os.path.join(args["save_path"], task_name, task_config)
    args["need_plan"] = False
    args["save_data"] = False
    args["render_freq"] = 0
    return args


def load_task(task_name):
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


def camera_rgb(camera):
    camera.take_picture()
    rgba = camera.get_picture("Color")
    return (rgba[:, :, :3] * 255).clip(0, 255).astype("uint8")


def resize(rgb, size):
    return np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR))


def annotate(rgb, text):
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    y = 12
    for line in text.split("\n"):
        draw.text((12, y), line, fill=(0, 0, 0), font=font, stroke_width=3, stroke_fill=(255, 255, 255))
        draw.text((12, y), line, fill=(255, 240, 60), font=font)
        y += 28
    return np.asarray(image)


def build_frame(env, gripper_value):
    env.scene.update_render()
    observer = resize(env.cameras.get_observer_rgb(), (640, 480))
    world1 = resize(camera_rgb(env.cameras.world_camera1), (640, 480))
    frame = np.concatenate([observer, world1], axis=1)
    return annotate(
        frame,
        (
            f"RJ2506 gripper sweep | normalized={gripper_value:.2f}\n"
            "0.00 = min/closed, 1.00 = max/open | left: observer, right: world_camera1"
        ),
    )


def gripper_joint_state(robot, side):
    gripper = robot.left_gripper if side == "left" else robot.right_gripper
    return {
        joint[0].get_name(): {
            "drive_target": float(joint[0].drive_target[0]),
            "limit": [float(x) for x in joint[0].get_limits()[0]],
        }
        for joint in gripper
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default="beat_block_hammer")
    parser.add_argument("--task-config", default="demo_rj2506")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="diagnostics/rj2506_gripper_scale_sweep.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=20)
    args = parser.parse_args()

    task_args = load_args(args.task_name, args.task_config)
    env = load_task(args.task_name)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env.setup_demo(now_ep_num=0, seed=args.seed, **task_args)
    values = np.concatenate(
        [
            np.linspace(0.0, 1.0, 41),
            np.linspace(1.0, 0.0, 41),
        ]
    )

    frames = []
    summary = {
        "gripper_scale": task_args["left_embodiment_config"]["gripper_scale"],
        "left_initial": gripper_joint_state(env.robot, "left"),
        "right_initial": gripper_joint_state(env.robot, "right"),
        "output": str(output_path),
    }

    for value in values:
        env.robot.set_gripper(float(value), "left", gripper_eps=0)
        env.robot.set_gripper(float(value), "right", gripper_eps=0)
        for _ in range(args.settle_steps):
            env.scene.step()
        frames.append(build_frame(env, float(value)))

    summary["left_final"] = gripper_joint_state(env.robot, "left")
    summary["right_final"] = gripper_joint_state(env.robot, "right")
    env.close_env()

    imageio.mimsave(output_path, frames, fps=args.fps, macro_block_size=1)
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)

    print(f"Saved video: {output_path}")
    print(f"Saved summary: {output_path.with_suffix('.json')}")
    print(summary)


if __name__ == "__main__":
    main()
