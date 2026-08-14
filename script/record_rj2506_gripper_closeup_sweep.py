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

    def embodiment_config(path):
        with open(os.path.join(path, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = embodiment_config(robot_file)
    args["right_embodiment_config"] = embodiment_config(robot_file)
    args["embodiment_name"] = str(args["embodiment"][0])
    args["task_config"] = task_config
    args["save_path"] = os.path.join(args["save_path"], task_name, task_config)
    args["need_plan"] = False
    args["save_data"] = False
    args["render_freq"] = 0
    return args


def look_at_camera(scene, name, position, target, width=960, height=720):
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - position
    forward /= np.linalg.norm(forward)
    left = np.cross([0, 0, 1.0], forward)
    if np.linalg.norm(left) < 1e-6:
        left = np.array([1.0, 0, 0])
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    up /= np.linalg.norm(up)
    mat = np.eye(4)
    mat[:3, :3] = np.stack([forward, left, up], axis=1)
    mat[:3, 3] = position
    camera = scene.add_camera(name=name, width=width, height=height, fovy=np.deg2rad(28), near=0.03, far=10)
    camera.entity.set_pose(sapien.Pose(mat))
    return camera


def camera_rgb(camera):
    camera.take_picture()
    rgba = camera.get_picture("Color")
    return (rgba[:, :, :3] * 255).clip(0, 255).astype("uint8")


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


def link_pos(entity, name):
    return np.asarray(entity.find_link_by_name(name).get_entity_pose().p, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default="beat_block_hammer")
    parser.add_argument("--task-config", default="demo_rj2506")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument("--output", default="diagnostics/rj2506_right_gripper_closeup_sweep.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=20)
    args = parser.parse_args()

    task_args = load_args(args.task_name, args.task_config)
    module = importlib.import_module(f"envs.{args.task_name}")
    env = getattr(module, args.task_name)()
    env.setup_demo(now_ep_num=0, seed=args.seed, **task_args)

    entity = env.robot.right_entity if args.side == "right" else env.robot.left_entity
    prefix = args.side
    values = np.concatenate([np.linspace(0.0, 1.0, 41), np.linspace(1.0, 0.0, 41)])
    frames = []

    # Start from open to choose a stable camera centered on the two fingers.
    env.robot.set_gripper(1.0, args.side, gripper_eps=0)
    for _ in range(args.settle_steps):
        env.scene.step()
    f1 = link_pos(entity, f"{prefix}_hand_finger1")
    f2 = link_pos(entity, f"{prefix}_hand_finger2")
    mid = (f1 + f2) / 2
    cam = look_at_camera(
        env.scene,
        f"{args.side}_gripper_closeup_camera",
        mid + np.array([0.10, -0.22, 0.16]),
        mid,
    )

    summary = {
        "side": args.side,
        "gripper_scale": task_args["left_embodiment_config"]["gripper_scale"],
        "finger_distance_samples": [],
        "hammer_scaled_extents_estimate_m": [0.0282, 0.1525, 0.1115],
        "handle_local_width_estimate_m": "about 0.024 on one thin axis, about 0.036-0.044 on the other local cross axis",
    }

    for value in values:
        env.robot.set_gripper(float(value), args.side, gripper_eps=0)
        for _ in range(args.settle_steps):
            env.scene.step()
        f1 = link_pos(entity, f"{prefix}_hand_finger1")
        f2 = link_pos(entity, f"{prefix}_hand_finger2")
        distance = float(np.linalg.norm(f1 - f2))
        summary["finger_distance_samples"].append({"value": float(value), "finger_link_origin_distance": distance})
        env.scene.update_render()
        frame = camera_rgb(cam)
        frames.append(
            annotate(
                frame,
                (
                    f"{args.side} gripper closeup | normalized={value:.2f}\n"
                    f"finger link origin distance={distance * 1000:.1f} mm | gripper_scale=[0,25] mm"
                ),
            )
        )

    env.close_env()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=args.fps, macro_block_size=1)
    with open(output.with_suffix(".json"), "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)
    print(f"Saved video: {output}")
    print(f"Saved summary: {output.with_suffix('.json')}")
    print(summary["finger_distance_samples"][0], summary["finger_distance_samples"][40])


if __name__ == "__main__":
    main()
