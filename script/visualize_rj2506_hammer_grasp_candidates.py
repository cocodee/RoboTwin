import argparse
import importlib
import json
import os
import sys
from pathlib import Path

sys.path.append("./")

import imageio
import numpy as np
import sapien.core as sapien
import transforms3d as t3d
import yaml
from PIL import Image, ImageDraw, ImageFont

from envs import CONFIGS_PATH
from envs.utils import create_box


FIXED_CONTACT_TO_GRASP = np.array(
    [
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ]
)


def load_task(task_name):
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


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
    if len(embodiment) != 1:
        raise ValueError("This diagnostic expects a single dual-arm embodiment config.")

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
    args["render_freq"] = 0
    args["save_data"] = False
    return args


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def pose_to_list(pose):
    return list(np.asarray(pose.p, dtype=float)) + list(np.asarray(pose.q, dtype=float))


def make_pose_from_matrix(matrix):
    return sapien.Pose(matrix[:3, 3], t3d.quaternions.mat2quat(matrix[:3, :3]))


def raw_pre_grasp_pose(actor, contact_point_id, pre_dis):
    contact_matrix = actor.get_contact_point(contact_point_id, "matrix")
    if contact_matrix is None:
        return None
    grasp_matrix = contact_matrix @ FIXED_CONTACT_TO_GRASP
    grasp_pos = grasp_matrix[:3, 3] + grasp_matrix[:3, :3] @ np.array([-0.12 - pre_dis, 0, 0]).T
    grasp_quat = t3d.quaternions.mat2quat(grasp_matrix[:3, :3])
    return list(grasp_pos) + list(grasp_quat)


def final_grasp_pose(pre_grasp_pose, pre_dis, target_dis):
    pose = np.asarray(pre_grasp_pose, dtype=float).copy()
    direction_mat = t3d.quaternions.quat2mat(pose[-4:])
    pose[:3] += np.array([pre_dis - target_dis, 0, 0]) @ np.linalg.inv(direction_mat)
    return pose.tolist()


def add_marker(env, pos, color, name, size=0.012):
    create_box(
        scene=env,
        pose=sapien.Pose(pos, [1, 0, 0, 0]),
        half_size=(size, size, size),
        color=color,
        is_static=True,
        name=name,
    )


def add_pose_axes(env, pose_list, name, length=0.07, width=0.004):
    pos = np.asarray(pose_list[:3], dtype=float)
    rot = t3d.quaternions.quat2mat(pose_list[-4:])
    colors = [(1, 0.05, 0.05), (0.05, 0.8, 0.05), (0.05, 0.25, 1.0)]
    labels = ["x", "y", "z"]
    for axis_id, (color, label) in enumerate(zip(colors, labels)):
        mat = np.eye(4)
        mat[:3, :3] = rot
        mat[:3, 3] = pos + rot[:, axis_id] * length * 0.5
        half = [width, width, width]
        half[axis_id] = length * 0.5
        create_box(
            scene=env,
            pose=make_pose_from_matrix(mat),
            half_size=tuple(half),
            color=color,
            is_static=True,
            name=f"{name}_{label}",
        )


def add_camera(scene, name, position, target, width=1600, height=900):
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = normalize(target - position)
    world_up = np.array([0, 0, 1.0])
    left = normalize(np.cross(world_up, forward))
    if np.linalg.norm(left) < 1e-6:
        left = np.array([1, 0, 0], dtype=float)
    up = normalize(np.cross(forward, left))
    mat = np.eye(4)
    mat[:3, :3] = np.stack([forward, left, up], axis=1)
    mat[:3, 3] = position
    camera = scene.add_camera(name=name, width=width, height=height, fovy=np.deg2rad(45), near=0.05, far=100)
    camera.entity.set_pose(sapien.Pose(mat))
    return camera


def save_camera(camera, output_path, text_lines):
    camera.take_picture()
    rgba = camera.get_picture("Color")
    image = (rgba * 255).clip(0, 255).astype("uint8")
    imageio.imwrite(output_path, image[:, :, :3])
    annotate_image(output_path, text_lines)


def save_rgb_array(rgb, output_path, text_lines):
    imageio.imwrite(output_path, rgb)
    annotate_image(output_path, text_lines)


def annotate_image(output_path, text_lines):
    output_path = str(output_path)

    img = Image.open(output_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    y = 18
    for line, color in text_lines:
        draw.text((18, y), line, fill=(0, 0, 0), font=font, stroke_width=3, stroke_fill=(255, 255, 255))
        draw.text((18, y), line, fill=color, font=font)
        y += 30
    img.save(output_path)


def as_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, list):
        return [as_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: as_jsonable(v) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default="beat_block_hammer")
    parser.add_argument("--task-config", default="demo_rj2506")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pre-grasp-dis", type=float, default=0.12)
    parser.add_argument("--grasp-dis", type=float, default=0.01)
    parser.add_argument("--output-dir", default="diagnostics")
    args = parser.parse_args()

    task_args = load_args(args.task_name, args.task_config)
    env = load_task(args.task_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env.setup_demo(now_ep_num=0, seed=args.seed, **task_args)
    hammer = env.hammer
    block_pose = env.block.get_functional_point(0, "pose").p
    selected_arm = "left" if block_pose[0] < 0 else "right"

    results = {
        "task_name": args.task_name,
        "task_config": args.task_config,
        "seed": args.seed,
        "selected_arm_from_block": selected_arm,
        "robot_pose_config": task_args["left_embodiment_config"]["robot_pose"],
        "hammer_pose": pose_to_list(hammer.get_pose()),
        "block_functional_pose": pose_to_list(env.block.get_functional_point(0, "pose")),
        "pre_grasp_dis": args.pre_grasp_dis,
        "grasp_dis": args.grasp_dis,
        "arms": {},
    }

    for arm in ["left", "right"]:
        arm_entries = []
        for contact_id, contact_pose in hammer.iter_contact_points("list"):
            pre_pose = raw_pre_grasp_pose(hammer, contact_id, args.pre_grasp_dis)
            if pre_pose is None:
                arm_entries.append({"contact_id": contact_id, "error": "contact_matrix_none"})
                continue
            grasp_pose = final_grasp_pose(pre_pose, args.pre_grasp_dis, args.grasp_dis)
            target_poses = env.robot.create_target_pose_list(pre_pose, contact_pose, arm)
            plan = env.robot.left_plan_multi_path(target_poses) if arm == "left" else env.robot.right_plan_multi_path(target_poses)
            endlink_targets = [
                pose_to_list(env.robot._trans_from_gripper_to_endlink(target_pose, arm_tag=arm))
                for target_pose in target_poses
            ]
            arm_entries.append(
                {
                    "contact_id": contact_id,
                    "contact_pose": contact_pose,
                    "raw_pre_grasp_pose": pre_pose,
                    "grasp_pose": grasp_pose,
                    "target_pose_count": len(target_poses),
                    "target_poses": target_poses,
                    "endlink_targets": endlink_targets,
                    "planner_status": plan.get("status", []),
                    "planner_success_count": sum(1 for status in plan.get("status", []) if status == "Success"),
                }
            )
        results["arms"][arm] = arm_entries

    contact_pos = np.asarray(hammer.get_contact_point(0, "list")[:3], dtype=float)
    add_marker(env, contact_pos, (0.45, 0.22, 0.05), "hammer_contact", size=0.014)
    add_marker(env, env.block.get_functional_point(1, "pose").p, (1, 0, 1), "block_target", size=0.014)

    for arm, color, z_offset in [("left", (0.0, 0.2, 1.0), 0.0), ("right", (1.0, 0.55, 0.0), 0.025)]:
        entry = results["arms"][arm][0]
        pre_pose = entry["raw_pre_grasp_pose"]
        grasp_pose = entry["grasp_pose"]
        add_marker(env, np.asarray(pre_pose[:3]) + [0, 0, z_offset], color, f"{arm}_pre_grasp", size=0.012)
        add_marker(env, np.asarray(grasp_pose[:3]) + [0, 0, z_offset], (0.0, 0.8, 0.25), f"{arm}_grasp", size=0.01)
        add_pose_axes(env, pre_pose, f"{arm}_pre_grasp_axes")
        add_pose_axes(env, entry["endlink_targets"][0], f"{arm}_endlink_axes", length=0.055, width=0.003)

    for _ in range(4):
        env.scene.update_render()

    tag = f"rj2506_hammer_seed{args.seed}_grasp_candidates"
    observer_path = output_dir / f"{tag}_observer.png"
    world1_path = output_dir / f"{tag}_world1.png"
    world2_path = output_dir / f"{tag}_world2.png"
    head_path = output_dir / f"{tag}_head.png"
    json_path = output_dir / f"{tag}.json"

    legend = [
        ("brown: hammer contact", (115, 56, 13)),
        ("blue/orange: left/right pre-grasp", (30, 90, 255)),
        ("green: final grasp", (0, 160, 55)),
        ("RGB axes: candidate pose orientation", (220, 0, 0)),
        ("magenta: block target", (220, 0, 220)),
    ]

    env.scene.update_render()
    save_rgb_array(env.cameras.get_observer_rgb(), observer_path, legend)
    save_camera(env.cameras.world_camera1, world1_path, legend)
    save_camera(env.cameras.world_camera2, world2_path, legend)
    env.cameras.update_picture()
    head_rgba = env.cameras.get_rgba().get("head_camera", {}).get("rgba")
    if head_rgba is not None:
        save_rgb_array(head_rgba[:, :, :3], head_path, legend)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(as_jsonable(results), f, indent=2)

    print(f"Saved observer: {observer_path}")
    print(f"Saved world1: {world1_path}")
    print(f"Saved world2: {world2_path}")
    if head_rgba is not None:
        print(f"Saved head: {head_path}")
    print(f"Saved json: {json_path}")
    for arm in ["left", "right"]:
        status = results["arms"][arm][0]["planner_status"]
        print(f"{arm}: {status}")

    env.close_env()


if __name__ == "__main__":
    main()
