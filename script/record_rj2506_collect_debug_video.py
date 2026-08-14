import argparse
import importlib
import os
import sys
from pathlib import Path

sys.path.append("./")

import imageio.v2 as imageio
import numpy as np
import sapien.core as sapien
import transforms3d as t3d
import yaml
from PIL import Image, ImageDraw, ImageFont

from envs import CONFIGS_PATH
from envs.utils import ArmTag


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
        raise ValueError("This debug recorder expects a single dual-arm embodiment config.")

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


def make_visual_box(scene, pose, half_size, color, name):
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)
    material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(sapien.render.RenderShapeBox(half_size, material))
    entity.add_component(render_component)
    scene.add_entity(entity)
    return entity


def pose_from_matrix(matrix):
    return sapien.Pose(matrix[:3, 3], t3d.quaternions.mat2quat(matrix[:3, :3]))


def add_pose_axes(scene, pose_list, name, length=0.08, width=0.005):
    pos = np.asarray(pose_list[:3], dtype=float)
    rot = t3d.quaternions.quat2mat(pose_list[-4:])
    colors = [(1, 0, 0), (0, 0.8, 0), (0, 0.25, 1)]
    labels = ["x", "y", "z"]
    for axis_id, (color, label) in enumerate(zip(colors, labels)):
        mat = np.eye(4)
        mat[:3, :3] = rot
        mat[:3, 3] = pos + rot[:, axis_id] * length * 0.5
        half = [width, width, width]
        half[axis_id] = length * 0.5
        make_visual_box(scene, pose_from_matrix(mat), tuple(half), color, f"{name}_{label}")


def camera_rgb(camera):
    camera.take_picture()
    rgba = camera.get_picture("Color")
    return (rgba[:, :, :3] * 255).clip(0, 255).astype("uint8")


def resize_rgb(rgb, size=(640, 480)):
    return np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR))


def annotate(rgb, text):
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    lines = text.split("\n")
    y = 14
    for i, line in enumerate(lines):
        now_font = font if i == 0 else small_font
        draw.text((14, y), line, fill=(0, 0, 0), font=now_font, stroke_width=3, stroke_fill=(255, 255, 255))
        draw.text((14, y), line, fill=(255, 245, 80), font=now_font)
        y += 28 if i == 0 else 24
    return np.asarray(image)


def build_frame(env, label):
    env.scene.update_render()
    observer = resize_rgb(env.cameras.get_observer_rgb())
    world = resize_rgb(camera_rgb(env.cameras.world_camera1))
    frame = np.concatenate([observer, world], axis=1)
    return annotate(frame, label)


def append_hold(frames, env, label, count):
    for _ in range(count):
        frames.append(build_frame(env, label))


def patch_capture(env, frames, label_holder, sample_every):
    original_take_picture = env._take_picture
    counter = {"n": 0}

    def capture():
        counter["n"] += 1
        if counter["n"] % sample_every == 0:
            frames.append(build_frame(env, label_holder["text"]))

    env._take_picture = capture
    return original_take_picture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default="beat_block_hammer")
    parser.add_argument("--task-config", default="demo_rj2506")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="diagnostics/rj2506_beat_block_hammer_seed0_collect_debug.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=80)
    args = parser.parse_args()

    task_args = load_args(args.task_name, args.task_config)
    env = load_task(args.task_name)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    label = {"text": "setup"}
    env.setup_demo(now_ep_num=0, seed=args.seed, **task_args)
    original_take_picture = patch_capture(env, frames, label, args.sample_every)

    block_pose = env.block.get_functional_point(0, "pose").p
    arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
    summary = {
        "task": args.task_name,
        "config": args.task_config,
        "seed": args.seed,
        "arm": str(arm_tag),
        "rotate_lim": task_args["left_embodiment_config"]["rotate_lim"],
        "stages": [],
        "video": str(output_path),
    }

    append_hold(frames, env, f"initial | seed={args.seed} arm={arm_tag}\nleft: observer camera | right: world_camera1", 20)

    try:
        label["text"] = "stage 1: grasp hammer"
        grasp_actions = env.grasp_actor(env.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.04)
        grasp_ok = env.move(grasp_actions, save_freq=1)
        summary["stages"].append({"name": "grasp", "ok": bool(grasp_ok), "plan_success": bool(env.plan_success)})
        append_hold(frames, env, f"after grasp | ok={grasp_ok} plan_success={env.plan_success}", 16)

        if env.plan_success:
            label["text"] = "stage 2: lift hammer"
            lift_ok = env.move(env.move_by_displacement(arm_tag, z=0.07, move_axis="arm"), save_freq=1)
            summary["stages"].append({"name": "lift", "ok": bool(lift_ok), "plan_success": bool(env.plan_success)})
            append_hold(frames, env, f"after lift | ok={lift_ok} plan_success={env.plan_success}", 16)

        if env.plan_success:
            place_actions = env.place_actor(
                env.hammer,
                target_pose=env.block.get_functional_point(1, "pose"),
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.06,
                dis=0,
                is_open=False,
            )
            pre_place = place_actions[1][0].target_pose
            final_place = place_actions[1][1].target_pose
            make_visual_box(env.scene, sapien.Pose(pre_place[:3], [1, 0, 0, 0]), (0.018, 0.018, 0.018), (0, 1, 1), "pre_place_target")
            make_visual_box(env.scene, sapien.Pose(final_place[:3], [1, 0, 0, 0]), (0.018, 0.018, 0.018), (1, 0, 1), "final_place_target")
            make_visual_box(
                env.scene,
                env.block.get_functional_point(1, "pose"),
                (0.018, 0.018, 0.018),
                (1, 0, 0),
                "block_hit_target",
            )
            add_pose_axes(env.scene, pre_place, "pre_place_axes")
            add_pose_axes(env.scene, final_place, "final_place_axes")
            append_hold(
                frames,
                env,
                "stage 3: place/beat target preview\ncyan=pre-place gripper, magenta=final gripper, red=block hit target",
                40,
            )

            label["text"] = "stage 3: execute place/beat"
            place_ok = env.move(place_actions, save_freq=1)
            summary["stages"].append({"name": "place", "ok": bool(place_ok), "plan_success": bool(env.plan_success)})
            summary["place_targets"] = {
                "pre_place": [float(x) for x in pre_place],
                "final_place": [float(x) for x in final_place],
            }
            append_hold(frames, env, f"after place | ok={place_ok} plan_success={env.plan_success}", 40)

        summary["success"] = bool(env.check_success())
    finally:
        env._take_picture = original_take_picture
        env.close_env()

    if not frames:
        raise RuntimeError("No frames were captured.")

    imageio.mimsave(output_path, frames, fps=args.fps, macro_block_size=1)
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)

    print(f"Saved video: {output_path}")
    print(f"Saved summary: {json_path}")
    print(summary)


if __name__ == "__main__":
    main()
