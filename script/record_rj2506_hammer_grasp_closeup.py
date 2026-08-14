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
        raise ValueError("This recorder expects a single dual-arm embodiment config.")

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


def look_at_pose(position, target):
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - position
    forward /= np.linalg.norm(forward)
    left = np.cross([0, 0, 1.0], forward)
    if np.linalg.norm(left) < 1e-6:
        left = np.array([1.0, 0.0, 0.0])
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    up /= np.linalg.norm(up)

    mat = np.eye(4)
    mat[:3, :3] = np.stack([forward, left, up], axis=1)
    mat[:3, 3] = position
    return sapien.Pose(mat)


def add_closeup_camera(scene, name, width, height, fovy_deg):
    camera = scene.add_camera(name=name, width=width, height=height, fovy=np.deg2rad(fovy_deg), near=0.01, far=10)
    return camera


def color_entity_render(entity, color):
    for component in entity.get_components():
        if not isinstance(component, sapien.render.RenderBodyComponent):
            continue
        for shape in component.render_shapes:
            material = shape.get_material()
            material.base_color = color


def make_marker(scene, name, radius, color):
    entity = sapien.Entity()
    entity.set_name(name)
    material = sapien.render.RenderMaterial(base_color=[*color[:3], color[3]])
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(sapien.render.RenderShapeSphere(radius=radius, material=material))
    entity.add_component(render_component)
    scene.add_entity(entity)
    return entity


def camera_rgb(camera):
    camera.take_picture()
    rgba = camera.get_picture("Color")
    return (rgba[:, :, :3] * 255).clip(0, 255).astype("uint8")


def resize_rgb(rgb, size):
    return np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR))


def annotate(rgb, text):
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    y = 10
    for i, line in enumerate(text.split("\n")):
        now_font = font if i == 0 else small_font
        draw.text((10, y), line, fill=(0, 0, 0), font=now_font, stroke_width=3, stroke_fill=(255, 255, 255))
        draw.text((10, y), line, fill=(255, 240, 60), font=now_font)
        y += 25 if i == 0 else 20
    return np.asarray(image)


def link_pos(entity, name):
    link = entity.find_link_by_name(name)
    if link is None:
        return None
    return np.asarray(link.get_entity_pose().p, dtype=float)


def joint_scalar(joint):
    try:
        value = joint.get_qpos()
    except AttributeError:
        value = joint.get_drive_target()
    return float(np.asarray(value).reshape(-1)[0])


def gripper_state(env, arm_tag):
    if arm_tag == "right":
        entity = env.robot.right_entity
        joints = env.robot.right_gripper
        norm = env.robot.get_right_gripper_val()
    else:
        entity = env.robot.left_entity
        joints = env.robot.left_gripper
        norm = env.robot.get_left_gripper_val()

    prefix = str(arm_tag)
    tcp = link_pos(entity, f"{prefix}_hand_tcp")
    f1 = link_pos(entity, f"{prefix}_hand_finger1")
    f2 = link_pos(entity, f"{prefix}_hand_finger2")
    finger_center = np.mean([f1, f2], axis=0) if f1 is not None and f2 is not None else None
    gap = float(np.linalg.norm(f1 - f2)) if f1 is not None and f2 is not None else None
    joint_values = [joint_scalar(joint[0]) for joint in joints]
    return {
        "tcp": tcp,
        "finger1": f1,
        "finger2": f2,
        "finger_center": finger_center,
        "gap": gap,
        "norm": float(norm),
        "joint_values": joint_values,
    }


def set_closeup_views(env, arm_tag, cams, markers):
    state = gripper_state(env, arm_tag)
    hammer_pos = np.asarray(env.hammer.get_pose().p, dtype=float)
    contact_pose = env.hammer.get_contact_point(0, "pose")
    contact_pos = np.asarray(contact_pose.p, dtype=float) if contact_pose is not None else hammer_pos
    fingers = [p for p in [state["finger1"], state["finger2"]] if p is not None]
    finger_mid = np.mean(fingers, axis=0) if fingers else contact_pos
    target = 0.90 * contact_pos + 0.10 * finger_mid + np.array([0.0, 0.0, 0.028])

    markers["hammer_contact"].set_pose(sapien.Pose(contact_pos + np.array([0.0, 0.0, 0.025]), [1, 0, 0, 0]))
    if state["finger1"] is not None:
        markers["finger1"].set_pose(sapien.Pose(state["finger1"] + np.array([0.0, 0.0, 0.012]), [1, 0, 0, 0]))
    if state["finger2"] is not None:
        markers["finger2"].set_pose(sapien.Pose(state["finger2"] + np.array([0.0, 0.0, 0.012]), [1, 0, 0, 0]))
    if state["tcp"] is not None:
        markers["tcp"].set_pose(sapien.Pose(state["tcp"], [1, 0, 0, 0]))
    if state["finger_center"] is not None:
        markers["finger_center"].set_pose(
            sapien.Pose(state["finger_center"] + np.array([0.0, 0.0, 0.014]), [1, 0, 0, 0])
        )

    # Front/table-side view and lateral low view to avoid the wrist shell blocking the handle.
    cams[0].entity.set_pose(look_at_pose(target + np.array([0.00, 0.34, 0.12]), target))
    cams[1].entity.set_pose(look_at_pose(target + np.array([0.32, 0.03, 0.10]), target))
    return state, hammer_pos, contact_pos


def contact_count(env, actor_name):
    count = 0
    for contact in env.scene.get_contacts():
        if contact.bodies[0].entity.name == actor_name or contact.bodies[1].entity.name == actor_name:
            other = contact.bodies[1].entity.name if contact.bodies[0].entity.name == actor_name else contact.bodies[0].entity.name
            if other in env.robot.gripper_name:
                count += len(contact.points)
    return count


def build_frame(env, arm_tag, cams, markers, label):
    state, hammer_pos, contact_pos = set_closeup_views(env, arm_tag, cams, markers)
    env.scene.update_render()
    left = resize_rgb(camera_rgb(cams[0]), (640, 480))
    right = resize_rgb(camera_rgb(cams[1]), (640, 480))
    frame = np.concatenate([left, right], axis=1)
    joint_text = ", ".join(f"{v:.4f}" for v in state["joint_values"])
    gap_text = "n/a" if state["gap"] is None else f"{state['gap'] * 1000:.1f} mm"
    tcp_dist = None if state["tcp"] is None else float(np.linalg.norm(contact_pos - state["tcp"]) * 1000.0)
    center_dist = (
        None if state["finger_center"] is None else float(np.linalg.norm(contact_pos - state["finger_center"]) * 1000.0)
    )
    tcp_text = "n/a" if tcp_dist is None else f"{tcp_dist:.1f} mm"
    center_text = "n/a" if center_dist is None else f"{center_dist:.1f} mm"
    text = (
        f"{label} | {arm_tag} gripper closeup\n"
        f"cyan=hammer grasp, magenta=right_hand_tcp, yellow=finger-center, red/green=finger links\n"
        f"gripper_norm={state['norm']:.3f}, joints=[{joint_text}], finger-link distance={gap_text}\n"
        f"contact->tcp={tcp_text}, contact->finger-center={center_text}, hammer_z={hammer_pos[2]:.6f}, contacts={contact_count(env, env.hammer.get_name())}"
    )
    return annotate(frame, text)


def stage_metrics(env, arm_tag):
    state = gripper_state(env, arm_tag)
    contact_pose = env.hammer.get_contact_point(0, "pose")
    contact_pos = np.asarray(contact_pose.p, dtype=float) if contact_pose is not None else None
    metrics = {
        "hammer_z": float(env.hammer.get_pose().p[2]),
        "gripper_norm": state["norm"],
        "finger_link_distance": state["gap"],
        "joint_values": state["joint_values"],
        "gripper_contact_count": contact_count(env, env.hammer.get_name()),
    }
    if contact_pos is not None:
        metrics["hammer_grasp_point"] = contact_pos.tolist()
        if state["tcp"] is not None:
            metrics["right_hand_tcp"] = state["tcp"].tolist()
            metrics["hammer_grasp_to_tcp_mm"] = float(np.linalg.norm(contact_pos - state["tcp"]) * 1000.0)
        if state["finger_center"] is not None:
            metrics["finger_center"] = state["finger_center"].tolist()
            metrics["hammer_grasp_to_finger_center_mm"] = float(np.linalg.norm(contact_pos - state["finger_center"]) * 1000.0)
    return metrics


def append_hold(frames, env, arm_tag, cams, markers, label, count):
    for _ in range(count):
        frames.append(build_frame(env, arm_tag, cams, markers, label))


def patch_capture(env, frames, arm_tag, cams, label_holder, sample_every_holder):
    original_take_picture = env._take_picture
    counter = {"n": 0}

    def capture():
        counter["n"] += 1
        if counter["n"] % sample_every_holder["value"] == 0:
            frames.append(build_frame(env, arm_tag, cams, label_holder["text"]))

    env._take_picture = capture
    return original_take_picture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default="beat_block_hammer")
    parser.add_argument("--task-config", default="demo_rj2506")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="diagnostics/rj2506_beat_block_hammer_seed0_grasp_closeup.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--move-sample-every", type=int, default=12)
    parser.add_argument("--close-sample-every", type=int, default=2)
    args = parser.parse_args()

    task_args = load_args(args.task_name, args.task_config)
    env = load_task(args.task_name)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    summary = {
        "task": args.task_name,
        "config": args.task_config,
        "seed": args.seed,
        "video": str(output_path),
        "stages": [],
    }

    print("Setting up environment...", flush=True)
    env.setup_demo(now_ep_num=0, seed=args.seed, **task_args)
    color_entity_render(env.hammer.actor, [0.0, 0.25, 1.0, 1.0])
    block_pose = env.block.get_functional_point(0, "pose").p
    arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
    summary["arm"] = str(arm_tag)
    summary["gripper_scale"] = task_args["left_embodiment_config"]["gripper_scale"]
    summary["rotate_lim"] = task_args["left_embodiment_config"]["rotate_lim"]

    cams = [
        add_closeup_camera(env.scene, "hammer_grasp_closeup_oblique", 960, 720, 22),
        add_closeup_camera(env.scene, "hammer_grasp_closeup_low", 960, 720, 20),
    ]

    try:
        markers = {
            "hammer_contact": make_marker(env.scene, "hammer_contact_marker", 0.012, (0.0, 1.0, 1.0, 1.0)),
            "tcp": make_marker(env.scene, "right_hand_tcp_marker", 0.010, (1.0, 0.0, 1.0, 1.0)),
            "finger_center": make_marker(env.scene, "finger_center_marker", 0.010, (1.0, 1.0, 0.0, 1.0)),
            "finger1": make_marker(env.scene, "finger1_marker", 0.008, (1.0, 0.0, 0.0, 1.0)),
            "finger2": make_marker(env.scene, "finger2_marker", 0.008, (0.0, 1.0, 0.0, 1.0)),
        }
        append_hold(frames, env, arm_tag, cams, markers, f"initial | seed={args.seed}", 20)
        hammer_z0 = float(env.hammer.get_pose().p[2])

        print("Planning grasp pose...", flush=True)
        actions_arm, actions = env.grasp_actor(env.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.04)

        for i, action in enumerate(actions):
            if action.action != "move":
                continue
            print(f"Executing grasp move action {i} without video capture...", flush=True)
            ok = env.move((actions_arm, [action]), save_freq=None)
            state = gripper_state(env, arm_tag)
            metrics = stage_metrics(env, arm_tag)
            metrics.update({"name": f"grasp action {i}: move to target", "ok": bool(ok), "plan_success": bool(env.plan_success)})
            summary["stages"].append(metrics)
            append_hold(frames, env, arm_tag, cams, markers, f"after grasp move action {i} | ok={ok}", 16)
            if not ok or not env.plan_success:
                break

        if env.plan_success:
            close_actions = [action for action in actions if action.action != "move"]
            close_target = close_actions[-1].target_gripper_pos if close_actions else 0.0
            current = env.robot.get_right_gripper_val() if arm_tag == "right" else env.robot.get_left_gripper_val()
            print(f"Recording finger close: {current:.3f} -> {close_target:.3f}", flush=True)
            close_values = np.linspace(float(current), float(close_target), 201)
            for step_id, value in enumerate(close_values):
                env.robot.set_gripper(float(value), str(arm_tag), gripper_eps=0)
                env.scene.step()
                if step_id % args.close_sample_every == 0:
                    frames.append(build_frame(env, arm_tag, cams, markers, f"manual close fingers | step={step_id:03d} value={value:.3f}"))
            append_hold(frames, env, arm_tag, cams, markers, "after manual close fingers", 24)
            metrics = stage_metrics(env, arm_tag)
            metrics.update({"name": "manual close fingers", "ok": True, "plan_success": bool(env.plan_success)})
            summary["stages"].append(metrics)

        if env.plan_success:
            print("Executing lift without dense video capture...", flush=True)
            lift_ok = env.move(env.move_by_displacement(arm_tag, z=0.07, move_axis="arm"), save_freq=None)
            summary["stages"].append(
                {
                    "name": "lift after close",
                    "ok": bool(lift_ok),
                    "plan_success": bool(env.plan_success),
                    **stage_metrics(env, arm_tag),
                    "hammer_z": float(env.hammer.get_pose().p[2]),
                    "lift_delta": float(env.hammer.get_pose().p[2] - hammer_z0),
                }
            )
            append_hold(frames, env, arm_tag, cams, markers, f"after lift | ok={lift_ok}", 24)
    finally:
        env.close_env()

    if not frames:
        raise RuntimeError("No frames were captured.")

    imageio.mimsave(output_path, frames, fps=args.fps, macro_block_size=1)
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)

    print(f"Saved video: {output_path}")
    print(f"Saved summary: {output_path.with_suffix('.json')}")
    print(summary)


if __name__ == "__main__":
    main()
