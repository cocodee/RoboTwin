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


def load_args(task_name, task_config, gripper_center_offset=None, gripper_endlink_offset=None):
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
    robot_config = get_embodiment_config(robot_file)
    if gripper_center_offset is not None:
        robot_config["gripper_center_offset"] = gripper_center_offset
    if gripper_endlink_offset is not None:
        robot_config["gripper_endlink_offset"] = gripper_endlink_offset
    args["left_embodiment_config"] = robot_config
    args["right_embodiment_config"] = robot_config
    args["embodiment_name"] = str(embodiment[0])
    args["task_config"] = task_config
    args["save_path"] = os.path.join(args["save_path"], task_name, task_config)
    args["need_plan"] = True
    args["save_data"] = False
    args["render_freq"] = 0
    return args


def as_list(value):
    if value is None:
        return None
    return [float(x) for x in np.asarray(value, dtype=float).reshape(-1)]


def pose_to_list(pose):
    return as_list(np.concatenate([np.asarray(pose.p), np.asarray(pose.q)]))


def look_at_pose(position, target):
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - position
    forward /= np.linalg.norm(forward)
    left = np.cross([0.0, 0.0, 1.0], forward)
    if np.linalg.norm(left) < 1e-6:
        left = np.array([1.0, 0.0, 0.0])
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    up /= np.linalg.norm(up)
    mat = np.eye(4)
    mat[:3, :3] = np.stack([forward, left, up], axis=1)
    mat[:3, 3] = position
    return sapien.Pose(mat)


def add_camera(scene, name, width=1280, height=720, fovy_deg=24):
    return scene.add_camera(name=name, width=width, height=height, fovy=np.deg2rad(fovy_deg), near=0.01, far=10)


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


def annotate(rgb, lines):
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    y = 10
    for line in lines:
        draw.text((10, y), line, fill=(0, 0, 0), font=font, stroke_width=3, stroke_fill=(255, 255, 255))
        draw.text((10, y), line, fill=(255, 240, 60), font=font)
        y += 23
    return np.asarray(image)


def link_pose(entity, name):
    link = entity.find_link_by_name(name)
    if link is None:
        return None
    return link.get_entity_pose()


def joint_scalar(joint):
    try:
        value = joint.get_qpos()
    except AttributeError:
        value = joint.get_drive_target()
    return float(np.asarray(value).reshape(-1)[0])


def contact_points(env, actor_name):
    points = []
    pairs = []
    for contact in env.scene.get_contacts():
        if contact.bodies[0].entity.name == actor_name or contact.bodies[1].entity.name == actor_name:
            other = contact.bodies[1].entity.name if contact.bodies[0].entity.name == actor_name else contact.bodies[0].entity.name
            if other in env.robot.gripper_name:
                for point in contact.points:
                    points.append(np.asarray(point.position, dtype=float))
                    pairs.append(other)
    return points, pairs


def vector_metrics(a, b):
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {
        "vector_m": as_list(diff),
        "distance_m": float(np.linalg.norm(diff)),
        "distance_mm": float(np.linalg.norm(diff) * 1000.0),
    }


def projected_metrics(point, center, tcp_pose):
    rot = t3d.quaternions.quat2mat(tcp_pose.q)
    local = rot.T @ (np.asarray(point) - np.asarray(center))
    return {
        "tcp_local_vector_m": as_list(local),
        "tcp_local_vector_mm": as_list(local * 1000.0),
        "x_along_tcp_mm": float(local[0] * 1000.0),
        "y_open_axis_mm": float(local[1] * 1000.0),
        "z_tcp_mm": float(local[2] * 1000.0),
    }


def arm_entity_and_joints(env, arm_tag):
    if arm_tag == "right":
        return env.robot.right_entity, env.robot.right_gripper, env.robot.get_right_gripper_val()
    return env.robot.left_entity, env.robot.left_gripper, env.robot.get_left_gripper_val()


def measure(env, arm_tag, stage, target_gripper_pose=None):
    entity, joints, norm = arm_entity_and_joints(env, arm_tag)
    prefix = str(arm_tag)

    contact_pose = env.hammer.get_contact_point(0, "pose")
    contact_pos = np.asarray(contact_pose.p, dtype=float)
    tcp_pose = link_pose(entity, f"{prefix}_hand_tcp")
    base_pose = link_pose(entity, f"{prefix}_hand_base")
    finger1_pose = link_pose(entity, f"{prefix}_hand_finger1")
    finger2_pose = link_pose(entity, f"{prefix}_hand_finger2")
    finger_positions = [np.asarray(finger1_pose.p), np.asarray(finger2_pose.p)]
    finger_center = np.mean(finger_positions, axis=0)
    gripper_contacts, contact_bodies = contact_points(env, env.hammer.get_name())
    contact_center = np.mean(gripper_contacts, axis=0) if gripper_contacts else None

    result = {
        "stage": stage,
        "hammer_pose": pose_to_list(env.hammer.get_pose()),
        "hammer_contact_pose": pose_to_list(contact_pose),
        "target_gripper_pose": as_list(target_gripper_pose),
        "actual_tcp_pose": pose_to_list(tcp_pose),
        "actual_base_pose": pose_to_list(base_pose),
        "finger1_pose": pose_to_list(finger1_pose),
        "finger2_pose": pose_to_list(finger2_pose),
        "finger_center": as_list(finger_center),
        "gripper_norm": float(norm),
        "joint_values": [joint_scalar(joint[0]) for joint in joints],
        "finger_origin_gap_m": float(np.linalg.norm(finger_positions[0] - finger_positions[1])),
        "gripper_contact_count": len(gripper_contacts),
        "gripper_contact_bodies": contact_bodies,
        "contact_point_to_tcp": vector_metrics(contact_pos, tcp_pose.p),
        "contact_point_to_finger_center": vector_metrics(contact_pos, finger_center),
        "contact_point_to_finger1": vector_metrics(contact_pos, finger_positions[0]),
        "contact_point_to_finger2": vector_metrics(contact_pos, finger_positions[1]),
        "contact_point_in_tcp_frame": projected_metrics(contact_pos, tcp_pose.p, tcp_pose),
    }
    if contact_center is not None:
        result["gripper_contact_center"] = as_list(contact_center)
        result["hammer_contact_to_physics_contact_center"] = vector_metrics(contact_pos, contact_center)
        result["physics_contact_center_to_finger_center"] = vector_metrics(contact_center, finger_center)
    return result


def update_markers(markers, measurement):
    def pos(key):
        return np.asarray(measurement[key][:3], dtype=float)

    markers["hammer_contact"].set_pose(sapien.Pose(pos("hammer_contact_pose") + [0.0, 0.0, 0.025], [1, 0, 0, 0]))
    markers["tcp"].set_pose(sapien.Pose(pos("actual_tcp_pose"), [1, 0, 0, 0]))
    markers["finger_center"].set_pose(sapien.Pose(np.asarray(measurement["finger_center"]) + [0.0, 0.0, 0.014], [1, 0, 0, 0]))
    markers["finger1"].set_pose(sapien.Pose(pos("finger1_pose") + [0.0, 0.0, 0.012], [1, 0, 0, 0]))
    markers["finger2"].set_pose(sapien.Pose(pos("finger2_pose") + [0.0, 0.0, 0.012], [1, 0, 0, 0]))


def render_diagnostic(env, arm_tag, measurement, output_path):
    color_entity_render(env.hammer.actor, [0.0, 0.25, 1.0, 1.0])
    markers = {
        "hammer_contact": make_marker(env.scene, "diag_hammer_contact_marker", 0.012, (0.0, 1.0, 1.0, 1.0)),
        "tcp": make_marker(env.scene, "diag_tcp_marker", 0.009, (1.0, 0.0, 1.0, 1.0)),
        "finger_center": make_marker(env.scene, "diag_finger_center_marker", 0.010, (1.0, 1.0, 0.0, 1.0)),
        "finger1": make_marker(env.scene, "diag_finger1_marker", 0.007, (1.0, 0.0, 0.0, 1.0)),
        "finger2": make_marker(env.scene, "diag_finger2_marker", 0.007, (0.0, 1.0, 0.0, 1.0)),
    }
    update_markers(markers, measurement)

    contact_pos = np.asarray(measurement["hammer_contact_pose"][:3], dtype=float)
    finger_center = np.asarray(measurement["finger_center"], dtype=float)
    target = 0.75 * contact_pos + 0.25 * finger_center + np.array([0.0, 0.0, 0.03])
    cam1 = add_camera(env.scene, "diag_alignment_front", 1280, 720, 22)
    cam2 = add_camera(env.scene, "diag_alignment_side", 1280, 720, 22)
    cam1.entity.set_pose(look_at_pose(target + np.array([0.0, 0.34, 0.12]), target))
    cam2.entity.set_pose(look_at_pose(target + np.array([0.34, 0.03, 0.10]), target))
    env.scene.update_render()
    left = camera_rgb(cam1)
    right = camera_rgb(cam2)
    frame = np.concatenate([left, right], axis=1)

    cpf = measurement["contact_point_to_finger_center"]
    cpt = measurement["contact_point_to_tcp"]
    local = measurement["contact_point_in_tcp_frame"]
    lines = [
        f"{measurement['stage']} | {arm_tag} alignment",
        "cyan=hammer grasp, magenta=tcp, yellow=finger center, red/green=fingers",
        f"contact->finger_center: {cpf['distance_mm']:.1f} mm, vector={np.array(cpf['vector_m']) * 1000}",
        f"contact->tcp: {cpt['distance_mm']:.1f} mm, vector={np.array(cpt['vector_m']) * 1000}",
        f"contact in tcp frame mm: x={local['x_along_tcp_mm']:.1f}, y(open)={local['y_open_axis_mm']:.1f}, z={local['z_tcp_mm']:.1f}",
    ]
    imageio.imwrite(output_path, annotate(frame, lines))


def set_robot_offsets(robot, center_offset=None, endlink_offset=None):
    if center_offset is not None:
        robot.left_gripper_center_offset = np.asarray(center_offset, dtype=float)
        robot.right_gripper_center_offset = np.asarray(center_offset, dtype=float)
    if endlink_offset is not None:
        robot.left_gripper_endlink_offset = np.asarray(endlink_offset, dtype=float)
        robot.right_gripper_endlink_offset = np.asarray(endlink_offset, dtype=float)


def offset_target_pose(target_pose, offset):
    if offset is None:
        return target_pose
    pose = np.asarray(target_pose, dtype=float).copy()
    rot = t3d.quaternions.quat2mat(pose[-4:])
    pose[:3] += rot @ np.asarray(offset, dtype=float)
    return pose.tolist()


def rotate_target_pose(target_pose, rpy):
    if rpy is None:
        return target_pose
    pose = np.asarray(target_pose, dtype=float).copy()
    rot = t3d.quaternions.quat2mat(pose[-4:])
    rot_offset = t3d.euler.euler2mat(float(rpy[0]), float(rpy[1]), float(rpy[2]), axes="sxyz")
    pose[-4:] = t3d.quaternions.mat2quat(rot @ rot_offset)
    return pose.tolist()


def offset_move_actions(actions, offset):
    if offset is None:
        return
    for action in actions:
        if action.action == "move":
            action.target_pose = offset_target_pose(action.target_pose, offset)


def rotate_move_actions(actions, rpy):
    if rpy is None:
        return
    for action in actions:
        if action.action == "move":
            action.target_pose = rotate_target_pose(action.target_pose, rpy)


def run_diagnostic(args):
    gripper_center_offset = None
    if args.gripper_center_offset:
        gripper_center_offset = [float(item) for item in args.gripper_center_offset.split(",")]
        if len(gripper_center_offset) != 3:
            raise ValueError("--gripper-center-offset must have three comma-separated floats.")

    gripper_endlink_offset = None
    if args.gripper_endlink_offset:
        gripper_endlink_offset = [float(item) for item in args.gripper_endlink_offset.split(",")]
        if len(gripper_endlink_offset) != 3:
            raise ValueError("--gripper-endlink-offset must have three comma-separated floats.")

    target_pose_offset = None
    if args.target_pose_offset:
        target_pose_offset = [float(item) for item in args.target_pose_offset.split(",")]
        if len(target_pose_offset) != 3:
            raise ValueError("--target-pose-offset must have three comma-separated floats.")

    target_rpy_offset = None
    if args.target_rpy_offset:
        target_rpy_offset = [float(item) for item in args.target_rpy_offset.split(",")]
        if len(target_rpy_offset) != 3:
            raise ValueError("--target-rpy-offset must have three comma-separated floats.")

    task_args = load_args(
        args.task_name,
        args.task_config,
        gripper_center_offset=None if args.apply_offset_after_choose else gripper_center_offset,
        gripper_endlink_offset=None if args.apply_offset_after_choose else gripper_endlink_offset,
    )
    env = load_task(args.task_name)
    output_json = Path(args.output_json)
    output_png = Path(args.output_png)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "task": args.task_name,
        "config": args.task_config,
        "seed": args.seed,
        "pre_grasp_dis": args.pre_grasp_dis,
        "grasp_dis": args.grasp_dis,
        "insert_base_local_y": args.insert_base_local_y,
        "measurements": [],
    }

    try:
        print("Setting up environment...", flush=True)
        env.setup_demo(now_ep_num=0, seed=args.seed, **task_args)
        block_pose = env.block.get_functional_point(0, "pose").p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
        result["arm"] = str(arm_tag)
        result["gripper_scale"] = task_args["left_embodiment_config"]["gripper_scale"]
        result["gripper_bias"] = task_args["left_embodiment_config"].get("gripper_bias")
        result["gripper_center_offset"] = task_args["left_embodiment_config"].get("gripper_center_offset", [0.0, 0.0, 0.0])
        result["gripper_endlink_offset"] = task_args["left_embodiment_config"].get("gripper_endlink_offset", [0.0, 0.0, 0.0])
        result["target_pose_offset"] = target_pose_offset or [0.0, 0.0, 0.0]
        result["target_rpy_offset"] = target_rpy_offset or [0.0, 0.0, 0.0]
        result["rotate_lim"] = task_args["left_embodiment_config"].get("rotate_lim")
        result["delta_matrix"] = task_args["left_embodiment_config"].get("delta_matrix")

        print("Choosing grasp pose...", flush=True)
        pre_pose, grasp_pose = env.choose_grasp_pose(
            env.hammer,
            arm_tag=arm_tag,
            pre_dis=args.pre_grasp_dis,
            target_dis=args.grasp_dis,
        )
        result["chosen_pre_grasp_pose"] = as_list(pre_pose)
        result["chosen_final_grasp_pose"] = as_list(grasp_pose)
        result["transformed_final_endlink_pose"] = pose_to_list(
            env.robot._trans_from_gripper_to_endlink(grasp_pose, arm_tag=str(arm_tag))
        )

        print("Executing final grasp moves...", flush=True)
        actions = env.grasp_actor(env.hammer, arm_tag=arm_tag, pre_grasp_dis=args.pre_grasp_dis, grasp_dis=args.grasp_dis)
        rotate_move_actions(actions[1], target_rpy_offset)
        offset_move_actions(actions[1], target_pose_offset)
        grasp_pose = rotate_target_pose(grasp_pose, target_rpy_offset)
        grasp_pose = offset_target_pose(grasp_pose, target_pose_offset)
        if args.apply_offset_after_choose:
            set_robot_offsets(env.robot, center_offset=gripper_center_offset, endlink_offset=gripper_endlink_offset)
            result["gripper_center_offset"] = gripper_center_offset or [0.0, 0.0, 0.0]
            result["gripper_endlink_offset"] = gripper_endlink_offset or [0.0, 0.0, 0.0]
            result["transformed_final_endlink_pose_after_offset"] = pose_to_list(
                env.robot._trans_from_gripper_to_endlink(grasp_pose, arm_tag=str(arm_tag))
            )
        for action in actions[1]:
            if action.action == "move":
                ok = env.move((actions[0], [action]), save_freq=None)
                if not ok or not env.plan_success:
                    result["error"] = "move to grasp pose failed"
                    break

        final_open = measure(env, arm_tag, "at final grasp pose before close", target_gripper_pose=grasp_pose)
        result["measurements"].append(final_open)

        if env.plan_success:
            if abs(args.insert_base_local_y) > 0:
                entity, _, _ = arm_entity_and_joints(env, arm_tag)
                base_pose = link_pose(entity, f"{arm_tag}_hand_base")
                base_rot = t3d.quaternions.quat2mat(base_pose.q)
                # Positive local-y insertion moves the hand frame along -Y, so the object appears
                # deeper along +Y in the gripper slot.
                world_delta = -base_rot[:, 1] * float(args.insert_base_local_y)
                print(f"Executing base-local-y insertion, world_delta={world_delta.tolist()}...", flush=True)
                insert_ok = env.move(
                    env.move_by_displacement(
                        arm_tag,
                        x=float(world_delta[0]),
                        y=float(world_delta[1]),
                        z=float(world_delta[2]),
                        move_axis="world",
                    ),
                    save_freq=None,
                )
                result["insert_ok"] = bool(insert_ok)
                result["insert_plan_success"] = bool(env.plan_success)
                inserted = measure(env, arm_tag, "after base-local-y insertion before close", target_gripper_pose=grasp_pose)
                result["measurements"].append(inserted)

        if env.plan_success:
            print("Closing gripper manually for measurement...", flush=True)
            env.robot.set_gripper(0.0, str(arm_tag), gripper_eps=0)
            for _ in range(args.settle_steps):
                env.scene.step()
            closed = measure(env, arm_tag, "after close", target_gripper_pose=grasp_pose)
            result["measurements"].append(closed)
            render_diagnostic(env, arm_tag, closed, output_png)

        if env.plan_success:
            print("Executing lift for measurement...", flush=True)
            env.move(env.move_by_displacement(arm_tag, z=args.lift_z, move_axis=args.lift_axis), save_freq=None)
            lifted = measure(env, arm_tag, "after lift", target_gripper_pose=grasp_pose)
            result["measurements"].append(lifted)

    finally:
        env.close_env()

    with open(output_json, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, allow_unicode=True, sort_keys=False)
    print(f"Saved diagnostic json: {output_json}")
    print(f"Saved diagnostic image: {output_png}")
    return result


def print_summary(result):
    print("stage,contact_to_finger_center_mm,contact_to_tcp_mm,tcp_local_x_mm,tcp_local_y_open_mm,tcp_local_z_mm,contacts")
    for measurement in result["measurements"]:
        cpf = measurement["contact_point_to_finger_center"]["distance_mm"]
        cpt = measurement["contact_point_to_tcp"]["distance_mm"]
        local = measurement["contact_point_in_tcp_frame"]
        print(
            f"{measurement['stage']},"
            f"{cpf:.3f},"
            f"{cpt:.3f},"
            f"{local['x_along_tcp_mm']:.3f},"
            f"{local['y_open_axis_mm']:.3f},"
            f"{local['z_tcp_mm']:.3f},"
            f"{measurement['gripper_contact_count']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default="beat_block_hammer")
    parser.add_argument("--task-config", default="demo_rj2506")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pre-grasp-dis", type=float, default=0.12)
    parser.add_argument("--grasp-dis", type=float, default=0.04)
    parser.add_argument("--lift-z", type=float, default=0.07)
    parser.add_argument("--lift-axis", choices=["world", "arm"], default="arm")
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--gripper-center-offset", default=None)
    parser.add_argument("--gripper-endlink-offset", default=None)
    parser.add_argument("--target-pose-offset", default=None)
    parser.add_argument("--target-rpy-offset", default=None)
    parser.add_argument("--insert-base-local-y", type=float, default=0.0)
    parser.add_argument("--apply-offset-after-choose", action="store_true")
    parser.add_argument("--output-json", default="diagnostics/rj2506_hammer_grasp_alignment_seed0.yml")
    parser.add_argument("--output-png", default="diagnostics/rj2506_hammer_grasp_alignment_seed0.png")
    args = parser.parse_args()

    result = run_diagnostic(args)
    print_summary(result)


if __name__ == "__main__":
    main()
