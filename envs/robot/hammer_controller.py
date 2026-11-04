"""RJ2506 hammer demonstration using a continuous, joint-limit-checked approach.

A feasible grasp and a feasible pre-grasp can lie on different IK branches. Start
from the grasp solution and retreat vertically to select a consistent pre-grasp;
use the graph planner only for the collision-avoiding transit to that joint state.
All executed arm paths use RoboTwin's normal trajectory recording/replay format.
"""

from copy import deepcopy

import numpy as np
import sapien.core as sapien
import torch
import transforms3d as t3d
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig


class HammerController:
    def __init__(self, env, arm):
        self.env = env
        self.arm = str(arm)
        self.arm_tag = arm
        self.robot = env.robot
        self.entity = getattr(self.robot, f"{self.arm}_entity")
        self.ee = getattr(self.robot, f"{self.arm}_ee").child_link
        self.planner = getattr(self.robot, f"{self.arm}_planner")
        self.indices = [self.planner.all_joints.index(name)
                        for name in getattr(self.robot, f"{self.arm}_arm_joints_name")]
        self.limits = self.entity.get_qlimits()[self.indices]
        self.mask = np.zeros(self.entity.dof, dtype=int)
        self.mask[self.indices] = 1
        self.model = self.entity.create_pinocchio_model() if env.need_plan else None
        self.ee_index = self.entity.get_links().index(self.ee)

    def linear_path(self, qpos, start, target, steps):
        """Translation with fixed orientation, seeded from the previous IK sample."""
        qpos = np.array(qpos, dtype=float, copy=True)
        positions = []
        for fraction in np.linspace(0, 1, steps):
            pose = sapien.Pose(start.p + fraction * (target.p - start.p), target.q)
            qpos, success, _ = self.model.compute_inverse_kinematics(
                self.ee_index,
                self.entity.get_pose().inv() * pose,
                initial_qpos=qpos,
                active_qmask=self.mask,
                eps=1e-5,
                max_iterations=100,
            )
            joints = qpos[self.indices]
            if (not success or np.any(joints < self.limits[:, 0])
                    or np.any(joints > self.limits[:, 1])):
                return None
            positions.append(joints.copy())
        positions = np.asarray(positions)
        # Validate every sample against the same table/self-collision model used
        # by CuRobo's transit planner, rather than bypassing collision checks.
        metrics = self.planner.motion_gen.rollout_fn.rollout_constraint(
            torch.tensor(positions, dtype=torch.float32, device="cuda").unsqueeze(1),
            use_batch_env=False,
        )
        if not metrics.feasible.all().item():
            return None
        return positions

    def execute(self, positions):
        paths = getattr(self.env, f"{self.arm}_joint_path")
        if self.env.need_plan:
            if positions is None:
                self.env.plan_success = False
                return False
            # Settle at zero endpoint velocity before closing or striking.
            positions = np.concatenate([positions, np.repeat(positions[-1:], 100, axis=0)])
            path = {
                "status": "Success",
                "position": positions,
                "velocity": np.gradient(positions, self.env.scene.get_timestep(), axis=0),
            }
            paths.append(deepcopy(path))
        else:
            counter = getattr(self.env, f"{self.arm}_cnt")
            path = deepcopy(paths[counter])
            setattr(self.env, f"{self.arm}_cnt", counter + 1)
        controls = dict(left_arm=None, right_arm=None, left_gripper=None, right_gripper=None)
        controls[f"{self.arm}_arm"] = path
        self.env.take_dense_action(controls)
        return True

    def grasp_paths(self):
        contact = self.env.hammer.get_contact_point(0, "matrix")
        frame = contact @ np.array([[0, 0, 1, 0], [-1, 0, 0, 0],
                                    [0, -1, 0, 0], [0, 0, 0, 1]])
        origin = np.r_[frame[:3, 3] + frame[:3, :3] @ [-0.12, 0, 0],
                       t3d.quaternions.mat2quat(frame[:3, :3])]
        plan = getattr(self.robot, f"{self.arm}_plan_path")
        rotate_key = f"{self.arm}_rotate_lim"
        original_limits = getattr(self.robot, rotate_key)
        try:
            # Steep approaches keep the hammer handle clear of the palm.
            for angle in (0.0, 0.1, 0.2, 0.3, 0.4):
                setattr(self.robot, rotate_key, [angle, angle])
                pose = np.array(self.robot.create_target_pose_list(
                    origin, self.env.hammer.get_contact_point(0, "list"), self.arm)[0])
                # Put the handle 12 mm inside the fingertips, not beyond them.
                pose[:3] += t3d.quaternions.quat2mat(pose[3:])[:, 0] * 0.012
                result = plan(pose)
                if result["status"] != "Success":
                    continue
                qpos = self.entity.get_qpos().astype(float)
                qpos[self.indices] = result["position"][-1]
                goal = self.robot._trans_from_gripper_to_endlink(pose, self.arm)
                above = sapien.Pose(goal.p + [0, 0, 0.07], goal.q)
                retreat = self.linear_path(qpos, goal, above, 400)
                if retreat is None:
                    continue
                start = JointState.from_position(
                    torch.tensor(self.entity.get_qpos()[self.indices], device="cuda").reshape(1, -1),
                    joint_names=self.planner.active_joints_name,
                )
                end = JointState.from_position(
                    torch.tensor(retreat[-1], dtype=torch.float32, device="cuda").reshape(1, -1),
                    joint_names=self.planner.active_joints_name,
                )
                transit = self.planner.motion_gen.plan_single_js(
                    start, end, MotionGenPlanConfig(enable_graph=True, enable_opt=False, max_attempts=5))
                if transit.success.item():
                    return transit.interpolated_plan.position.cpu().numpy(), retreat
        finally:
            setattr(self.robot, rotate_key, original_limits)
        return None, None

    def run(self):
        initial_z = float(self.env.hammer.get_pose().p[2])
        transit, retreat = self.grasp_paths() if self.env.need_plan else (None, None)
        if not self.execute(transit):
            return False
        if not self.execute(retreat[::-1] if retreat is not None else None):
            return False
        # A half command preloads the handle without driving the fingers past
        # their meeting point (the imported 25 mm stroke exceeds the 17 mm
        # travel needed for each finger to reach the center).
        self.env.move(self.env.close_gripper(self.arm_tag, pos=0.5))
        if not self.execute(retreat):
            return False
        lift = float(self.env.hammer.get_pose().p[2]) - initial_z
        self.env.info["hammer_lift_m"] = lift
        if lift < 0.05:
            self.env.plan_success = False
            return False

        # Preserve the held orientation: align the hammer head, then strike down.
        for stage in ("align", "strike"):
            positions = None
            if self.env.need_plan:
                head = self.env.hammer.get_functional_point(0, "pose").p
                block = self.env.block.get_functional_point(1, "pose").p
                delta = block - head
                delta[2] = max(0, delta[2] + 0.035) if stage == "align" else delta[2] - 0.002
                start = self.ee.get_entity_pose()
                target = sapien.Pose(start.p + delta, start.q)
                positions = self.linear_path(self.entity.get_qpos(), start, target, 700)
            if not self.execute(positions):
                return False
        return True
