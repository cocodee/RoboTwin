"""Deterministic RJ2506 bell demonstration with ordinary trajectory replay.

This scripted expert uses simulator target geometry, like the hammer expert.
It has no agent, model-service, or uni-robot-agent dependency.
"""
from copy import deepcopy

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch


class BellController:
    def __init__(self, env, arm):
        self.env, self.arm_tag, self.arm = env, arm, str(arm)
        self.debug = env.info.setdefault("bell_controller", {})
        self.robot = env.robot
        self.entity = getattr(self.robot, f"{self.arm}_entity")
        self.planner = getattr(self.robot, f"{self.arm}_planner")
        self.indices = [self.planner.all_joints.index(n)
                        for n in getattr(self.robot, f"{self.arm}_arm_joints_name")]
        self.limits = self.entity.get_qlimits()[self.indices]
        self.model = self.entity.create_pinocchio_model() if env.need_plan else None
        self.tcp = self.entity.find_link_by_name(f"{self.arm}_hand_tcp")
        self.tcp_index = self.entity.get_links().index(self.tcp)
        self.base_q = self.entity.get_qpos().astype(float)
        left_rotation = Rotation.from_euler("xyz", [np.pi - .5, 0, 3*np.pi/4]).as_matrix()
        if self.arm == "left":
            self.rotation = left_rotation
        else:
            # The right fingers extend along -Y; this reachable tilt differs
            # from a naive reflection of the left wrist orientation.
            self.rotation = Rotation.from_euler("xyz", [np.pi/2-.5, 0, np.pi/4]).as_matrix()

    def fk(self, joints):
        q = self.base_q.copy()
        q[self.indices] = joints
        self.model.compute_forward_kinematics(q)
        return (self.entity.get_pose() * self.model.get_link_pose(self.tcp_index)).to_transformation_matrix()

    def solve(self, position, seed):
        def residual(joints):
            pose = self.fk(joints)
            return np.r_[pose[:3, 3] - position,
                         .15 * Rotation.from_matrix(self.rotation @ pose[:3, :3].T).as_rotvec()]
        result = least_squares(residual, np.clip(seed, self.limits[:, 0]+1e-7,
                                                self.limits[:, 1]-1e-7),
                               bounds=(self.limits[:, 0], self.limits[:, 1]),
                               diff_step=1e-3, max_nfev=300,
                               xtol=1e-9, ftol=1e-9, gtol=1e-9)
        error = residual(result.x)
        return result.x if np.linalg.norm(error[:3]) < .002 and np.linalg.norm(error[3:]) < .003 else None

    def feasible(self, positions):
        if positions is None or not np.isfinite(positions).all():
            return False
        if np.any(positions < self.limits[:, 0]) or np.any(positions > self.limits[:, 1]):
            return False
        metrics = self.planner.motion_gen.rollout_fn.rollout_constraint(
            torch.as_tensor(positions, dtype=torch.float32, device="cuda").unsqueeze(1),
            use_batch_env=False)
        return bool(metrics.feasible.all().item())

    def approach(self, point):
        start = self.entity.get_qpos()[self.indices].astype(float)
        seeds = [np.array([1.02, .24, 1.08, 1.26, .58, 1.17]), start]
        seeds += [seeds[0] * np.array([a, 1, b, 1, c, d])
                  for a, b, c, d in [(1,-1,-1,-1),(-1,1,-1,1),(-1,-1,1,-1)]]
        for seed in seeds:
            goal = self.solve(point + [0, 0, .05], seed)
            if goal is None:
                continue
            for lift in ([.3] if self.arm == "left" else [.3, .5, .7, 1.0]):
                for bump in [.15, .20, .25, .30]:
                    t = np.linspace(0, 1, 401)[:, None]
                    positions = start + (goal-start)*t
                    positions[:, 1] += (lift*4*t*(1-t))[:, 0]
                    positions[:, 3] += (bump*4*t*(1-t))[:, 0]
                    if not self.feasible(positions):
                        continue
                    minimum_z = min(self.fk(q)[2, 3] for q in positions)
                    self.debug["candidate_min_tcp_z"] = float(minimum_z)
                    if minimum_z < .74 + self.env.table_z_bias + .08:
                        continue
                    self.debug.update(approach_lift=lift, approach_bump=bump)
                    # Give the physical drives time to follow the checked arc.
                    dense_t = np.linspace(0, 400, 8001)
                    return np.column_stack([np.interp(dense_t, np.arange(401), positions[:, j])
                                            for j in range(len(start))])
        return None

    def linear(self, target):
        joints = self.entity.get_qpos()[self.indices].astype(float)
        start = self.tcp.get_entity_pose().p
        positions = []
        for t in np.linspace(0, 1, 41):
            joints = self.solve(start + t*(target-start), joints)
            if joints is None:
                return None
            positions.append(joints.copy())
        positions = np.asarray(positions)
        if not self.feasible(positions):
            return None
        return np.column_stack([np.interp(np.linspace(0, 40, 401), np.arange(41), positions[:, j])
                                for j in range(len(joints))])

    def execute(self, positions):
        paths = getattr(self.env, f"{self.arm}_joint_path")
        if self.env.need_plan:
            if positions is None:
                self.env.plan_success = False
                return False
            positions = np.concatenate([positions, np.repeat(positions[-1:], 100, axis=0)])
            path = dict(status="Success", position=positions,
                        velocity=np.gradient(positions, self.env.scene.get_timestep(), axis=0))
            paths.append(deepcopy(path))
        else:
            counter = getattr(self.env, f"{self.arm}_cnt")
            path = deepcopy(paths[counter])
            setattr(self.env, f"{self.arm}_cnt", counter+1)
        # Evaluate original contact criteria throughout execution, including
        # transient press contact, while retaining the normal replay format.
        for offset in range(0, len(path["position"]), 100):
            controls = dict(left_arm=None, right_arm=None, left_gripper=None, right_gripper=None)
            controls[f"{self.arm}_arm"] = {"status": "Success",
                "position": path["position"][offset:offset+100],
                "velocity": path["velocity"][offset:offset+100]}
            self.env.take_dense_action(controls)
            self.env.check_success()
        return True

    def unhook_right_wrist(self):
        """Rotate the wrist camera away from its initial tabletop contact."""
        start = self.entity.get_qpos()[self.indices].astype(float)
        goal = start.copy()
        goal[5] = -np.pi/2
        samples = start + (goal-start)*np.linspace(0, 1, 401)[:, None]
        if not self.feasible(samples):
            return None
        return start + (goal-start)*np.linspace(0, 1, 1001)[:, None]

    def raise_right_arm(self):
        """Leave the low outboard right rest before entering the tabletop."""
        start = self.entity.get_qpos()[self.indices].astype(float)
        goal = np.array([0., 1.4, 0., .8, 0., -np.pi/2])
        samples = start + (goal-start)*np.linspace(0, 1, 401)[:, None]
        if not self.feasible(samples):
            return None
        # The checked transit stays outboard while below tabletop clearance.
        xyz = np.array([self.fk(q)[:3, 3] for q in samples])
        low = xyz[:, 2] < .74 + self.env.table_z_bias + .08
        if np.any((xyz[low, 0] < .64) & (xyz[low, 1] > -.40)):
            return None
        return start + (goal-start)*np.linspace(0, 1, 2001)[:, None]

    def run(self):
        if self.arm == "right":
            if not self.execute(self.unhook_right_wrist() if self.env.need_plan else None):
                return False
            if not self.execute(self.raise_right_arm() if self.env.need_plan else None):
                return False
            self.debug["raised_tcp_z"] = float(self.tcp.get_entity_pose().p[2])
            if self.tcp.get_entity_pose().p[2] < .74 + self.env.table_z_bias + .075:
                self.env.plan_success = False
                return False
        # Contact reference is below the visible button; use its current world
        # height plus 20 mm, never a stored seed-specific world target.
        point = np.asarray(self.env.bell.get_contact_point(0, "list")[:3]) + [0, 0, .02]
        if not self.execute(self.approach(point) if self.env.need_plan else None):
            return False
        self.env.move(self.env.close_gripper(self.arm_tag, pos=0.0))
        for height in [.035, .020, .005]:
            if not self.execute(self.linear(point+[0, 0, height]) if self.env.need_plan else None):
                return False
        pressed = self.tcp.get_entity_pose().p.copy()
        # Always withdraw, even if this scene did not meet the original criterion.
        if not self.execute(self.linear(point+[0, 0, .08]) if self.env.need_plan else None):
            return False
        self.env.info["bell_withdrawal_m"] = float(self.tcp.get_entity_pose().p[2] - pressed[2])
        return bool(self.env.check_success())
