"""Inspect measured RJ2506 geometry during the original bell sequence."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, ".")
from diagnose_rj2506_hammer_grasp_alignment import load_args, load_task
from envs.utils import ArmTag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()
    env = load_task("click_bell")
    try:
        env.setup_demo(now_ep_num=0, seed=options.seed, **load_args("click_bell", "demo_rj2506"))
        arm = ArmTag("right" if env.bell.get_pose().p[0] > 0 else "left")
        entity = getattr(env.robot, f"{arm}_entity")

        def measure(stage):
            tcp = entity.find_link_by_name(f"{arm}_hand_tcp").get_entity_pose()
            base = entity.find_link_by_name(f"{arm}_hand_base").get_entity_pose()
            contacts = env.get_gripper_actor_contact_position("050_bell")
            print(json.dumps({"stage": stage, "arm": str(arm), "bell_id": int(env.bell_id),
                              "target": env.bell.get_contact_point(0, "list"),
                              "tcp": tcp.p.tolist(), "tcp_q": tcp.q.tolist(),
                              "base": base.p.tolist(),
                              "gripper": float(getattr(env.robot, f"get_{arm}_gripper_val")()),
                              "contacts": [np.asarray(p).tolist() for p in contacts],
                              "plan_success": bool(env.plan_success),
                              "success": bool(env.check_success())}), flush=True)

        measure("initial")
        env.move(env.grasp_actor(env.bell, arm_tag=arm, pre_grasp_dis=0.1,
                                 grasp_dis=0.1, contact_point_id=0))
        measure("above")
        env.move(env.move_by_displacement(arm, z=-0.045))
        measure("pressed")
        env.move(env.move_by_displacement(arm, z=0.045))
        measure("released")
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
