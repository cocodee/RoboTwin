"""Exercise real grasp/strike physics and replay the recorded arm trajectories.

Run from the repository root in the myrobotwin conda environment.
"""

import argparse
from copy import deepcopy
import json
import sys

sys.path.insert(0, ".")

from diagnose_rj2506_hammer_grasp_alignment import load_args, load_task


def run(seed, replay):
    args = load_args("beat_block_hammer", "demo_rj2506")
    env = load_task("beat_block_hammer")
    try:
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        env.play_once()
        result = {
            "seed": seed,
            "plan_success": bool(env.plan_success),
            "success": bool(env.check_success()),
            "hammer_lift_m": env.info.get("hammer_lift_m", 0),
            "arm": env.info.get("info", {}).get("{a}"),
            "left_paths": len(env.left_joint_path),
            "right_paths": len(env.right_joint_path),
        }
        print(json.dumps(result), flush=True)
        assert result["plan_success"] and result["success"], result
        assert result["hammer_lift_m"] >= 0.05, result
        paths = deepcopy(dict(left_joint_path=env.left_joint_path, right_joint_path=env.right_joint_path))
    finally:
        env.close_env()
    if replay:
        args.update(paths, need_plan=False)
        # Match collection: reuse the environment, reset actors and replay paths.
        try:
            env.setup_demo(now_ep_num=0, seed=seed, **args)
            env.play_once()
            result["replay_success"] = bool(env.plan_success and env.check_success())
            result["replay_lift_m"] = env.info.get("hammer_lift_m", 0)
            assert result["replay_success"] and result["replay_lift_m"] >= 0.05, result
            assert env.left_cnt == len(paths["left_joint_path"])
            assert env.right_cnt == len(paths["right_joint_path"])
            print(json.dumps(result), flush=True)
        finally:
            env.close_env()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--replay", action="store_true")
    options = parser.parse_args()
    for seed in options.seeds:
        run(seed, options.replay)
