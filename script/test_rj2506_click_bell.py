"""Run the model-free RJ2506 bell expert and optional recorded-path replay."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, ".")
from diagnose_rj2506_hammer_grasp_alignment import load_args, load_task


def run(seed, replay, output_dir, record):
    args = load_args("click_bell", "demo_rj2506")
    args["save_freq"] = None
    env = load_task("click_bell")
    result = {"seed": seed, "controller": "script_only", "model_calls": 0}
    directory = output_dir / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    frames = []
    original_action = env.take_dense_action

    def action_with_frame(*a, **kw):
        value = original_action(*a, **kw)
        if record:
            from PIL import Image
            env._update_render()
            image = env.get_obs()["observation"]["head_camera"]["rgb"]
            frame = directory / f"frame_{len(frames):04d}.png"
            Image.fromarray(image).save(frame)
            frames.append(frame)
        return value

    if record:
        env.take_dense_action = action_with_frame
    try:
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        env.play_once()
        result.update(plan_success=bool(env.plan_success), success=bool(env.check_success()),
                      arm=env.info.get("info", {}).get("{a}"), bell_id=int(env.bell_id),
                      withdrawal_m=env.info.get("bell_withdrawal_m", 0),
                      left_paths=len(env.left_joint_path), right_paths=len(env.right_joint_path),
                      controller_info=deepcopy(env.info.get("bell_controller", {})))
        paths = deepcopy(dict(left_joint_path=env.left_joint_path, right_joint_path=env.right_joint_path))
    finally:
        env.close_env()
        (directory / "result.json").write_text(json.dumps(result, indent=2))
    if frames:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "5",
                        "-i", str(directory / "frame_%04d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(directory / "scripted_click_bell.mp4")], check=True)
    print(json.dumps(result), flush=True)
    assert result["plan_success"] and result["success"] and result["withdrawal_m"] > .04, result
    if replay:
        env.take_dense_action = original_action
        args.update(paths, need_plan=False)
        try:
            env.setup_demo(now_ep_num=0, seed=seed, **args)
            env.play_once()
            result.update(replay_success=bool(env.plan_success and env.check_success()),
                          replay_withdrawal_m=env.info.get("bell_withdrawal_m", 0),
                          replay_left_consumed=env.left_cnt, replay_right_consumed=env.right_cnt)
            assert result["replay_success"] and result["replay_withdrawal_m"] > .04, result
            assert env.left_cnt == len(paths["left_joint_path"])
            assert env.right_cnt == len(paths["right_joint_path"])
        finally:
            env.close_env()
            (directory / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    for seed in options.seeds:
        run(seed, options.replay, options.output_dir, options.record)
