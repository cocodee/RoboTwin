"""Bounded scripted-task baseline for RJ2506; this is not an agent evaluation.

Run from any directory using the myrobotwin environment. Each task/seed runs in
an isolated process so native crashes and timeouts cannot poison later trials.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


def task_names():
    return sorted(path.stem for path in (ROOT / "envs").glob("*.py")
                  if not path.stem.startswith("_") and path.stem != "beat_block_hammer")


def save_views(env, directory, stage):
    from PIL import Image

    for name, view in env.get_obs()["observation"].items():
        if "rgb" in view:
            Image.fromarray(view["rgb"]).save(directory / f"{stage}_{name}.png")


def worker(options):
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "script"))
    result = {"task": options.worker, "seed": options.seeds[0],
              "mode": "scripted_baseline", "status": "error", "phase": "imports"}
    env = None
    started = time.monotonic()
    directory = options.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    try:
        from diagnose_rj2506_hammer_grasp_alignment import load_args, load_task

        result["phase"] = "setup"
        args = load_args(options.worker, options.config)
        if args["embodiment"] != ["RJ2506"]:
            raise ValueError("The test config must select embodiment: [RJ2506].")
        env = load_task(options.worker)
        env.setup_demo(now_ep_num=0, seed=options.seeds[0], **args)
        if options.snapshots:
            save_views(env, directory, "initial")
        result["phase"] = "execution"
        env.play_once()
        result.update(plan_success=bool(env.plan_success),
                      task_success=bool(env.check_success()),
                      left_paths=len(env.left_joint_path),
                      right_paths=len(env.right_joint_path))
        if options.snapshots:
            save_views(env, directory, "final")
        result["status"] = "passed" if result["plan_success"] and result["task_success"] else "failed"
        if result["status"] == "passed" and options.replay:
            paths = deepcopy(dict(left_joint_path=env.left_joint_path,
                                  right_joint_path=env.right_joint_path))
            env.close_env()
            result["phase"] = "replay"
            args.update(paths, need_plan=False)
            env.setup_demo(now_ep_num=0, seed=options.seeds[0], **args)
            env.play_once()
            result["replay_plan_success"] = bool(env.plan_success)
            result["replay_task_success"] = bool(env.check_success())
            result["replay_paths_consumed"] = (
                env.left_cnt == len(paths["left_joint_path"])
                and env.right_cnt == len(paths["right_joint_path"]))
            if not all(result[key] for key in (
                    "replay_plan_success", "replay_task_success", "replay_paths_consumed")):
                result["status"] = "failed"
            if options.snapshots:
                save_views(env, directory, "replay")
        result["phase"] = "complete"
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        if options.snapshots and env is not None and hasattr(env, "cameras"):
            try:
                save_views(env, directory, "failure")
            except Exception as snapshot_error:
                result["snapshot_error"] = str(snapshot_error)
    finally:
        if env is not None and hasattr(env, "scene"):
            try:
                env.close_env()
            except Exception as error:
                result["cleanup_error"] = str(error)
                result["status"] = "error"
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tasks", nargs="+", choices=task_names())
    selection.add_argument("--all", action="store_true", help="All tasks except beat_block_hammer")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--config", default="demo_rj2506")
    parser.add_argument("--timeout", type=float, default=180,
                        help="Wall seconds per task/seed, including startup and replay")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--snapshots", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=task_names(), help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.list:
        print("\n".join(task_names()))
        return 0
    if options.timeout <= 0:
        parser.error("--timeout must be positive")
    if options.worker:
        if options.output is None or len(options.seeds) != 1:
            parser.error("worker requires --output and exactly one seed")
        return worker(options)
    if not options.tasks and not options.all:
        parser.error("select --tasks, --all, or --list")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (options.output or ROOT / "diagnostics" / f"rj2506_tasks_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for task in task_names() if options.all else options.tasks:
        for seed in options.seeds:
            directory = output / task / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            result_file = directory / "result.json"
            if result_file.exists():
                parser.error(f"Refusing to overwrite an existing trial: {directory}")
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--worker", task, "--seeds", str(seed), "--config", options.config,
                       "--output", str(directory)]
            command += [flag for flag in ("--replay", "--snapshots")
                        if getattr(options, flag[2:])]
            print(f"Running {task} seed={seed}; log={directory / 'run.log'}", flush=True)
            with (directory / "run.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=options.timeout)
                    status = "crashed" if code else "error"
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    if isinstance(error, KeyboardInterrupt):
                        raise
                    status = "timeout"
                    code = process.returncode
            result = (json.loads(result_file.read_text()) if result_file.exists() else
                      {"task": task, "seed": seed, "status": status,
                       "mode": "scripted_baseline"})
            if status == "timeout":
                result["status"] = status
            elif code and result["status"] == "passed":
                result["status"] = "crashed"
            result["returncode"] = code
            result["log"] = str(directory / "run.log")
            result_file.write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
            (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
            print(f"{task} seed={seed}: {result['status']}", flush=True)
    print(f"Results: {output / 'summary.json'}", flush=True)
    return 0 if all(result["status"] == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
