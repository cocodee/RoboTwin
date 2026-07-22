# RJ2506 Integration Plan

## Summary

Add the RJ2506 dual-arm robot as a RoboTwin embodiment. Copy assets from `/home/kdi/workspace/supre_robot_assets/assets/robots/RJ2506/` into `assets/embodiments/RJ2506/`, configure RoboTwin and CuRobo files, and validate in the `myrobotwin` conda Python environment.

## Implementation

- Create and use `myrobotwin` for all Python development, testing, and runs:
  - `conda create -n myrobotwin python=3.10 -y`
  - `conda activate myrobotwin`
  - `bash script/_install.sh`
- Use `assets/embodiments/RJ2506/urdf/relpath_RJ2506_fixed.urdf` as the main URDF.
- Modify only the copied URDF to add the missing right TCP:
  - link: `right_hand_tcp`
  - fixed joint: `right_hand_tcp_joint`
  - parent: `right_hand_base`
  - child: `right_hand_tcp`
  - origin: `xyz="0.047 -0.215 0"`, `rpy="0 0 0"`
- Add `assets/embodiments/RJ2506/config.yml` with left/right 6-DoF arm joints, finger grippers, TCP move groups, and default home state.
- Add CuRobo templates for both arms and generate concrete configs with `python script/update_embodiment_config_path.py`.
- Register `RJ2506` in `task_config/_embodiment_config.yml`.
- Add `task_config/demo_rj2506.yml` as a lightweight validation task config.

## Validation

- Confirm `myrobotwin` exists and imports core packages.
- Parse the URDF and verify all mesh paths resolve.
- Load the URDF in a minimal SAPIEN scene.
- Run CuRobo FK smoke tests for `curobo_left.yml` and `curobo_right.yml`.
- Run a representative task:
  - `conda run -n myrobotwin bash collect_data.sh beat_block_hammer demo_rj2506 0`

## Assumptions

- RJ2506 is a dual-arm single-URDF embodiment, so task configs use `embodiment: [RJ2506]`.
- Assets under `assets/embodiments/` are ignored by default; force-add them only when they must be versioned.
- If later visual tests show incorrect grasp direction, calibrate `delta_matrix` and `global_trans_matrix` without changing the external source assets.

## Further Validation Log

Validation was run in `myrobotwin` using `conda run -n myrobotwin ...`.

### Latest Integration Summary

The RJ2506 integration now passes the static and startup validation path:

- EE frame alignment between SAPIEN TCP links and CuRobo FK passes with maximum position error about `0.000001 m` and maximum rotation error about `0.000031 deg`.
- RoboTwin EE pose reads now use the EE joint child link pose. SAPIEN joint `global_pose` has the correct TCP position but not the TCP link orientation.
- RJ2506 `delta_matrix` is calibrated per arm in TCP link frame. Gripper `+X` aligns with the physical `hand_base -> hand_tcp` direction, and gripper/endlink position plus rotation roundtrip checks pass.
- CuRobo start-state self-collision is fixed by ignoring the fixed torso pair `body_link1 -> body_link2`.
- RJ2506 CuRobo planner config uses `planner.skip_warmup: true`; `CuroboPlanner` skips warmup and CUDA graph setup for that mode.
- `CuroboPlanner.plan_path` has a no-op fast path and catches CuRobo `RuntimeError` failures instead of crashing the process.
- Joint ordering is verified: SAPIEN active joints are interleaved, while RoboTwin config and CuRobo cspace use matching per-arm name order.

Current validation commands:

- `conda run -n myrobotwin python script/validate_rj2506_ee_frame.py`: PASS.
- `conda run -n myrobotwin python script/validate_rj2506_gripper_frame.py`: PASS.
- `conda run -n myrobotwin python script/validate_rj2506_collision.py`: PASS.
- `conda run -n myrobotwin python script/validate_rj2506_joint_order.py`: PASS.

Remaining blocker:

- Small Cartesian displacement planning still triggers a CuRobo trajopt/Newton TorchScript CUDA illegal-instruction error on the current GPU/software stack. The error is now caught and returned as `{"status": "Fail", "error": ...}`. Full data collection remains blocked until RJ2506 uses a stable CuRobo planning mode or a fallback planner for small EE displacements.

### Blocker Handoff: CuRobo Small-Displacement Planning Crash

Problem:

- RJ2506 can load in RoboTwin, passes FK alignment, passes gripper-frame calibration, has valid CuRobo start states, and no-op planning returns `Success`.
- Any non-zero small Cartesian displacement target, for example moving the current gripper pose by `0.03 m` along gripper `+X`, enters CuRobo trajopt and can trigger a CUDA illegal-instruction failure.
- This blocks seed search and full data collection because task execution depends on successful Cartesian EE planning.

Known failure signature:

- CuRobo path: `MotionGen.plan_single()` -> `_solve_trajopt_from_solve_state()` -> `trajopt.solve_single()` -> Newton optimizer.
- Observed kernel locations include:
  - `envs/curobo/src/curobo/opt/newton/newton_base.py`
  - `envs/curobo/src/curobo/curobolib/ls.py`
  - TorchScript function `get_x_set_jit(...)`
- Representative errors:
  - `RuntimeError: CUDA error: an illegal instruction was encountered`
  - `RuntimeError: CUDA driver error: an illegal instruction was encountered`
  - `Warp CUDA error 715: an illegal instruction was encountered`
- After the first illegal instruction, the CUDA context is contaminated in the process; later CUDA tensor creation or planner calls can fail even for unrelated cases.

Minimal reproduction from repository root:

```bash
conda run -n myrobotwin bash -lc 'timeout 240 python - <<"PY"
import os, sys, yaml, numpy as np, transforms3d as t3d
sys.path.insert(0, os.getcwd())
from envs.stack_blocks_two import stack_blocks_two

with open("task_config/demo_rj2506.yml", encoding="utf-8") as f:
    args = yaml.safe_load(f)
with open("task_config/_embodiment_config.yml", encoding="utf-8") as f:
    emb = yaml.safe_load(f)
robot_file = emb[args["embodiment"][0]]["file_path"]
with open(os.path.join(robot_file, "config.yml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

args.update({
    "task_name": "stack_blocks_two",
    "left_robot_file": robot_file,
    "right_robot_file": robot_file,
    "dual_arm_embodied": True,
    "left_embodiment_config": cfg,
    "right_embodiment_config": cfg,
    "now_ep_num": 0,
    "seed": 13,
    "save_data": False,
})

env = stack_blocks_two()
env.setup_demo(**args)
pose = np.array(env.robot.get_left_ee_pose(), dtype=float)
target = pose.copy()
target[:3] += t3d.quaternions.quat2mat(target[-4:])[:, 0] * -0.03
print(env.robot.left_plan_path(target.tolist()))
PY'
```

Control checks that currently pass:

- `conda run -n myrobotwin python script/validate_rj2506_ee_frame.py`
- `conda run -n myrobotwin python script/validate_rj2506_gripper_frame.py`
- `conda run -n myrobotwin python script/validate_rj2506_collision.py`
- `conda run -n myrobotwin python script/validate_rj2506_joint_order.py`

What has already been ruled out:

- Static EE frame mismatch: SAPIEN TCP link and CuRobo FK agree within tiny tolerances.
- Gripper transform mismatch: gripper/endlink position and rotation roundtrip checks pass.
- Start-state self-collision: CuRobo `check_start_state()` returns `(True, None)` for both arms.
- Joint order mismatch: SAPIEN active joints are interleaved, but name-based extraction and CuRobo cspace order are correct.
- No-op target handling: current EE pose planning succeeds through the new no-op fast path.
- Warmup-specific crash: `planner.skip_warmup: true` avoids warmup, but real non-zero displacement still hits the trajopt/Newton CUDA path.

Relevant files:

- `envs/robot/planner.py`
  - `CuroboPlanner.__init__()` reads `planner.skip_warmup`.
  - `CuroboPlanner.plan_path()` contains the no-op fast path and `RuntimeError` catch.
- `envs/robot/robot.py`
  - EE pose reads use `left_ee.child_link.get_entity_pose()` and `right_ee.child_link.get_entity_pose()`.
- `assets/embodiments/RJ2506/curobo_left_tmp.yml`
- `assets/embodiments/RJ2506/curobo_right_tmp.yml`
  - RJ2506 sets `planner.skip_warmup: true`.
  - `self_collision_ignore` includes `{"body_link1": ["body_link2"]}`.
- `assets/embodiments/RJ2506/config.yml`
  - Contains calibrated per-arm `delta_matrix`.

Likely next investigation paths:

- Test the same small-displacement query with `CUDA_LAUNCH_BLOCKING=1` to get a more precise failing kernel location.
- Try a CuRobo configuration that avoids trajopt/Newton for tiny moves, or uses graph/IK-only planning if available.
- Check binary compatibility among GPU driver, CUDA runtime, PyTorch `2.4.1+cu121`, Warp, and the editable local CuRobo extension.
- Rebuild CuRobo native extensions inside `myrobotwin` for the current GPU architecture.
- Add an RJ2506-specific fallback planner for small Cartesian motions if CuRobo trajopt remains unstable.

### Additional Fixes

- Added RoboTwin wrist camera links to the copied URDF:
  - `left_camera` attached to `left_hand_base`
  - `right_camera` attached to `right_hand_base`
- Changed non-task RJ2506 joints in the copied URDF from movable to fixed:
  - wheel joints
  - caster joints
  - `body_joint1`
  - `body_joint2`
- Regenerated `curobo_left.yml` and `curobo_right.yml` after template changes:
  - `conda run -n myrobotwin python script/update_embodiment_config_path.py`

### Passing Checks

- URDF static validation:
  - 35 links and 34 joints.
  - Required TCP and camera links/joints exist.
  - All mesh paths resolve.
  - No joint limits remain with `effort="0"` or `velocity="0"`.
- SAPIEN validation:
  - URDF loads successfully.
  - Scene stepping works.
  - Active joints are reduced to 16: left/right 6-DoF arms plus four gripper joints.
- RoboTwin task setup:
  - `stack_blocks_two.setup_demo()` succeeds with `demo_rj2506.yml`.
  - `left_camera` and `right_camera` bind correctly.
  - `left_ee` resolves to `left_hand_tcp_joint`.
  - `right_ee` resolves to `right_hand_tcp_joint`.
  - Left and right gripper planning return 200-step open/close trajectories.
- CuRobo config validation:
  - `MotionGenConfig.load_from_robot_config()` loads both generated configs.

### Current Limitation

- Full CuRobo arm pose planning still returns `Fail`, including no-op targets at the current EE pose.
- `collect_data.sh` should not yet be considered fully validated because data collection depends on successful arm pose planning.
- The likely remaining work is planner calibration:
  - align CuRobo end-effector frame with RoboTwin/SAPIEN TCP frame,
  - refine `delta_matrix` and `global_trans_matrix`,
  - tune collision spheres and self-collision settings,
  - then rerun `conda run -n myrobotwin bash collect_data.sh stack_blocks_two demo_rj2506 0`.

## Follow-up Solutions

These items should be handled in order. Keep each change isolated and rerun the listed checks before moving to the next item.

### 1. Align EE Frames

Problem: SAPIEN/RoboTwin TCP poses and CuRobo FK poses are not confirmed to match exactly. Pose planning can fail if the target frame sent to CuRobo is not the same frame used by its kinematic model.

Solution:

- Write a small FK alignment script that uses the same 6-DoF qpos for SAPIEN and CuRobo.
- Compare SAPIEN `left_hand_tcp` / `right_hand_tcp` pose against `CudaRobotModel.get_state()`.
- If the error is a fixed translation/rotation, adjust only the copied URDF TCP joints first:
  - `left_hand_tcp_joint`
  - `right_hand_tcp_joint`
- Acceptance: position error is below `1e-3` to `5e-3` meters and rotation error is below `1` to `3` degrees.

Status: resolved.

- Added `script/validate_rj2506_ee_frame.py`.
- Run with:
  - `conda run -n myrobotwin python script/validate_rj2506_ee_frame.py`
- Result:
  - zero, home, and offset qpos samples all pass.
  - maximum position error: about `1e-6` meters.
  - maximum rotation error: about `0.000031` degrees.
- Important validation detail:
  - EE frame alignment must be checked as pure FK immediately after `set_qpos`.
  - Do not call `scene.step()` in this FK alignment test; stepping enters physics simulation and can introduce small state drift that is unrelated to static EE frame alignment.
- Conclusion:
  - `left_hand_tcp` / `right_hand_tcp` in SAPIEN are aligned with CuRobo `ee_link`.
  - Remaining pose-planning failures are not caused by a static EE frame mismatch.

### 2. Calibrate Transform Matrices

Problem: `delta_matrix` and `global_trans_matrix` are currently identity matrices. That is enough for initialization, but not enough to prove RoboTwin grasp frames match RJ2506 end-effector frames.

Solution:

- Keep URDF TCP links as the physical TCP definition.
- Use `delta_matrix` for gripper-frame-to-EE-frame rotation bias.
- Use `global_trans_matrix` for robot-frame-to-RoboTwin-frame rotation bias.
- Calibrate left and right arms separately because RJ2506 has asymmetric hand TCP offsets.
- After every config change, regenerate CuRobo configs:
  - `conda run -n myrobotwin python script/update_embodiment_config_path.py`
- Acceptance: `get_left_tcp_pose()` and `get_right_tcp_pose()` point in the direction expected by `grasp_actor()` and pre-grasp targets.

Status:

- Implemented per-arm matrix support in `envs/robot/robot.py`; existing single `3x3` matrices remain supported, and RJ2506 can now use a `2x3x3` `delta_matrix`.
- Changed RoboTwin EE pose reads to use the EE joint child link pose, because SAPIEN joint `global_pose` has the correct TCP position but a different orientation from the TCP link.
- Calibrated `assets/embodiments/RJ2506/config.yml` so each gripper-frame `+X` axis aligns with the actual `hand_base -> hand_tcp` direction in TCP link frame.
- Kept `global_trans_matrix` as identity; the required correction is local to each EE frame.
- Added `script/validate_rj2506_gripper_frame.py`.

Validation:

- Command: `conda run -n myrobotwin python script/validate_rj2506_gripper_frame.py`
- Result: PASS.
- Left arm: `angle=0.000000 deg`, `roundtrip_pos_err=0.000e+00`.
- Right arm: `angle=0.014739 deg`, `roundtrip_pos_err=0.000e+00`.
- Rotation roundtrip error is below `0.05 deg` for both arms.
- Matrix checks: both `delta_matrix` entries are orthonormal with determinant approximately `1.0`.

### 3. Refine CuRobo Collision Settings

Problem: The current `collision.yml` uses coarse hand-written spheres. Coarse or misplaced spheres can make the start state look self-colliding and cause planning failure.

Solution:

- Test planning with no scene obstacles first.
- Check whether the homestate is in self-collision.
- Replace coarse spheres with link-specific spheres:
  - 2 to 4 spheres per arm link.
  - Small or omitted spheres for fingers, wrist cameras, and TCP links.
  - Minimal base/body spheres to avoid blocking arm workspace.
- Add `self_collision_ignore` for adjacent links and hand-local parts.
- Acceptance: homestate is not self-colliding, no-op planning succeeds, and a small EE displacement plan succeeds.

Status:

- Diagnosed the first failure as `MotionGenStatus.INVALID_START_STATE_SELF_COLLISION`.
- Enumerated current collision sphere overlaps at runtime. The only negative-margin pair was the fixed torso pair `body_link1` and `body_link2`; arm links were not self-colliding.
- Added `self_collision_ignore: {"body_link1": ["body_link2"]}` to both RJ2506 CuRobo templates and regenerated `curobo_left.yml` / `curobo_right.yml`.
- Added `planner.skip_warmup` support in `envs/robot/planner.py` and enabled it for RJ2506. This avoids a CuRobo trajopt warmup CUDA illegal-instruction crash after the start state becomes valid.
- Added a no-op fast path in `CuroboPlanner.plan_path`; if the target EE pose already matches current FK, it returns a one-step successful trajectory instead of invoking CuRobo trajopt.
- Added `script/validate_rj2506_collision.py`.

Validation:

- Command: `conda run -n myrobotwin python script/validate_rj2506_collision.py`
- Result: PASS for collision configuration, start-state validity, and no-op planning.
- Left start state: valid, `status=None`.
- Right start state: valid, `status=None`.
- Left no-op plan: `Success`.
- Right no-op plan: `Success`.

### 4. Improve Homestate and Retract Config

Problem: The current homestate is estimated. It initializes the robot, but may place the arms near singularities, body collision, or poor planner seeds.

Solution:

- Choose a stable home qpos from RJ2506 documentation or real robot defaults.
- Ensure TCPs start above the table and away from the body.
- Update both:
  - `assets/embodiments/RJ2506/config.yml` `homestate`
  - `assets/embodiments/RJ2506/curobo_left_tmp.yml` / `curobo_right_tmp.yml` `retract_config`
- Regenerate CuRobo configs after changes.
- Acceptance: task setup looks physically reasonable and no-op plus small-displacement CuRobo planning succeeds.

Status:

- Directly setting qpos to the configured homestate does not by itself make small-displacement planning succeed.
- Current start states are collision-valid after the collision fix above.
- Small-displacement planning still triggers a CuRobo CUDA illegal-instruction error inside the trajopt/Newton TorchScript kernel. `CuroboPlanner.plan_path` now catches this and returns `{"status": "Fail", "error": ...}` instead of crashing the process.
- Remaining work: either find a CuRobo planning mode that avoids this kernel on the current GPU/software stack, or switch RJ2506 motion generation to another planner/fallback path for small Cartesian displacements.

### 5. Verify Joint Ordering

Problem: SAPIEN active joints are interleaved left/right, while RoboTwin and CuRobo use per-arm joint lists. Name-based extraction should work, but it needs an explicit regression check.

Solution:

- Assert all `arm_joints_name` entries exist in SAPIEN active joints.
- Assert CuRobo `cspace.joint_names` exactly matches the corresponding RoboTwin arm joint list.
- Print start joint angles before planner calls during debugging.
- Acceptance: left and right arm qpos ordering is stable and identical between RoboTwin config and CuRobo config.

Status:

- Added `script/validate_rj2506_joint_order.py`.
- Verified that all configured arm joint names exist in SAPIEN active joints.
- Verified that each CuRobo `cspace.joint_names` list exactly matches the corresponding RoboTwin `arm_joints_name` list.

Validation:

- Command: `conda run -n myrobotwin python script/validate_rj2506_joint_order.py`
- Result: PASS.
- Left arm active indices in SAPIEN: `[0, 2, 4, 6, 8, 10]`.
- Right arm active indices in SAPIEN: `[1, 3, 5, 7, 9, 11]`.
- Name-based extraction is required and currently works correctly.

### 6. Re-enable Full Data Collection

Problem: `collect_data.sh` enters RJ2506 config and seed search, but full collection depends on successful arm pose planning.

Solution:

- Validate in this order:
  1. `setup_demo()` succeeds.
  2. Gripper open/close planning succeeds.
  3. Current EE no-op planning succeeds.
  4. Small EE displacement planning succeeds, such as `z + 0.02`.
  5. Simple pre-grasp pose planning succeeds.
  6. Full collection succeeds:
     - `conda run -n myrobotwin bash collect_data.sh stack_blocks_two demo_rj2506 0`
- Acceptance: `seed.txt` is non-empty and at least one episode is collected.
