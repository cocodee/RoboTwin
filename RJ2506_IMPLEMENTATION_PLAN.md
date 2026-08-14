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

Status update 2026-07-30:

- The requested run used the `RoboTwin` conda environment, not `myrobotwin`:
  - `TORCH_CUDA_ARCH_LIST=9.0 conda run --no-capture-output -n RoboTwin bash collect_data.sh beat_block_hammer demo_rj2506 0`
- Initial symptom:
  - `target_pose cannot be None for move action.`
  - This was only a secondary error. `envs/beat_block_hammer.py` calls `grasp_actor(self.hammer, ...)`; `choose_grasp_pose()` found no feasible pre-grasp/grasp pose for `020_hammer`, returned `(None, None)`, and `grasp_actor()` then attempted to create a move `Action` with `target_pose=None`.
- CuRobo environment issue found during diagnosis:
  - The CuRobo source tree contained stale compiled extensions:
    - `kinematics_fused_cu.cpython-310-x86_64-linux-gnu.so`
    - `geom_cu.cpython-310-x86_64-linux-gnu.so`
    - `tensor_step_cu.cpython-310-x86_64-linux-gnu.so`
    - `lbfgs_step_cu.cpython-310-x86_64-linux-gnu.so`
    - `line_search_cu.cpython-310-x86_64-linux-gnu.so`
  - These were incompatible with the active `RoboTwin` PyTorch/CUDA ABI and produced `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb`.
  - They were renamed to `.so.bak`, forcing CuRobo to use current-environment JIT extensions from `~/.cache/torch_extensions`.
- CuRobo planning issue found during diagnosis:
  - With clean JIT extensions, no-op planning succeeded, but non-no-op planning still triggered `CUDA illegal instruction` in CuRobo trajopt/Newton/LBFGS kernels.
  - Direct `MotionGenPlanConfig(enable_graph=True, enable_opt=False)` testing showed graph-only planning succeeds for small RJ2506 displacements.
- Code/config changes made:
  - `envs/robot/planner.py`
    - Added support for CuRobo planner YAML keys:
      - `planner.graph_only`
      - `planner.max_attempts`
      - `planner.timeout`
    - When `graph_only: true`, `CuroboPlanner` uses `MotionGenPlanConfig(enable_graph=True, enable_opt=False, ...)`, avoiding the unstable optimizer kernels.
    - `plan_batch()` now returns a status array sized to the actual batch and includes the caught error text.
  - `assets/embodiments/RJ2506/curobo_left.yml`
  - `assets/embodiments/RJ2506/curobo_right.yml`
  - `assets/embodiments/RJ2506/curobo_left_tmp.yml`
  - `assets/embodiments/RJ2506/curobo_right_tmp.yml`
    - Added:
      - `graph_only: true`
      - `max_attempts: 1`
      - `timeout: 2.0`
  - `envs/_base_task.py`
    - Fixed `choose_best_pose()` so `now_step` is updated when a shorter successful path is selected.
    - Added an explicit failure in `grasp_actor()` when no feasible grasp pose is found:
      - `No feasible grasp pose for <actor> with <arm> arm.`
    - This replaces the misleading `target_pose cannot be None for move action` failure.
- Validation after the graph-only change:
  - `TORCH_CUDA_ARCH_LIST=9.0 conda run --no-capture-output -n RoboTwin python script/validate_rj2506_joint_order.py`
    - PASS.
  - `TORCH_CUDA_ARCH_LIST=9.0 conda run --no-capture-output -n RoboTwin python script/validate_rj2506_collision.py`
    - PASS.
  - Manual small-displacement validation:
    - Left no-op: `Success`
    - Left `z + 0.03`: `Success`
    - Left `y + 0.03`: `Success`
    - Right no-op: `Success`
    - Right `z + 0.03`: `Success`
    - Right `y + 0.03`: `Success`
- Remaining blocker for `beat_block_hammer demo_rj2506`:
  - Full collection still fails in seed search, now with the clearer task-level error:
    - `No feasible grasp pose for 020_hammer with left arm.`
    - `No feasible grasp pose for 020_hammer with right arm.`
  - The remaining issue is not the RJ2506 small-displacement planner kernel. It is the `020_hammer` grasp definition/placement for this task:
    - `020_hammer` has only one contact point.
    - The default hammer contact-derived grasp pose is not feasible for RJ2506.
    - Seed changes move the block and therefore switch selected arm, but the hammer pose/contact point stays effectively fixed, so all tested seeds fail at the same first grasp stage.
- Next required task-level fix:
  - Add one or more RJ2506-compatible hammer grasp poses/contact points, or adjust the hammer initial pose for `beat_block_hammer` when `embodiment_name == "RJ2506"`.
  - Acceptance for this remaining fix: `choose_grasp_pose(self.hammer, ...)` returns non-`None`, the first hammer grasp action plans and executes, and `seed.txt` receives at least one successful seed for `beat_block_hammer/demo_rj2506`.

## 2026-07-31 Follow-up: RJ2506 Gripper Direction

- The RJ2506 gripper URDF uses prismatic finger limits `[0, 0.025]`, but visual/URDF inspection indicates:
  - `q = 0.0`: gripper open.
  - `q = 0.025`: gripper closed.
- RoboTwin task semantics are normalized in the opposite naming convention:
  - `close_gripper(pos=0.0)` means closed.
  - `open_gripper(pos=1.0)` means open.
- The previous RJ2506 config had `gripper_scale: [0.0, 0.025]`, so RoboTwin close commands mapped to the URDF open joint value.
- Fixed `assets/embodiments/RJ2506/config.yml`:
  - `gripper_scale: [0.025, 0.0]`
- Mapping after the fix:
  - `close`: normalized `0.0` -> joint target `0.025`
  - `half`: normalized `0.5` -> joint target `0.0125`
  - `open`: normalized `1.0` -> joint target `0.0`
- Validation command:
  - `CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0 timeout 240 conda run --no-capture-output -n RoboTwin python -u script/record_rj2506_collect_debug_video.py --seed 0 --output diagnostics/rj2506_beat_block_hammer_seed0_collect_debug_gripper_inverted.mp4`
- Validation result:
  - `grasp`: action/planning succeeded.
  - `lift`: action/planning succeeded.
  - `place`: still failed at planning.
- Current remaining blocker:
  - The original no-grasp/physical non-pickup issue is addressed by the gripper direction fix.
  - The next blocker is now the `place_actor` / beat target planning pose for seed 0.

## 2026-07-31 Follow-up: Hammer Still Not Lifted

- User review of the debug video showed the hammer still was not actually lifted.
- Rechecked with the corrected gripper scale:
  - `grasp` and `lift` actions can report planning/action success.
  - Hammer height remains effectively unchanged:
    - before grasp: about `0.783403 m`
    - after lift: about `0.783421 m`
    - lift delta: about `0.000017 m`
  - Therefore the hammer is contacted but not physically grasped.
- Tested whether this was a simple 90 degree horizontal grasp orientation issue:
  - `rotate_lim` exact sweep from `0` to `90 deg`:
    - feasible angles around `28.65 deg` and `42.97 deg` still produced near-zero lift delta.
    - `71.62 deg` and `90 deg` produced no feasible grasp pose for seed 0.
  - negative-angle sweep down to `-90 deg`:
    - feasible angles around `-28.65 deg` and `-42.97 deg` still produced near-zero lift delta.
    - `-71.62 deg` and `-90 deg` produced no feasible grasp pose.
  - world-Z yaw tests:
    - `+90 deg`: grasp/lift action succeeded but lift delta remained `0`.
    - `-90 deg` and `180 deg`: planning failed.
  - `delta_matrix` TCP-roll tests:
    - `+90 deg` and `-90 deg` both made the tested right-arm grasp candidates infeasible.
- Runtime axis check after grasp:
  - right finger opening axis vs hammer long axis: about `89.55 deg`.
  - This means the selected grasp is already laterally perpendicular to the hammer handle; the failure is not explained by a simple 90 degree in-plane pose rotation.
- Re-tested `gripper_bias` after fixing `gripper_scale`:
  - `0.04`: feasible at one angle, but lift delta remained `0`.
  - `0.08` and `0.12`: tested candidates became infeasible.
- Current interpretation:
  - The remaining blocker is likely the RJ2506 gripper geometry/TCP/contact definition relative to the hammer handle, not just a 90 degree pose rotation.
  - The gripper makes contact with the hammer but does not create a stable clamping grasp.

## 2026-07-31 Follow-up: Quantitative Grasp Alignment and Offset Tests

- Added close-up and quantitative diagnostics:
  - `script/record_rj2506_hammer_grasp_closeup.py`
  - `script/diagnose_rj2506_hammer_grasp_alignment.py`
- Baseline seed 0 alignment at final grasp / after close:
  - `contact -> right_hand_tcp`: about `42 mm`.
  - `contact in TCP frame`: about `x=9 mm, y(open)=-41 mm, z=4 mm`.
  - Closed gripper reached joint values `[0.025, 0.025]`.
  - Contact count after close was only `1`, and after lift returned to `0`.
- Added generic offset support in `envs/robot/robot.py` for future calibration:
  - `gripper_center_offset`: offset in abstract gripper-target frame.
  - `gripper_endlink_offset`: offset in planned endlink/TCP frame.
  - Defaults are `[0, 0, 0]`, so existing embodiments are unchanged unless configured.
- Offset and rotation tests performed, all using `conda run --no-capture-output -n RoboTwin`:
  - `gripper_center_offset` y sweep: `-40 mm`, `+40 mm`, `-20 mm`, `+20 mm`.
    - Did not reduce the TCP open-axis error.
    - Some values made the grasp infeasible or worse.
  - `gripper_endlink_offset` y sweep: `-40 mm`, `+40 mm`, `-20 mm`, `+20 mm`.
    - Also did not reduce the open-axis error.
  - Task-level `target_pose_offset` sweeps:
    - x `+/-40 mm`: worse / infeasible.
    - y `+/-40 mm`: mainly changed TCP local z or made planning worse.
    - z `-40 mm`: increased close contact count to `8`, but the hammer still detached during lift.
    - z `+40 mm`: worse / infeasible.
  - World-frame lift (`move_axis="world"`) was also tested:
    - Baseline and `target_pose_offset z=-40 mm` both still lost contact after lift.
  - Local target-pose `90 deg` rotations around x/y/z were tested:
    - Most were infeasible or moved TCP far from the hammer.
    - No 90 degree rotation produced a stable pickup.
- Current conclusion:
  - The visible cyan-point offset alone is not safely fixable by a simple translational offset or 90 degree local rotation.
  - The best translational candidate (`target_pose_offset=[0, 0, -0.04]`) improves contact count but still does not lift the hammer, so it should not be committed as the final RJ2506 config.
  - The remaining likely issue is deeper than a single pose offset: RJ2506 finger collision/contact geometry, TCP frame placement, and/or hammer handle contact geometry need to be calibrated against the actual finger pad/contact surfaces.

## 2026-07-31 Follow-up: Direct `right_hand_tcp` Origin Shift Test

- Current URDF definition:
  - `right_hand_tcp_joint` is fixed under `right_hand_base`.
  - Original origin: `xyz="0.047 -0.215 0"`.
- Tried shifting `right_hand_tcp_joint` directly from the baseline after-close TCP-local contact error:
  - Forward shift:
    - `xyz="0.055913929411563916 -0.25553313628131286 0.003554157596203344"`
    - Result: close contact count became `0`; worse than baseline.
  - Opposite shift:
    - `xyz="0.03808607058843608 -0.17446686371868714 -0.003554157596203344"`
    - Result: final grasp execution did not proceed to a stable close measurement; worse than baseline.
- Restored `right_hand_tcp_joint` to the original origin:
  - `xyz="0.047 -0.215 0"`.
- Interpretation:
  - Directly shifting the URDF TCP origin by the observed one-shot error is not sufficient.
  - The TCP origin, `delta_matrix`, and the generated grasp candidate orientation are coupled. Moving only the URDF TCP changes planning/IK behavior and candidate feasibility, and did not improve the hammer pickup.

## 2026-08-03 Follow-up: Finger Inertial X Offset Test

- User observed `right_hand_finger1` inertial origin:
  - `xyz="0.0434939503684225 -0.180370756431385 -0.0197407586121252"`.
- Interpretation:
  - This origin is in the local frame of `right_hand_finger1`, not in world frame or directly in `right_hand_tcp` frame.
  - It can be tested as a candidate local grasp-pose x offset, but it is not automatically the physical grasp center.
- Tested task-level final grasp target offsets using:
  - `--target-pose-offset 0.0434939503684225,0,0`
  - `--target-pose-offset=-0.0434939503684225,0,0`
- Results:
  - `+x`:
    - final pose `contact -> tcp`: `86.18 mm`, worse than baseline about `42 mm`.
    - final pose `contact -> finger_center`: `300.74 mm`.
    - no gripper contact; later move failed.
  - `-x`:
    - final pose `contact -> tcp`: `95.25 mm`.
    - after close `contact -> tcp`: `86.09 mm`.
    - after close contact count: `0`.
    - after lift contact count: `0`.
- Conclusion:
  - Applying the finger inertial x value as a direct final-grasp x offset does not improve RJ2506 hammer pickup.
  - This should not be committed as a permanent grasp/TCP correction.
  - The more likely useful calibration target is the actual finger pad/contact surface center from collision or visual mesh bounds, not the inertial center.

## 2026-08-03 Follow-up: Gripper Slot vs Hammer Contact Position

- Measured RJ2506 right finger mesh bounds from STL:
  - `right_hand_finger1.STL` y range: about `-218 mm` to `-151 mm` in `right_hand_base` local frame.
  - `right_hand_finger2.STL` y range: about `-218 mm` to `-151 mm`.
  - `right_hand_tcp_joint` origin remains `xyz="0.047 -0.215 0"`.
- Baseline seed 0 after close:
  - Hammer grasp point in `right_hand_base` local frame: about `[55.9, -255.5, 3.6] mm`.
  - This places the hammer grasp point about `37 mm` beyond the fingertip y range, so the gripper cannot actually clamp the hammer.
- Candidate geometry enumeration:
  - With current `grasp_dis=0.04`, feasible candidates place the contact at about `[55.5, -254.1, 0] mm`.
  - With `grasp_dis=0.0`, the geometry would place the contact at about `[47.0, -215.0, 0] mm`, which is inside the fingertip/TCP region.
  - However, all `grasp_dis=0.0`, `0.02`, and negative-depth candidates failed CuRobo planning in graph-only mode.
- Tested a base-local-y insertion after the current feasible final grasp:
  - `--insert-base-local-y 0.04` generated a world delta with a large downward component.
  - Planner rejected the insertion (`insert_ok: False`), so it did not change the measured pose.
- Tested a pure URDF TCP y shift:
  - Changed `right_hand_tcp_joint` y from `-0.215` to `-0.175` temporarily.
  - Result: final grasp move failed and contact-to-TCP distance worsened to about `127 mm`.
  - Restored the original TCP origin.
- Current conclusion:
  - The core failure is now identified: the planned reachable grasp stops outside the RJ2506 finger slot.
  - The geometrically correct grasp depth exists, but current CuRobo graph-only planning rejects it.
  - Simple `grasp_dis`, target offset, finger inertial offset, or single-axis TCP origin changes are not enough.
  - Remaining fix direction should focus on a grasp-stage collision/planning policy:
    - allow a short final insertion toward the target object,
    - or plan to a collision-free pre-grasp then use a local Cartesian/IK servo for the last few centimeters,
    - or build a task/robot-specific grasp target that is collision-feasible while placing the physical hammer handle inside the finger slot.

## 2026-08-03 Follow-up: 2x Gripper Open Width Test

- Tested whether RJ2506 fails because the gripper is not open wide enough before reaching the hammer.
- Temporary changes for the test:
  - `gripper_scale: [0.025, -0.025]`
  - finger prismatic joint lower limits changed from `0` to `-0.025`
  - close remained `0.025`; open became `-0.025`
- Result:
  - Open finger-link origin gap increased to about `50.0 mm`, so the widened-open test was active.
  - Final grasp planning still succeeded at the shallow pose.
  - After close:
    - `contact -> tcp`: about `41.32 mm`, essentially unchanged from baseline.
    - gripper contact count: `2`.
  - After lift:
    - gripper contact count returned to `0`.
    - hammer height still did not meaningfully increase.
- Restored the original limits/config after the test:
  - finger lower limits: `0`
  - `gripper_scale: [0.025, 0.0]`
- Conclusion:
  - Increasing the open width alone does not solve the hammer pickup.
  - The failure is dominated by grasp depth/placement: the feasible planned pose still leaves the hammer grasp point outside the effective finger slot.

## 2026-08-10 Follow-up: Other `collect_data` Examples

- Tested additional data-collection examples with the `RoboTwin` conda environment.
- Commands used direct unbuffered Python invocation because `collect_data.sh` output can appear stalled during CuRobo/SAPIEN initialization:
  - `CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0 timeout 180 conda run --no-capture-output -n RoboTwin python -u script/collect_data.py stack_blocks_two demo_rj2506`
  - `CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0 timeout 180 conda run --no-capture-output -n RoboTwin python -u script/collect_data.py place_empty_cup demo_rj2506`
  - `CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0 timeout 150 conda run --no-capture-output -n RoboTwin python -u script/collect_data.py grab_roller demo_rj2506`
- Results:
  - `stack_blocks_two demo_rj2506`:
    - Seed search reached at least seed `22`.
    - Repeated failures; some explicit errors: `No feasible grasp pose for box with left arm.`
    - No seed collected.
  - `place_empty_cup demo_rj2506`:
    - Seed search reached at least seed `19`.
    - Repeated explicit errors: `No feasible grasp pose for 021_cup with left/right arm.`
    - No seed collected.
  - `grab_roller demo_rj2506`:
    - Seed search reached at least seed `22`.
    - Repeated explicit errors: `No feasible grasp pose for 102_roller with left/right arm.`
    - No seed collected.
- Also tested an official aloha-agilex one-episode smoke config based on `demo_clean`:
  - `CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0 timeout 300 conda run --no-capture-output -n RoboTwin python -u script/collect_data.py beat_block_hammer demo_clean_1ep`
  - It entered the task but repeatedly failed with `CUDA error: an illegal instruction was encountered`, consistent with the known CuRobo optimizer kernel issue when not using the RJ2506 graph-only planner path.
- Output check:
  - Empty seed files were created for the tested runs.
  - No successful `episode*.hdf5` was produced.
- Current conclusion:
  - The RJ2506 issue is not isolated to `beat_block_hammer`.
  - Multiple object-grasp tasks fail at the same abstraction boundary: `choose_grasp_pose()` / CuRobo feasibility for object contact-derived poses.
  - Fixing hammer alone can make that task work, but robust RJ2506 data collection needs a broader grasp-pose/planning calibration for the embodiment.

## 2026-08-10 Follow-up: Franka-Panda `beat_block_hammer` Smoke Test

- Goal:
  - Run a one-episode Franka-Panda `beat_block_hammer` data-collection example in the `RoboTwin` conda environment.
- Added config:
  - `task_config/demo_franka_1ep.yml`
  - One episode, clean scene, `embodiment: [franka-panda]`.
- Franka asset/config fixes required:
  - Current `Robot.set_planner()` expects `curobo_left.yml` and `curobo_right.yml` when `dual_arm_embodied=True`.
  - Franka assets only had `curobo.yml`, so the first run failed with:
    - missing `assets/embodiments/franka-panda/curobo_left.yml`
  - Created:
    - `assets/embodiments/franka-panda/curobo_left.yml`
    - `assets/embodiments/franka-panda/curobo_right.yml`
    - both copied from the official `curobo.yml`.
  - Added graph-only planner keys to `curobo.yml`, `curobo_left.yml`, and `curobo_right.yml`:
    - `skip_warmup: true`
    - `graph_only: true`
    - `max_attempts: 1`
    - `timeout: 2.0`
  - This avoids the known CuRobo optimizer/Newton/LBFGS CUDA illegal-instruction path on this machine.
- Franka gripper config fix:
  - URDF finger limits are `[0.0, 0.04]`.
  - Original config had `gripper_scale: [-0.01, 0.05]`, which drives both close/open beyond URDF limits.
  - Changed to `gripper_scale: [0.0, 0.04]`.
- `beat_block_hammer` task adjustment:
  - The earlier RJ2506-specific `grasp_dis=0.04` was made conditional:
    - RJ2506 uses `grasp_dis=0.04`.
    - Franka and other embodiments use default `grasp_dis=0.0`.
  - Added a Franka-Panda-specific execution branch:
    - grasp hammer,
    - lift hammer,
    - move in world XY so the hammer contacts the block.
  - For Franka-Panda, `check_success()` uses hammer-block contact. The hammer functional point XY is too strict after Franka's held hammer pose changes, while actual contact correctly indicates the beat action.
- Diagnostic:
  - Added `script/diagnose_franka_beat_block_hammer_once.py`.
  - Seed 0 after world XY align:
    - `hammer_block_contact: True`
    - `check_success: True`
- Successful collect command:
  - `CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0 timeout 1200 conda run --no-capture-output -n RoboTwin python -u script/collect_data.py beat_block_hammer demo_franka_1ep`
- Result:
  - Seed search succeeded immediately:
    - `simulate data episode 0 success! (seed = 0)`
  - Data collection completed.
  - Output files:
    - `data/beat_block_hammer/demo_franka_1ep/seed.txt`
    - `data/beat_block_hammer/demo_franka_1ep/_traj_data/episode0.pkl`
    - `data/beat_block_hammer/demo_franka_1ep/data/episode0.hdf5`
    - `data/beat_block_hammer/demo_franka_1ep/video/episode0.mp4`
    - `data/beat_block_hammer/demo_franka_1ep/instructions/episode0.json`
  - HDF5 validation:
    - `episode0.hdf5` size about `57 MB`.
    - `1366` frames in endpose/action/camera streams.
  - Video:
    - `episode0.mp4` saved with `1366` frames at `320x240`, `30 FPS`.
