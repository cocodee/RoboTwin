# Repository Guidelines

## Project Structure & Module Organization

RoboTwin is a Python-based bimanual robotic manipulation benchmark. Core task environments live in `envs/`, with shared robot, camera, and utility code under `envs/robot/`, `envs/camera/`, and `envs/utils/`. Task configuration YAML files are in `task_config/`. Data collection and evaluation entry points are in `script/`, with `collect_data.sh` as the main wrapper. Policy baselines are organized under `policy/` (`ACT`, `DP`, `RDT`, `TinyVLA`, `DexVLA`, `GO1`, etc.), each with its own setup files and run scripts. Visual resources belong in `assets/`; task and object language assets live under `description/`. Code-generation utilities are in `code_gen/`.

## Build, Test, and Development Commands

All Python development, testing, and local runs must happen inside the project conda environment:

- `conda create -n myrobotwin python=3.10 -y`: create the required environment.
- `conda activate myrobotwin`: activate it before running repository Python code.
- `bash script/_install.sh`: install the main environment after activation; follow the official docs for system and simulator prerequisites.
- `bash script/_download_assets.sh`: download required simulation assets.
- `bash collect_data.sh beat_block_hammer demo_randomized 0`: collect data for one task/config on GPU `0`.
- `python script/test_render.py`: quick SAPIEN render sanity check.
- `python code_gen/test_gen_code.py`: run the available code-generation test script.
- `bash policy/<PolicyName>/eval.sh`: run a policy-specific evaluation script when provided.

Policy baselines may require extra packages, such as `policy/DexVLA/conda_env.yaml` or `policy/TinyVLA/requirements.txt`; install them only after confirming they are compatible with `myrobotwin`, or document any unavoidable policy-specific environment.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and follow the existing module style. Task environment files use lowercase snake_case names matching task names, for example `envs/beat_block_hammer.py`, with corresponding configs such as `task_config/demo_randomized.yml`. Keep shell scripts bash-compatible and parameterized by task, config, and GPU where possible. Prefer paths relative to the repository root, and avoid committing local absolute paths.

## Testing Guidelines

There is no centralized test runner configured. For simulator-facing changes, run `python script/test_render.py` and at least one representative `collect_data.sh` command. For policy changes, run that policy's eval script and document the environment used. Name new lightweight tests as `test_*.py`, colocated with the relevant module or nearest existing test location.

## Commit & Pull Request Guidelines

Recent history uses short imperative or descriptive commit subjects such as `fix _install.sh` and `Modify PyTorch3D installation command`. Keep commits focused and mention the affected area when useful. Pull requests should include a concise summary, changed tasks/policies/configs, commands run, environment details, and asset or dataset assumptions. Include screenshots or videos for rendering or simulation-behavior changes when useful.

## Security & Configuration Tips

Do not commit downloaded datasets, generated trajectory data, checkpoints, secrets, or machine-specific cache files. Keep large artifacts in documented external storage and reference their expected paths in the PR description.
