#!/usr/bin/env bash
# Run one fresh click_bell episode; never launch the task suite.
set -euo pipefail
ROBOTWIN_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
AGENT_PYTHON=${AGENT_PYTHON:-"$ROBOTWIN_ROOT/../agent-env/bin/python"}
NATIVE_PYTHON=${NATIVE_PYTHON:-"${CONDA_PREFIX:?Activate myrobotwin or set NATIVE_PYTHON}/bin/python"}
MODEL_URL=${MODEL_URL:-http://127.0.0.1:18083/v1}
MODEL_NAME=${MODEL_NAME:-qwen3.8-27b}
ENV_PORT=${ENV_PORT:-18780}
ENV_URL="http://127.0.0.1:$ENV_PORT"
RUN_DIR=${RUN_DIR:-"$ROBOTWIN_ROOT/diagnostics/click_bell_$(date -u +%Y%m%dT%H%M%SZ)_$$"}
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export UNI_ROBOT_AGENT_API_KEY=${UNI_ROBOT_AGENT_API_KEY:-local}
test -x "$AGENT_PYTHON"
test -x "$NATIVE_PYTHON"
curl --noproxy '*' --fail --silent --max-time 10 "$MODEL_URL/models" >/dev/null
if curl --noproxy '*' --silent --max-time 2 "$ENV_URL/info" >/dev/null; then
    echo "Environment port $ENV_PORT is occupied; choose ENV_PORT for a fresh episode." >&2
    exit 1
fi
mkdir "$RUN_DIR"
env_pid=
cleanup() {
    if [[ -n "$env_pid" ]]; then
        kill -TERM "$env_pid" 2>/dev/null || true
        wait "$env_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
CUDA_VISIBLE_DEVICES=${SIM_GPU:-1} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0} \
    "$AGENT_PYTHON" -m uni_robot_agent.integrations.robotwin.server \
    --native-python "$NATIVE_PYTHON" --native-root "$ROBOTWIN_ROOT" \
    --seed "${SEED:-0}" --port "$ENV_PORT" --output-dir "$RUN_DIR/environment" \
    >"$RUN_DIR/environment.log" 2>&1 &
env_pid=$!
ready=false
for ((attempt=0; attempt<180; attempt++)); do
    if ! kill -0 "$env_pid" 2>/dev/null; then
        tail -40 "$RUN_DIR/environment.log" >&2
        exit 1
    fi
    if curl --noproxy '*' --fail --silent --max-time 2 "$ENV_URL/info" >/dev/null; then
        ready=true
        break
    fi
    sleep 1
done
if [[ "$ready" != true ]]; then
    echo "Environment startup timed out; see $RUN_DIR/environment.log" >&2
    exit 1
fi
echo "Running one click_bell episode. Artifacts: $RUN_DIR"
perception_args=(--no-sam3)
if [[ -n "${SAM3_URL:-}" ]]; then
    perception_args=(--sam3-url "$SAM3_URL")
fi
"$AGENT_PYTHON" -m uni_robot_agent run \
    --integration robotwin --task click_bell --seed "${SEED:-0}" \
    --agent-python "$AGENT_PYTHON" --model "$MODEL_NAME" --reasoning-effort "${REASONING_EFFORT:-low}" \
    --provider-url "$MODEL_URL" --provider-profile vllm.chat_completions \
    --env-url "$ENV_URL" "${perception_args[@]}" --context-policy sliding_window \
    --tool-return-mode "${TOOL_RETURN_MODE:-immediate}" \
    --context-max-input-tokens 42000 --context-target-input-tokens 34000 \
    --context-keep-recent-blocks 1 --context-image-eviction-batch-size 3 \
    --episode-timeout-s "${EPISODE_TIMEOUT_S:-3600}" --output-dir "$RUN_DIR/agent" \
    2>&1 | tee "$RUN_DIR/agent.log"
