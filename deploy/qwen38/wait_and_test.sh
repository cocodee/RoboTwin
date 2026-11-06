#!/usr/bin/env bash
# Wait for local model readiness, then check the interfaces required by the agent.
set -euo pipefail
deployment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
camera_image=${1:?Provide a saved camera PNG}
cd "$deployment_dir"
for readiness_attempt in $(seq 1 480); do
    if curl --noproxy '*' --silent --fail --max-time 3 \
        "http://127.0.0.1:${MODEL_PORT:-18083}/health" > /dev/null; then
        NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
            conda run --no-capture-output -n myrobotwin python smoke_test.py \
            --url "http://127.0.0.1:${MODEL_PORT:-18083}/v1" --image "$camera_image"
        exit $?
    fi
    sleep 15
done
echo "Model was not ready within the two-hour readiness window" >&2
exit 1
