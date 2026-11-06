# Qwen3.8-27B for the RoboTwin agent

The selected model is `Qwen/Qwen3.8-27B`, with BF16 weights, vision and tool
calling enabled. Revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
This substitutes the model used in the paper; it is not a replication of its
Flash-Next result. The current robot task is only `click_bell`.

The service uses one H100 80GB. The selected GPU UUID and local image digest
are in ignored `.env`; defaults use GPU 7 and the existing vLLM 0.21.0 image.
No CPU weight offloading is configured. Context is 64K, with at most
two concurrent sequences. Validate actual memory use after startup.

The endpoint is `http://127.0.0.1:18083/v1`, model ID `qwen3.8-27b`.
Host networking makes the server's HTTP/HTTPS proxies reachable. Do not forward
a SOCKS `ALL_PROXY` into an image without SOCKS transport dependencies.
The default cache lives outside the repository at
`my_robotwin/model-cache/huggingface`; override `MODEL_CACHE_DIR` if needed.
Do not commit credentials, downloaded weights or generated artifacts.

```bash
cd deploy/qwen38
docker compose config --quiet
docker compose up -d
docker compose logs -f --tail 30 model
curl --noproxy '*' http://127.0.0.1:18083/health
curl --noproxy '*' http://127.0.0.1:18083/v1/models
```

Check text, RGB input, parsed tool calls and Responses using a saved camera PNG:

```bash
conda run -n myrobotwin python smoke_test.py --image /path/to/camera.png
```

The tool-call check never operates a robot. Connect uni-robot-agent only after
these checks pass. Use its `vllm.chat_completions` provider profile: the full
Codex request needs developer-role conversion, merged leading system messages,
tool namespace conversion, and vLLM reasoning aliases. Basic `/responses` smoke
success alone does not validate that full request. The image context estimator
must budget decoded images rather than count Base64 characters as text.

After the weights are fully cached, `MODEL_OFFLINE=1` avoids network access on
restart. Use `MODEL_OFFLINE=0` for the first download or a revision change.
`MODEL_MAX_LEN` overrides the default 65536-token context.

Stop with `docker compose stop`. Other services are outside this Compose project.

References:

- [Official model](https://huggingface.co/Qwen/Qwen3.8-27B)
- [uni-robot-agent paper, v2](https://arxiv.org/html/2611.01560v2)
