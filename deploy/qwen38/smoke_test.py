"""Check text, vision, tool calling and Responses on the local model endpoint."""

import argparse
import base64
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18083/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke.json"))
    args = parser.parse_args()
    results = {}

    def post(path, body):
        request = urllib.request.Request(
            args.url.rstrip("/") + path, json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)

    common = {"model": args.model, "max_tokens": 512,
              "chat_template_kwargs": {"enable_thinking": False}}
    results["text"] = post("/chat/completions", dict(common, messages=[
        {"role": "user", "content": "Reply with the word READY."}]))
    assert "READY" in (results["text"]["choices"][0]["message"]["content"] or "")
    encoded = base64.b64encode(args.image.read_bytes()).decode()
    image_url = "data:image/png;base64," + encoded
    results["vision"] = post("/chat/completions", dict(common, messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "Describe the visible objects and robot in this image."},
            {"type": "image_url", "image_url": {"url": image_url}}]}]))
    assert results["vision"]["choices"][0]["message"]["content"]
    results["tools"] = post("/chat/completions", dict(common, messages=[
        {"role": "user", "content": "Call record_observation with label bell. Use the tool."}],
        tools=[{"type": "function", "function": {
            "name": "record_observation", "description": "Record an observation label; no robot action.",
            "parameters": {"type": "object", "properties": {"label": {"type": "string"}},
                           "required": ["label"], "additionalProperties": False}}}], tool_choice="auto"))
    call = results["tools"]["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "record_observation"
    assert json.loads(call["function"]["arguments"]) == {"label": "bell"}
    results["responses"] = post("/responses", {
        "model": args.model, "input": "Reply with the word READY.",
        "max_output_tokens": 512})
    assert results["responses"].get("output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Text, image, tool call and Responses checks passed: {args.output}")


if __name__ == "__main__":
    main()
