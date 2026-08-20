# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage for dream-emitted learned-selection weights."""

import argparse
import json
import threading
import tomllib
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from switchyard.dream import cmd_dream


class _UpstreamHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length))
        model = request["model"]
        content = (
            '{"decision":{"target":"premium"}}'
            if model == "model/classifier"
            else "ok"
        )
        body = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


def test_dream_weights_drive_custom_classifier_selection(tmp_path: Path) -> None:
    bundle = tmp_path / "synthetic.dream"
    bundle.mkdir()
    routing_log = bundle / "routing.jsonl"
    routing_log.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in [
                {"model": "model/weak", "reward": 1.0},
                {"model": "model/weak", "reward": 1.0},
                {"model": "model/premium", "reward": 0.0},
                {"model": "model/premium", "reward": 0.0},
            ]
        )
    )
    weights = bundle / "weights.toml"
    cmd_dream(
        argparse.Namespace(
            log=routing_log,
            out=bundle / "labels.jsonl",
            emit_weights=weights,
            label=["weak=model/weak", "premium=model/premium"],
            mine=False,
            emit_skills=False,
            emit_tools=False,
            transcript=None,
            strong_model=None,
            base_url="https://example.invalid/v1",
            api_key=None,
        )
    )

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    config = tmp_path / "routes.toml"
    config.write_text(
        f'''schema_version = 1

[llm_clients.upstream]
format = "openai_chat"
base_url = "http://127.0.0.1:{upstream.server_port}/v1"

[targets.classifier]
id = "model/classifier"
llm_client = "upstream"

[targets.weak]
id = "model/weak"
llm_client = "upstream"

[targets.premium]
id = "model/premium"
llm_client = "upstream"

[routes.auto]
id = "switchyard/auto"
type = "llm_classifier"
mode = "custom"
classifier_target = "classifier"
targets = ["weak", "premium"]
default_target = "premium"
prompt = "Select a target"
response_schema = """{{"type":"object","properties":{{"decision":{{"type":"object","properties":{{"target":{{"type":"string","enum":["weak","premium"]}}}},"required":["target"],"additionalProperties":false}}}},"required":["decision"],"additionalProperties":false}}"""

[routes.auto.policy]
type = "target_selector"
selector = "/decision/target"

[routes.auto.learned_selection]
weights_path = {json.dumps(str(weights))}
'''
    )

    try:
        from switchyard_rust.server import Server

        with Server(config) as server:
            request = urllib.request.Request(
                f"{server.base_url}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "switchyard/auto",
                        "messages": [{"role": "user", "content": "route this"}],
                    }
                ).encode(),
                headers={"content-type": "application/json"},
            )
            with urllib.request.urlopen(request) as response:
                assert response.headers["x-model-router-selected-model"] == "model/weak"
    finally:
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join()


def test_dream_classifier_artifact_loads_on_auto_route(tmp_path: Path) -> None:
    bundle = tmp_path / "synthetic.dream"
    bundle.mkdir()
    routing_log = bundle / "routing.jsonl"
    routing_log.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in [
                {"task": "route this", "model": "model/premium", "reward": 1.0},
                {"task": "route this", "model": "model/premium", "reward": 1.0},
                {"task": "route this", "model": "model/weak", "reward": 0.0},
            ]
        )
    )
    classifier_route = bundle / "classifier_route.toml"
    run_bundle = cmd_dream(
        argparse.Namespace(
            log=routing_log,
            out=bundle / "labels.jsonl",
            emit_weights=None,
            emit_classifier=classifier_route,
            label=["weak=model/weak", "premium=model/premium"],
            mine=False,
            emit_skills=False,
            emit_tools=False,
            transcript=None,
            strong_model=None,
            base_url="https://example.invalid/v1",
            api_key=None,
        )
    )

    # The emitted dataset labels the observed-best target per task.
    dataset = [
        json.loads(line)
        for line in (run_bundle / "classifier_dataset.jsonl").read_text().splitlines()
    ]
    assert dataset == [{"text": "route this", "target": "premium", "stats": dataset[0]["stats"]}]

    # Overlay the emitted prompt/schema onto a routes.auto target_selector route and
    # confirm the native server accepts and serves it.
    overlay = tomllib.loads(classifier_route.read_text())["routes"]["auto"]

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    config = tmp_path / "routes.toml"
    config.write_text(
        f'''schema_version = 1

[llm_clients.upstream]
format = "openai_chat"
base_url = "http://127.0.0.1:{upstream.server_port}/v1"

[targets.classifier]
id = "model/classifier"
llm_client = "upstream"

[targets.weak]
id = "model/weak"
llm_client = "upstream"

[targets.premium]
id = "model/premium"
llm_client = "upstream"

[routes.auto]
id = "switchyard/auto"
type = "llm_classifier"
mode = "custom"
classifier_target = "classifier"
targets = ["weak", "premium"]
default_target = "premium"
prompt = {json.dumps(overlay["prompt"])}
response_schema = {json.dumps(overlay["response_schema"])}

[routes.auto.policy]
type = "target_selector"
selector = "/decision/target"
'''
    )

    try:
        from switchyard_rust.server import Server

        with Server(config) as server:
            request = urllib.request.Request(
                f"{server.base_url}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "switchyard/auto",
                        "messages": [{"role": "user", "content": "route this"}],
                    }
                ).encode(),
                headers={"content-type": "application/json"},
            )
            with urllib.request.urlopen(request) as response:
                assert response.headers["x-model-router-selected-model"] == "model/premium"
    finally:
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join()
