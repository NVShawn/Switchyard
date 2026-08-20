# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from switchyard.cli.dream_ui import _build_summary, _discover_runs


def _bundle(tmp_path):
    bundle = tmp_path / "dream_labels.jsonl.20260101T000000Z-deadbeef.dream"
    bundle.mkdir()
    (bundle / "dream_run.json").write_text(
        json.dumps(
            {
                "id": "20260101T000000Z-deadbeef",
                "created_at": "2026-01-01T00:00:00+00:00",
                "log": "x",
                "out": "y",
            }
        )
    )
    return bundle


def test_discover_runs_finds_bundles(tmp_path):
    bundle = _bundle(tmp_path)

    runs = _discover_runs(root=tmp_path)
    assert len(runs) == 1
    assert runs[0].run_id == bundle.name
    assert runs[0].meta["id"] == "20260101T000000Z-deadbeef"

    from fastapi.testclient import TestClient

    from switchyard.cli.dream_ui import build_app

    listed = TestClient(build_app(root=tmp_path)).get("/api/runs").json()
    assert listed[0]["id"] == bundle.name
    assert listed[0]["run_id"] == "20260101T000000Z-deadbeef"


def test_summary_aggregates_routing_and_mining_metrics(tmp_path):
    bundle = _bundle(tmp_path)
    records = [
        {
            "route_id": "auto",
            "model": "nano",
            "token_bucket": "small",
            "success": True,
            "reward": 0.8,
            "cost_usd": 0.01,
            "latency_ms": 100,
            "prompt_tokens": 10,
            "cached_tokens": 2,
            "completion_tokens": 3,
            "judge_p_solve": 0.9,
        },
        {
            "route_id": "auto",
            "model": "nano",
            "token_bucket": "small",
            "success": False,
            "reward": 0.0,
            "cost_usd": 0.02,
            "latency_ms": 300,
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "reasoning_tokens": 4,
            "judge_p_solve": 0.2,
        },
        {"route_id": "auto", "model": "judge", "tier": "classifier", "success": True},
    ]
    (bundle / "routing.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    intent = {"kind": "bash_search", "tool": "bash", "command": "rg needle src"}
    (bundle / "mining_report.json").write_text(
        json.dumps(
            {
                "session_count": 1,
                "intent_count": 3,
                "exact_duplicate_count": 1,
                "sessions": [
                    {
                        "session_id": "s1",
                        "duplicate_intents": [{"intent": intent, "count": 2}],
                    }
                ],
            }
        )
    )

    summary = _build_summary(bundle, {"id": "run", "created_at": "now"})
    assert summary["coverage"] == {
        "records": 3,
        "answer_calls": 2,
        "with_cost": 2,
        "with_latency": 2,
        "with_judge": 2,
    }
    assert summary["totals"]["success_rate"] == 0.5
    assert summary["totals"]["mean_reward"] == 0.4
    assert summary["totals"]["cost_usd"] == (
        (8 * 1.0 + 2 * 0.1 + 3 * 4.0) + (20 * 1.0 + 5 * 4.0)
    ) / 1_000_000
    assert summary["totals"]["p95_latency_ms"] == 300
    assert summary["arms"][0]["calls"] == 2
    assert summary["models"][0]["tokens"] == {
        "prompt": 30,
        "cached": 2,
        "completion": 8,
        "reasoning": 4,
    }
    assert summary["selector"] == [
        {
            "route_id": "auto",
            "answer_calls": 2,
            "classifier_calls": 1,
            "targets": [
                {
                    "target": "nano",
                    "calls": 2,
                    "share": 1.0,
                    "success_rate": 0.5,
                    "mean_reward": 0.4,
                    "p95_latency_ms": 300,
                    "buckets": {"small": 2},
                }
            ],
        }
    ]
    assert summary["calibration"]["brier"] == 0.025
    assert summary["mining"]["top"][0]["excess"] == 1


def test_trends_endpoint_aggregates_selector_attribution(tmp_path):
    from fastapi.testclient import TestClient

    from switchyard.cli.dream_ui import build_app

    bundle = _bundle(tmp_path)
    records = [
        {"ts": "2026-01-01T01:15:00Z", "route_id": "auto", "model": "nano"},
        {"ts": "2026-01-01T01:45:00Z", "route_id": "auto", "model": "nano"},
        {"ts": "2026-01-01T02:00:00Z", "route_id": "auto", "model": "large"},
        {
            "ts": "2026-01-01T02:00:00Z",
            "route_id": "auto",
            "model": "judge",
            "tier": "classifier",
        },
    ]
    (bundle / "routing.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )

    client = TestClient(build_app(root=tmp_path))
    response = client.get("/api/trends?period=hourly")

    assert response.status_code == 200
    assert response.json() == {
        "period": "hourly",
        "points": [
            {
                "start": "2026-01-01T01:00:00Z",
                "calls": 2,
                "routes": [
                    {
                        "route_id": "auto",
                        "calls": 2,
                        "targets": [{"target": "nano", "calls": 2, "share": 1.0}],
                    }
                ],
            },
            {
                "start": "2026-01-01T02:00:00Z",
                "calls": 1,
                "routes": [
                    {
                        "route_id": "auto",
                        "calls": 1,
                        "targets": [{"target": "large", "calls": 1, "share": 1.0}],
                    }
                ],
            },
        ],
    }
    assert client.get("/api/trends?period=monthly").status_code == 422


def test_summary_endpoint_and_dashboard_render(tmp_path):
    from fastapi.testclient import TestClient

    from switchyard.cli.dream_ui import build_app

    bundle = _bundle(tmp_path)
    (bundle / "routing.jsonl").write_text(
        json.dumps({"model": "nano", "success": True, "reward": 1.0}) + "\n"
    )

    client = TestClient(build_app(root=tmp_path))
    summary = client.get(f"/api/runs/{bundle.name}/summary")
    assert summary.status_code == 200
    assert summary.json()["coverage"]["answer_calls"] == 1

    page = client.get("/")
    assert page.status_code == 200
    assert "Routing Observatory" in page.text
    assert "Arm performance" in page.text
    assert "Cost / latency frontier" in page.text
    assert "Selector attribution trends" in page.text
    assert "Judge calibration" in page.text
    assert "custom target selectors do not emit probabilities" in page.text
    assert "Tool-call churn opportunities" in page.text


def test_summary_reports_missing_mining_data(tmp_path):
    bundle = _bundle(tmp_path)
    summary = _build_summary(bundle, {"id": "run"})
    assert summary["mining"] == {
        "available": False,
        "sessions": 0,
        "intents": 0,
        "exact_duplicates": 0,
        "top": [],
    }
