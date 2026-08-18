# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from switchyard.cli.dream_ui import _discover_runs


def test_discover_runs_finds_bundles(tmp_path):
    bundle = tmp_path / "dream_labels.jsonl.20260101T000000Z-deadbeef.dream"
    bundle.mkdir()
    (bundle / "dream_run.json").write_text(
        json.dumps({"id": "20260101T000000Z-deadbeef", "log": "x", "out": "y"})
    )

    runs = _discover_runs(root=tmp_path)
    assert len(runs) == 1
    assert runs[0].run_id == bundle.name
    assert runs[0].meta["id"] == "20260101T000000Z-deadbeef"


def test_mining_endpoint_returns_report_when_present(tmp_path):
    from fastapi.testclient import TestClient

    from switchyard.cli.dream_ui import build_app

    bundle = tmp_path / "dream_labels.jsonl.run.dream"
    bundle.mkdir()
    (bundle / "dream_run.json").write_text(json.dumps({"id": "run"}))
    (bundle / "mining_report.json").write_text(json.dumps({"version": 1, "session_count": 2}))

    client = TestClient(build_app(root=tmp_path))
    present = client.get(f"/api/runs/{bundle.name}/mining").json()
    assert present["available"] is True
    assert present["report"]["session_count"] == 2


def test_mining_endpoint_reports_absent(tmp_path):
    from fastapi.testclient import TestClient

    from switchyard.cli.dream_ui import build_app

    bundle = tmp_path / "dream_labels.jsonl.run.dream"
    bundle.mkdir()
    (bundle / "dream_run.json").write_text(json.dumps({"id": "run"}))

    client = TestClient(build_app(root=tmp_path))
    assert client.get(f"/api/runs/{bundle.name}/mining").json() == {"available": False}
