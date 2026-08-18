# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local web UI for exploring `switchyard dream` runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DreamRun:
    run_id: str
    bundle_dir: Path
    meta: dict[str, Any]


def _discover_runs(*, root: Path) -> list[DreamRun]:
    if not root.exists():
        return []

    runs = []
    for meta_path in sorted(root.glob("*.dream/dream_run.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        bundle_dir = meta_path.parent
        run_id = bundle_dir.name
        runs.append(DreamRun(run_id=run_id, bundle_dir=bundle_dir, meta=meta))
    return list(reversed(runs))


def serve_dream_ui(*, host: str, port: int, initial_bundle: Path) -> int:
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "dream --ui requires nemo-switchyard[cli] with fastapi+uvicorn installed"
        ) from exc

    app = build_app(root=initial_bundle.resolve().parent)
    print(f"dream ui: http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


def build_app(*, root: Path):  # type: ignore[no-untyped-def]
    """Build the Dream UI FastAPI application rooted at a bundle directory."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

    app = FastAPI(title="Switchyard Dream")

    @app.get("/api/runs")
    def runs() -> JSONResponse:
        payload = [
            {
                "id": run.meta.get("id") or run.run_id,
                "bundle_dir": str(run.bundle_dir),
                "created_at": run.meta.get("created_at"),
                "log": run.meta.get("log"),
                "out": run.meta.get("out"),
            }
            for run in _discover_runs(root=root)
        ]
        return JSONResponse(payload)

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> JSONResponse:
        bundle_dir = root / run_id
        meta_path = bundle_dir / "dream_run.json"
        if not meta_path.is_file():
            raise HTTPException(status_code=404, detail="run not found")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        labels_count = None
        labels_path = bundle_dir / "labels.jsonl"
        if labels_path.is_file():
            try:
                labels_count = sum(1 for _ in labels_path.read_text(encoding="utf-8").splitlines() if _.strip())
            except Exception:
                labels_count = None
        else:
            out_path = meta.get("out")
            if out_path:
                try:
                    labels_count = sum(
                        1 for _ in Path(out_path).read_text(encoding="utf-8").splitlines() if _.strip()
                    )
                except Exception:
                    labels_count = None

        return JSONResponse(
            {
                "id": run_id,
                "bundle_dir": str(bundle_dir),
                "meta": meta,
                "labels_count": labels_count,
            }
        )

    @app.get("/api/runs/{run_id}/labels")
    def labels(run_id: str, offset: int = 0, limit: int = 200) -> JSONResponse:
        bundle_dir = root / run_id
        meta_path = bundle_dir / "dream_run.json"
        if not meta_path.is_file():
            raise HTTPException(status_code=404, detail="run not found")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        labels_path = bundle_dir / "labels.jsonl"
        if not labels_path.is_file():
            out_path = meta.get("out")
            if not out_path:
                return JSONResponse({"items": [], "next_offset": None})
            labels_path = Path(out_path)

        if not labels_path.is_file():
            raise HTTPException(status_code=404, detail="labels file not found")

        lines = labels_path.read_text(encoding="utf-8").splitlines()
        items = []
        end = min(offset + limit, len(lines))
        for line in lines[offset:end]:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue

        next_offset = end if end < len(lines) else None
        return JSONResponse({"items": items, "next_offset": next_offset})

    @app.get("/api/runs/{run_id}/mining")
    def mining(run_id: str) -> JSONResponse:
        bundle_dir = root / run_id
        report_path = bundle_dir / "mining_report.json"
        if not report_path.is_file():
            return JSONResponse({"available": False})
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return JSONResponse({"available": True, "report": report})

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(
            """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Switchyard Dream</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system; margin: 16px; }
    .row { display: flex; gap: 16px; }
    .col { flex: 1; min-width: 280px; }
    select, input { width: 100%; padding: 8px; }
    pre { background: #0b1020; color: #d6e1ff; padding: 12px; overflow: auto; }
    .meta { color: #444; font-size: 12px; }
    button { padding: 8px 12px; }
  </style>
</head>
<body>
  <h2>Switchyard Dream</h2>
  <div class="row">
    <div class="col">
      <div class="meta">Runs discovered next to the initial --out bundle.</div>
      <select id="runs"></select>
      <button id="load">Load</button>
      <div id="runmeta" class="meta"></div>
      <h3>Tool-call mining</h3>
      <pre id="mining"></pre>
    </div>
    <div class="col">
      <div class="meta">Filter (substring match on task)</div>
      <input id="filter" placeholder="e.g. test" />
      <pre id="labels"></pre>
      <button id="more">More</button>
    </div>
  </div>
<script>
let nextOffset = 0;
let currentRun = null;
let all = [];

function esc(s) { return (""+s).replaceAll("<","&lt;").replaceAll(">","&gt;"); }

async function refreshRuns() {
  const res = await fetch('/api/runs');
  const runs = await res.json();
  const sel = document.getElementById('runs');
  sel.innerHTML = '';
  for (const r of runs) {
    const opt = document.createElement('option');
    opt.value = r.id;
    opt.textContent = r.id;
    sel.appendChild(opt);
  }
  if (runs.length) {
    sel.value = runs[0].id;
  }
}

async function loadRun() {
  const sel = document.getElementById('runs');
  currentRun = sel.value;
  nextOffset = 0;
  all = [];
  document.getElementById('labels').textContent = '';

  const metaRes = await fetch('/api/runs/' + encodeURIComponent(currentRun));
  const meta = await metaRes.json();
  document.getElementById('runmeta').textContent = JSON.stringify(meta, null, 2);
  const miningRes = await fetch('/api/runs/' + encodeURIComponent(currentRun) + '/mining');
  const mining = await miningRes.json();
  document.getElementById('mining').textContent = mining.available
    ? JSON.stringify(mining.report, null, 2)
    : 'No mining report for this run.';
  await loadMore();
}

function render() {
  const q = document.getElementById('filter').value.toLowerCase();
  const filtered = q ? all.filter(x => (x.task || '').toLowerCase().includes(q)) : all;
  document.getElementById('labels').textContent = JSON.stringify(filtered, null, 2);
}

async function loadMore() {
  if (!currentRun) return;
  const url = '/api/runs/' + encodeURIComponent(currentRun) + '/labels?offset=' + nextOffset + '&limit=200';
  const res = await fetch(url);
  const payload = await res.json();
  all = all.concat(payload.items || []);
  nextOffset = payload.next_offset;
  render();
}

document.getElementById('load').addEventListener('click', loadRun);
document.getElementById('more').addEventListener('click', loadMore);
document.getElementById('filter').addEventListener('input', render);

refreshRuns().then(loadRun);
</script>
</body>
</html>"""
        )

    @app.get("/health")
    def health() -> PlainTextResponse:
        return PlainTextResponse("ok")

    return app
