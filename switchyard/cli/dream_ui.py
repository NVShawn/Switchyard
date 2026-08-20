# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local web UI for exploring `switchyard dream` runs."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_CLASSIFIER_TIER = "classifier"


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
        runs.append(DreamRun(run_id=bundle_dir.name, bundle_dir=bundle_dir, meta=meta))
    return list(reversed(runs))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# Cache-aware unit prices in USD per 1M tokens, keyed by target model id.
#
# The routing log's ``cost_usd`` bills every input token (including cache reads)
# at the full input rate, which massively over-counts cache-heavy coding-agent
# traffic. We recompute from token counts here so historical and new records
# reflect the backend's real pricing.
#
# Rates were back-calculated from the LiteLLM proxy's own ``x-litellm-response-
# cost-original`` header by solving cost = prompt*input + completion*output over
# calls with differing output lengths. Cache economics were measured the same
# way against a repeated Opus prompt: cache reads bill at a tenth of the input
# rate and cache writes at 1.25x the input rate. Self-hosted NIMs (nemotron,
# deepseek) report $0 per token (billed as GPU-hours). ``gpt-5.6-sol`` and
# ``kimi-k3-max-preview`` return no cost header, so they keep estimated rates.
_CACHE_READ_MULTIPLIER = 0.10
_CACHE_WRITE_MULTIPLIER = 1.25
_DEFAULT_PRICE = (1.0, 4.0)
_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "aws/anthropic/bedrock-claude-opus-4-8": (5.0, 25.0),
    "azure/openai/gpt-5.2": (1.75, 14.0),
    "azure/openai/gpt-5.4-nano": (0.2, 1.25),
    "nvidia/zai-org/glm-5.2": (0.96, 3.02),
    "nvidia/nvidia/nemotron-nano-31b-v3": (0.0, 0.0),
    "nvidia/nvidia/nemotron-3-ultra": (0.0, 0.0),
    "nvidia/nvidia/nemotron-3.5-lightning": (0.0, 0.0),
    "nvidia/nvidia/Nemotron-3-Nano-30B-A3B": (0.0, 0.0),
    "nvidia/deepseek-ai/deepseek-v4-pro": (0.0, 0.0),
    # No cost header from the proxy; estimated pending real rates.
    "azure/openai/gpt-5.6-sol": (5.0, 25.0),
    "nvidia/moonshotai/kimi-k3-max-preview": (3.5, 14.0),
}


def _record_cost(record: dict[str, Any]) -> float | None:
    """Recompute a record's USD cost with backend-measured cache pricing.

    Returns ``None`` when the record carries no token counts to price.
    ``prompt_tokens`` already includes cached and cache-creation tokens
    (see ``token_usage`` in the server), so non-cached input is the remainder.
    Cache reads bill at ``_CACHE_READ_MULTIPLIER`` and cache writes at
    ``_CACHE_WRITE_MULTIPLIER`` of the model's input rate.
    """
    prompt = record.get("prompt_tokens")
    completion = record.get("completion_tokens")
    if not isinstance(prompt, (int, float)) or not isinstance(completion, (int, float)):
        return None
    cached = int(record.get("cached_tokens") or 0)
    cache_write = int(record.get("cache_creation_tokens") or 0)
    non_cached = max(int(prompt) - cached - cache_write, 0)
    input_rate, output_rate = _MODEL_PRICES.get(str(record.get("model") or ""), _DEFAULT_PRICE)
    return (
        non_cached * input_rate
        + cached * input_rate * _CACHE_READ_MULTIPLIER
        + cache_write * input_rate * _CACHE_WRITE_MULTIPLIER
        + int(completion) * output_rate
    ) / 1_000_000.0


def _record_costs(records: list[dict[str, Any]]) -> list[float]:
    return [cost for record in records if (cost := _record_cost(record)) is not None]


def _aggregate_mining(bundle_dir: Path) -> dict[str, Any]:
    report_path = bundle_dir / "mining_report.json"
    if not report_path.is_file():
        return {"available": False, "sessions": 0, "intents": 0, "exact_duplicates": 0, "top": []}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"available": False, "sessions": 0, "intents": 0, "exact_duplicates": 0, "top": []}

    grouped: dict[str, dict[str, Any]] = {}
    for session in report.get("sessions", []):
        for duplicate in session.get("duplicate_intents", []):
            intent = duplicate.get("intent", {})
            key = json.dumps(intent, sort_keys=True, separators=(",", ":"))
            item = grouped.setdefault(
                key,
                {"intent": intent, "sessions": 0, "occurrences": 0, "excess": 0},
            )
            count = int(duplicate.get("count", 0))
            item["sessions"] += 1
            item["occurrences"] += count
            item["excess"] += max(count - 1, 0)
    top = sorted(grouped.values(), key=lambda item: (-item["excess"], -item["occurrences"]))[:12]
    return {
        "available": True,
        "sessions": int(report.get("session_count", 0)),
        "intents": int(report.get("intent_count", 0)),
        "exact_duplicates": int(report.get("exact_duplicate_count", 0)),
        "top": top,
    }


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _trend_bucket(timestamp: datetime, period: str) -> datetime:
    timestamp = timestamp.astimezone(timezone.utc)
    if period == "hourly":
        return timestamp.replace(minute=0, second=0, microsecond=0)
    if period == "daily":
        return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    return timestamp.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    ) - timedelta(days=timestamp.weekday())


def _build_trends(*, root: Path, period: str) -> dict[str, Any]:
    buckets: dict[datetime, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for run in _discover_runs(root=root):
        fallback_timestamp = _parse_timestamp(run.meta.get("created_at"))
        for record in _read_jsonl(run.bundle_dir / "routing.jsonl"):
            route_id = record.get("route_id")
            if record.get("tier") == _CLASSIFIER_TIER or not isinstance(route_id, str) or not route_id:
                continue
            timestamp = _parse_timestamp(record.get("ts")) or fallback_timestamp
            if timestamp is None:
                continue
            target = str(record.get("model") or "unknown")
            buckets[_trend_bucket(timestamp, period)][route_id][target] += 1

    points = []
    for start, routes in sorted(buckets.items()):
        route_items: list[dict[str, Any]] = []
        for route_id, targets in sorted(routes.items()):
            calls = sum(targets.values())
            route_items.append(
                {
                    "route_id": route_id,
                    "calls": calls,
                    "targets": [
                        {"target": target, "calls": target_calls, "share": target_calls / calls}
                        for target, target_calls in sorted(targets.items())
                    ],
                }
            )
        points.append(
            {
                "start": start.isoformat().replace("+00:00", "Z"),
                "calls": sum(route["calls"] for route in route_items),
                "routes": route_items,
            }
        )
    return {"period": period, "points": points}


def _build_summary(bundle_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    records = _read_jsonl(bundle_dir / "routing.jsonl")
    answers = [record for record in records if record.get("tier") != _CLASSIFIER_TIER]
    completed = [record for record in answers if isinstance(record.get("success"), bool)]

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    model_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in answers:
        model = str(record.get("model") or "unknown")
        bucket = str(record.get("token_bucket") or "unknown")
        groups[(model, bucket)].append(record)
        model_groups[model].append(record)

    arms: list[dict[str, Any]] = []
    for (model, bucket), group in groups.items():
        successes = [record["success"] for record in group if isinstance(record.get("success"), bool)]
        rewards = [float(record["reward"]) for record in group if isinstance(record.get("reward"), (int, float))]
        costs = _record_costs(group)
        latencies = [float(record["latency_ms"]) for record in group if isinstance(record.get("latency_ms"), (int, float))]
        arms.append(
            {
                "model": model,
                "token_bucket": bucket,
                "calls": len(group),
                "success_rate": sum(successes) / len(successes) if successes else 0.0,
                "mean_reward": _mean(rewards),
                "cost_usd": sum(costs),
                "mean_cost_usd": _mean(costs),
                "p50_latency_ms": _percentile(latencies, 0.50),
                "p95_latency_ms": _percentile(latencies, 0.95),
            }
        )
    arms.sort(key=lambda arm: (-int(arm["calls"]), str(arm["model"]), str(arm["token_bucket"])))

    models: list[dict[str, Any]] = []
    for model, group in model_groups.items():
        successes = [record["success"] for record in group if isinstance(record.get("success"), bool)]
        costs = _record_costs(group)
        latencies = [float(record["latency_ms"]) for record in group if isinstance(record.get("latency_ms"), (int, float))]
        models.append(
            {
                "model": model,
                "calls": len(group),
                "success_rate": sum(successes) / len(successes) if successes else 0.0,
                "cost_usd": sum(costs),
                "mean_cost_usd": _mean(costs),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "tokens": {
                    "prompt": sum(int(record.get("prompt_tokens") or 0) for record in group),
                    "cached": sum(int(record.get("cached_tokens") or 0) for record in group),
                    "completion": sum(int(record.get("completion_tokens") or 0) for record in group),
                    "reasoning": sum(int(record.get("reasoning_tokens") or 0) for record in group),
                },
            }
        )
    models.sort(key=lambda model: (-int(model["calls"]), str(model["model"])))

    selector_routes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in answers:
        route_id = record.get("route_id")
        if isinstance(route_id, str) and route_id:
            selector_routes[route_id].append(record)
    classifier_by_route: dict[str, int] = defaultdict(int)
    for record in records:
        route_id = record.get("route_id")
        if record.get("tier") == _CLASSIFIER_TIER and isinstance(route_id, str) and route_id:
            classifier_by_route[route_id] += 1

    selector: list[dict[str, Any]] = []
    for route_id, route_records in selector_routes.items():
        targets: list[dict[str, Any]] = []
        by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in route_records:
            by_target[str(record.get("model") or "unknown")].append(record)
        for target, target_records in by_target.items():
            target_successes = [
                record["success"]
                for record in target_records
                if isinstance(record.get("success"), bool)
            ]
            target_rewards = [
                float(record["reward"])
                for record in target_records
                if isinstance(record.get("reward"), (int, float))
            ]
            target_latencies = [
                float(record["latency_ms"])
                for record in target_records
                if isinstance(record.get("latency_ms"), (int, float))
            ]
            buckets: dict[str, int] = defaultdict(int)
            for record in target_records:
                buckets[str(record.get("token_bucket") or "unknown")] += 1
            targets.append(
                {
                    "target": target,
                    "calls": len(target_records),
                    "share": len(target_records) / len(route_records),
                    "success_rate": (
                        sum(target_successes) / len(target_successes) if target_successes else 0.0
                    ),
                    "mean_reward": _mean(target_rewards),
                    "p95_latency_ms": _percentile(target_latencies, 0.95),
                    "buckets": dict(sorted(buckets.items())),
                }
            )
        targets.sort(key=lambda target: (-int(str(target["calls"])), str(target["target"])))
        selector.append(
            {
                "route_id": route_id,
                "answer_calls": len(route_records),
                "classifier_calls": classifier_by_route.get(route_id, 0),
                "targets": targets,
            }
        )
    selector.sort(key=lambda route: (-int(str(route["answer_calls"])), str(route["route_id"])))

    calibration_pairs = [
        (float(record["judge_p_solve"]), bool(record["success"]))
        for record in answers
        if isinstance(record.get("judge_p_solve"), (int, float))
        and isinstance(record.get("success"), bool)
    ]
    bins = []
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        values = [(prediction, success) for prediction, success in calibration_pairs if lower <= prediction <= upper and (index == 9 or prediction < upper)]
        if values:
            bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": len(values),
                    "mean_prediction": _mean([prediction for prediction, _ in values]),
                    "observed_success": _mean([1.0 if success else 0.0 for _, success in values]),
                }
            )
    brier = _mean([(prediction - (1.0 if success else 0.0)) ** 2 for prediction, success in calibration_pairs])

    successes = [record["success"] for record in completed]
    rewards = [float(record["reward"]) for record in answers if isinstance(record.get("reward"), (int, float))]
    costs = _record_costs(answers)
    latencies = [float(record["latency_ms"]) for record in answers if isinstance(record.get("latency_ms"), (int, float))]
    return {
        "run": {"id": meta.get("id"), "created_at": meta.get("created_at")},
        "coverage": {
            "records": len(records),
            "answer_calls": len(answers),
            "with_cost": len(costs),
            "with_latency": len(latencies),
            "with_judge": len(calibration_pairs),
        },
        "totals": {
            "success_rate": sum(successes) / len(successes) if successes else 0.0,
            "mean_reward": _mean(rewards),
            "cost_usd": sum(costs),
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p95_latency_ms": _percentile(latencies, 0.95),
        },
        "arms": arms,
        "models": models,
        "selector": selector,
        "calibration": {"brier": brier, "bins": bins},
        "mining": _aggregate_mining(bundle_dir),
    }


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

    def bundle(run_id: str) -> tuple[Path, dict[str, Any]]:
        bundle_dir = root / run_id
        meta_path = bundle_dir / "dream_run.json"
        if not meta_path.is_file():
            raise HTTPException(status_code=404, detail="run not found")
        return bundle_dir, json.loads(meta_path.read_text(encoding="utf-8"))

    @app.get("/api/runs")
    def runs() -> JSONResponse:
        return JSONResponse(
            [
                {
                    "id": run.run_id,
                    "run_id": run.meta.get("id"),
                    "created_at": run.meta.get("created_at"),
                }
                for run in _discover_runs(root=root)
            ]
        )

    @app.get("/api/trends")
    def trends(period: str = "daily") -> JSONResponse:
        if period not in {"hourly", "daily", "weekly"}:
            raise HTTPException(status_code=422, detail="period must be hourly, daily, or weekly")
        return JSONResponse(_build_trends(root=root, period=period))

    @app.get("/api/runs/{run_id}/summary")
    def summary(run_id: str) -> JSONResponse:
        bundle_dir, meta = bundle(run_id)
        return JSONResponse(_build_summary(bundle_dir, meta))

    @app.get("/api/runs/{run_id}/labels")
    def labels(run_id: str, offset: int = 0, limit: int = 100) -> JSONResponse:
        bundle_dir, meta = bundle(run_id)
        labels_path = bundle_dir / "labels.jsonl"
        if not labels_path.is_file() and meta.get("out"):
            labels_path = Path(meta["out"])
        lines = labels_path.read_text(encoding="utf-8").splitlines() if labels_path.is_file() else []
        items = []
        end = min(offset + limit, len(lines))
        for line in lines[offset:end]:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return JSONResponse({"items": items, "next_offset": end if end < len(lines) else None})

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_DASHBOARD_HTML)

    @app.get("/health")
    def health() -> PlainTextResponse:
        return PlainTextResponse("ok")

    return app


_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Switchyard Dream</title><style>
:root{--bg:#070b14;--panel:#0e1628;--panel2:#121d34;--ink:#edf4ff;--muted:#8da0bd;--line:#223252;--cyan:#52d9ff;--mint:#5ee6a8;--violet:#9d8cff;--amber:#ffc857;--red:#ff6b7a}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% -10%,#173760 0,transparent 35%),var(--bg);color:var(--ink);font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif}header{position:sticky;top:0;z-index:4;display:flex;gap:20px;align-items:center;padding:18px 28px;background:#070b14e8;backdrop-filter:blur(18px);border-bottom:1px solid var(--line)}.brand{font-size:20px;font-weight:800;letter-spacing:.04em}.brand b{color:var(--cyan)}.run-select{margin-left:auto;display:flex;align-items:center;gap:10px}select{color:var(--ink);background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:9px 12px;min-width:310px}main{max-width:1500px;margin:auto;padding:24px}.eyebrow{text-transform:uppercase;letter-spacing:.15em;color:var(--cyan);font-size:11px;font-weight:800}.hero{display:flex;justify-content:space-between;align-items:end;margin:6px 0 20px}.hero h1{font-size:32px;margin:5px 0}.sub{color:var(--muted)}.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:16px}.kpi,.card{background:linear-gradient(145deg,#111c32dd,#0c1425ee);border:1px solid var(--line);border-radius:14px;box-shadow:0 16px 40px #0004}.kpi{padding:16px}.kpi .value{font-size:25px;font-weight:800;margin-top:5px}.kpi .label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.12em}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}.card{grid-column:span 6;padding:18px;min-height:350px}.card.wide{grid-column:span 12}.card h2{margin:0;font-size:16px}.card .desc{color:var(--muted);font-size:12px;margin:3px 0 14px}.chart{height:275px;position:relative}.chart.rows{height:auto;min-height:120px}.chart svg{width:100%;height:100%;overflow:visible}.chart.rows svg{height:auto}.axis{stroke:#385071;stroke-width:1}.gridline{stroke:#1d2a45;stroke-width:1}.tick{fill:var(--muted);font-size:10px}.lbl{fill:var(--ink);font-size:11px;font-weight:600}.tooltip{position:fixed;display:none;z-index:10;pointer-events:none;background:#050912f5;border:1px solid #40577b;border-radius:9px;padding:9px 11px;box-shadow:0 10px 30px #0009;white-space:pre-line}.legend{display:flex;gap:14px;color:var(--muted);font-size:11px}.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px}.empty{display:grid;place-items:center;height:100%;color:var(--muted)}.table{overflow:auto;max-height:260px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.1em}.pill{padding:3px 7px;border-radius:99px;background:#183251;color:var(--cyan);font-size:11px}.loading{opacity:.45;filter:saturate(.4)}@media(max-width:1000px){.kpis{grid-template-columns:repeat(3,1fr)}.card{grid-column:span 12}}@media(max-width:620px){header{padding:14px;align-items:flex-start;flex-direction:column}.run-select{margin:0;width:100%;flex-direction:column;align-items:stretch}select{min-width:0;width:100%}main{padding:14px}.kpis{grid-template-columns:repeat(2,1fr)}.hero h1{font-size:24px}.card{padding:14px}}
</style></head><body><header><div class="brand">SWITCHYARD <b>/ DREAM</b></div><div class="run-select"><span class="sub">Historical run</span><select id="runs"></select></div></header><main id="main" class="loading"><div class="hero"><div><div class="eyebrow">Offline intelligence</div><h1>Routing Observatory</h1><div class="sub" id="runMeta">Loading run data…</div></div><div class="pill" id="coverage">— records</div></div><section class="kpis" id="kpis"></section><section class="grid"><article class="card"><h2>Arm performance</h2><div class="desc">Success and reward by model × token bucket</div><div class="chart rows" id="arms"></div></article><article class="card"><h2>Cost / latency frontier</h2><div class="desc">Circle size = call volume · color = success rate</div><div class="chart" id="scatter"></div></article><article class="card"><h2>Token composition</h2><div class="desc">Top models by routed call volume</div><div class="chart rows" id="tokens"></div></article><article class="card"><h2>Selector attribution trends</h2><div class="desc" id="selectorDesc">Models selected over time by route and target · click a point for details</div><select id="trendPeriod"><option value="hourly">Hourly</option><option value="daily" selected>Daily</option><option value="weekly">Weekly</option></select><div class="chart" id="selector"></div><div class="table" id="selectorDetail"></div></article><article class="card"><h2>Judge calibration</h2><div class="desc" id="brier">Capability-judge prediction calibration (custom target selectors do not emit probabilities)</div><div class="chart" id="calibration"></div></article><article class="card wide"><h2>Tool-call churn opportunities</h2><div class="desc">Repeated intents that a cache-backed tool or skill could eliminate</div><div class="chart rows" id="mining"></div></article><article class="card wide"><h2>Model detail</h2><div class="desc">Coverage and operational totals from this routing snapshot</div><div class="table" id="detail"></div></article></section></main><div class="tooltip" id="tip"></div><script>
const $=s=>document.querySelector(s), NS='http://www.w3.org/2000/svg', colors={cyan:'#52d9ff',mint:'#5ee6a8',violet:'#9d8cff',amber:'#ffc857',red:'#ff6b7a'};let data,trends;
const fmt=(n,d=1)=>Number(n||0).toLocaleString(undefined,{maximumFractionDigits:d}),pct=n=>fmt(n*100,1)+'%',money=n=>'$'+fmt(n,3);
function svgBox(h=270){let s=document.createElementNS(NS,'svg');s.setAttribute('viewBox','0 0 600 '+h);s.setAttribute('preserveAspectRatio','xMinYMin meet');return s}function el(tag,a={}){let x=document.createElementNS(NS,tag);for(let[k,v]of Object.entries(a))x.setAttribute(k,v);return x}function text(svg,x,y,t,cls='tick',anchor='start'){let n=el('text',{x,y,class:cls,'text-anchor':anchor});n.textContent=t;svg.append(n)}function hover(node,msg){node.onmousemove=e=>{let t=$('#tip');t.style.display='block';t.style.left=e.clientX+14+'px';t.style.top=e.clientY+14+'px';t.textContent=msg};node.onmouseleave=()=>$('#tip').style.display='none'}function empty(id,msg){$(id).innerHTML='<div class="empty">'+msg+'</div>'}
function axes(s,{xLabel='',yLabel='',xTicks=[],yTicks=[]}={}){for(let y of [30,80,130,180,230])s.append(el('line',{x1:55,y1:y,x2:580,y2:y,class:'gridline'}));s.append(el('line',{x1:55,y1:230,x2:580,y2:230,class:'axis'}));s.append(el('line',{x1:55,y1:20,x2:55,y2:230,class:'axis'}));for(let [v,l]of xTicks)text(s,55+525*v,249,l,'tick','middle');for(let [v,l]of yTicks)text(s,48,230-200*v,l,'tick','end');text(s,320,267,xLabel,'tick','middle');let y=text(s,12,130,yLabel,'tick','middle')}
function bars(){let a=data.arms.slice(0,14);if(!a.length)return empty('#arms','No arm data');let rh=34,s=svgBox(12+a.length*rh),max=Math.max(...a.map(x=>x.calls));for(let [i,x]of a.entries()){let y=10+i*rh;text(s,6,y+11,x.model+' · '+x.token_bucket,'lbl','start');let w=Math.max(2,585*x.calls/max);s.append(el('rect',{x:6,y:y+16,width:w,height:11,rx:3,fill:colors.cyan,opacity:.18+.72*x.success_rate}));text(s,12+w,y+25,pct(x.success_rate)+' / '+pct(x.mean_reward),'tick');hover(s.lastChild,`${x.model} [${x.token_bucket}]\n${fmt(x.calls,0)} calls\nSuccess ${pct(x.success_rate)}\nReward ${pct(x.mean_reward)}`)}$('#arms').replaceChildren(s)}
function scatter(){let m=data.models.filter(x=>x.mean_cost_usd>=0&&x.p95_latency_ms>=0);if(!m.length)return empty('#scatter','No cost or latency data');let s=svgBox(),mx=Math.max(...m.map(x=>x.mean_cost_usd),.001),my=Math.max(...m.map(x=>x.p95_latency_ms),1),mc=Math.max(...m.map(x=>x.calls));axes(s,{xLabel:'mean cost / call',yLabel:'p95 latency',xTicks:[[0,'$0'],[.5,money(mx/2)],[1,money(mx)]],yTicks:[[0,'0'],[.5,fmt(my/2000)+'s'],[1,fmt(my/1000)+'s']]});for(let x of m){let cx=55+525*x.mean_cost_usd/mx,cy=230-200*x.p95_latency_ms/my,r=5+13*Math.sqrt(x.calls/mc),h=120*x.success_rate;s.append(el('circle',{cx,cy,r,fill:`hsl(${h} 75% 58%)`,opacity:.78,stroke:'#d9f6ff','stroke-width':1}));hover(s.lastChild,`${x.model}\n${fmt(x.calls,0)} calls\n${money(x.mean_cost_usd)}/call\np95 ${fmt(x.p95_latency_ms/1000,2)}s\nSuccess ${pct(x.success_rate)}`)}$('#scatter').replaceChildren(s)}
function tokens(){let m=data.models.slice(0,10);if(!m.length)return empty('#tokens','No token data');let rh=34,s=svgBox(28+m.length*rh),keys=['prompt','cached','completion','reasoning'],cols=[colors.cyan,colors.mint,colors.violet,colors.amber],max=Math.max(...m.map(x=>keys.reduce((z,k)=>z+x.tokens[k],0)),1);m.forEach((x,i)=>{let y=10+i*rh;text(s,6,y+11,x.model,'lbl','start');let cur=6;keys.forEach((k,j)=>{let w=585*x.tokens[k]/max;if(w>0)s.append(el('rect',{x:cur,y:y+16,width:w,height:12,fill:cols[j]}));cur+=w});hover(s.lastChild,`${x.model}\nPrompt ${fmt(x.tokens.prompt,0)}\nCached ${fmt(x.tokens.cached,0)}\nCompletion ${fmt(x.tokens.completion,0)}\nReasoning ${fmt(x.tokens.reasoning,0)}`)});let lg=el('g'),ly=18+m.length*rh;keys.forEach((k,i)=>{lg.append(el('rect',{x:6+i*105,y:ly,width:9,height:9,fill:cols[i]}));let t=el('text',{x:19+i*105,y:ly+9,class:'tick'});t.textContent=k;lg.append(t)});s.append(lg);$('#tokens').replaceChildren(s)}
function selectorDetail(p){let rows=p.routes.flatMap(r=>r.targets.map(t=>`<tr><td>${r.route_id}</td><td>${t.target}</td><td>${fmt(t.calls,0)}</td><td>${pct(t.share)}</td></tr>`)).join('');$('#selectorDetail').innerHTML=`<table><thead><tr><th>Route</th><th>Target</th><th>Calls</th><th>Route share</th></tr></thead><tbody>${rows}</tbody></table>`;$('#selectorDesc').textContent=`${new Date(p.start).toLocaleString()} · ${fmt(p.calls,0)} attributed calls`}
function selector(){let points=trends?.points||[];if(!points.length)return empty('#selector','No route attribution yet. New server records include route_id; historical snapshots cannot be attributed.');let targets=[...new Set(points.flatMap(p=>p.routes.flatMap(r=>r.targets.map(t=>r.route_id+' → '+t.target))))],s=svgBox(),max=Math.max(...points.flatMap(p=>p.routes.flatMap(r=>r.targets.map(t=>t.calls))),1);axes(s,{xLabel:trends.period,yLabel:'selected calls',xTicks:points.map((p,i)=>[points.length===1?.5:i/(points.length-1),new Date(p.start).toLocaleDateString()]),yTicks:[[0,'0'],[.5,fmt(max/2,0)],[1,fmt(max,0)]]});targets.forEach((key,j)=>{let values=points.map((p,i)=>{let [route,target]=key.split(' → '),r=p.routes.find(x=>x.route_id===route),t=r?.targets.find(x=>x.target===target);return[points.length===1?317:55+525*i/(points.length-1),230-200*(t?.calls||0)/max,p,t?.calls||0]}),color=`hsl(${j*137.5%360} 75% 62%)`;s.append(el('polyline',{points:values.map(v=>v[0]+','+v[1]).join(' '),fill:'none',stroke:color,'stroke-width':3}));values.forEach(([x,y,p,c])=>{let n=el('circle',{cx:x,cy:y,r:5,fill:color,tabindex:0});n.onclick=()=>selectorDetail(p);s.append(n);hover(n,`${key}\n${fmt(c,0)} calls\n${new Date(p.start).toLocaleString()}`)})});$('#selector').replaceChildren(s);selectorDetail(points[points.length-1])}
function calibration(){let b=data.calibration.bins;if(!b.length)return empty('#calibration','This run uses custom target selectors, which choose a target but do not emit p_solve. See Custom selector decisions instead.');let s=svgBox();axes(s,{xLabel:'predicted solve probability',yLabel:'observed success',xTicks:[[0,'0'],[.5,'.5'],[1,'1']],yTicks:[[0,'0'],[.5,'.5'],[1,'1']]});s.append(el('line',{x1:55,y1:230,x2:580,y2:30,stroke:'#667a99','stroke-dasharray':'5 5'}));let pts=b.map(x=>[55+525*x.mean_prediction,230-200*x.observed_success,x]);s.append(el('polyline',{points:pts.map(p=>p[0]+','+p[1]).join(' '),fill:'none',stroke:colors.violet,'stroke-width':3}));for(let[x,y,v]of pts){s.append(el('circle',{cx:x,cy:y,r:4+Math.sqrt(v.count),fill:colors.violet,stroke:'#fff'}));hover(s.lastChild,`${fmt(v.lower,1)}–${fmt(v.upper,1)}\n${fmt(v.count,0)} predictions\nPredicted ${pct(v.mean_prediction)}\nObserved ${pct(v.observed_success)}`)}$('#calibration').replaceChildren(s);$('#brier').textContent='Predicted solve rate vs observed success · Brier '+fmt(data.calibration.brier,4)}
function mining(){let m=data.mining;if(!m.available)return empty('#mining','No mining report for this run');if(!m.top.length)return empty('#mining',`${fmt(m.intents,0)} intents across ${fmt(m.sessions,0)} sessions · no exact repeats yet`);let rh=34,s=svgBox(12+m.top.length*rh),max=Math.max(...m.top.map(x=>x.excess),1);m.top.forEach((x,i)=>{let y=10+i*rh,label=x.intent.command||x.intent.tool||x.intent.kind;text(s,6,y+11,label,'lbl','start');let w=Math.max(2,585*x.excess/max);s.append(el('rect',{x:6,y:y+16,width:w,height:12,rx:3,fill:colors.amber,opacity:.85}));text(s,12+w,y+26,x.excess+' avoidable','tick');hover(s.lastChild,`${label}\n${x.occurrences} occurrences in ${x.sessions} sessions\n${x.excess} excess executions`)});$('#mining').replaceChildren(s)}
function table(){let rows=data.models.map(x=>`<tr><td>${x.model}</td><td>${fmt(x.calls,0)}</td><td>${pct(x.success_rate)}</td><td>${money(x.cost_usd)}</td><td>${fmt(x.p95_latency_ms/1000,2)}s</td><td>${fmt(x.tokens.prompt+x.tokens.cached+x.tokens.completion,0)}</td></tr>`).join('');$('#detail').innerHTML=`<table><thead><tr><th>Model</th><th>Calls</th><th>Success</th><th>Cost</th><th>p95</th><th>Tokens</th></tr></thead><tbody>${rows}</tbody></table>`}
function render(){let t=data.totals,c=data.coverage,m=data.mining;$('#runMeta').textContent=new Date(data.run.created_at).toLocaleString()+' · '+(data.run.id||'run');$('#coverage').textContent=fmt(c.records,0)+' records';let items=[['Answer calls',fmt(c.answer_calls,0)],['Success',pct(t.success_rate)],['Mean reward',pct(t.mean_reward)],['Total cost',money(t.cost_usd)],['p95 latency',fmt(t.p95_latency_ms/1000,2)+'s'],['Avoidable calls',fmt(m.exact_duplicates,0)]];$('#kpis').innerHTML=items.map(x=>`<div class="kpi"><div class="label">${x[0]}</div><div class="value">${x[1]}</div></div>`).join('');bars();scatter();tokens();selector();calibration();mining();table();$('#main').classList.remove('loading')}
async function loadTrends(){trends=await(await fetch('/api/trends?period='+$('#trendPeriod').value)).json();selector()}
async function load(){let id=$('#runs').value;if(!id)return;$('#main').classList.add('loading');let r=await fetch('/api/runs/'+encodeURIComponent(id)+'/summary');data=await r.json();await loadTrends();render()}
async function init(){let runs=await (await fetch('/api/runs')).json(),s=$('#runs');s.innerHTML=runs.map(r=>`<option value="${r.id}">${new Date(r.created_at).toLocaleString()} · ${r.run_id||'run'}</option>`).join('');s.onchange=load;$('#trendPeriod').onchange=loadTrends;await load()}init().catch(e=>{$('#runMeta').textContent='Failed to load: '+e;$('#main').classList.remove('loading')});
</script></body></html>"""
