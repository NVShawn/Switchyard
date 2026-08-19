# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline dream-step: re-judge logged routing decisions and score calibration.

The serving router logs every routed call (model, cost, latency, success, reward,
token bucket) to a JSONL routing log. The dream step reads that log offline and:

1. Re-derives each (model, token bucket) arm's Beta posterior from its rewards —
   the offline mirror of the online bandit's priors.
2. Reports calibration: per-arm success rates and the cheap-but-wrong rate the
   router should be driving down.
3. Optionally re-judges each logged task with a strong model to emit fine-tune
   labels in the capability judge's contract, and scores that teacher's
   calibration (Brier) against the observed outcomes before any label is trusted.

This never runs in the request path; it is a batch tool driven on a cadence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from switchyard.dream_mining import (
    build_mining_report,
    infer_transcript_path,
    read_transcript_intents,
    write_mining_artifacts,
)

# Tier label the server writes for judge/classifier calls, which are not bandit arms.
_CLASSIFIER_TIER = "classifier"

# Capability rules the teacher's verdicts must name, mirrored from the packaged
# capability-classifier contract so emitted labels stay in-contract.
_RULES = (
    "SUP-1",
    "SUP-2",
    "SUP-3",
    "SUP-4",
    "SUP-5",
    "UNC-1",
    "UNC-2",
    "LIM-1",
    "LIM-2",
    "none",
)
_BOUNDARIES = ("supported", "uncertain", "unsupported", "unmatched")


@dataclass(frozen=True)
class ArmSummary:
    """Aggregated reward for one (model, token bucket) bandit arm."""

    model: str
    token_bucket: str
    count: int
    sum_reward: float

    @property
    def alpha(self) -> float:
        """Soft success count plus one, the Beta prior's alpha."""
        return self.sum_reward + 1.0

    @property
    def beta(self) -> float:
        """Soft failure count plus one, floored at one so the prior stays defined."""
        return max(self.count - self.sum_reward + 1.0, 1.0)

    @property
    def mean(self) -> float:
        """Posterior mean reward — the bandit's current estimate for the arm."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def success_rate(self) -> float:
        """Mean observed reward, 0..1 (rewards are already 0..1 normalized)."""
        return self.sum_reward / self.count if self.count else 0.0


def read_records(path: Path) -> list[dict[str, Any]]:
    """Parse the routing log, skipping blank or unparseable lines."""
    records = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def summarize_arms(records: list[dict[str, Any]]) -> dict[tuple[str, str], ArmSummary]:
    """Aggregate answer-call rewards per (model, token bucket) arm.

    Only answer calls carry a reward; classifier calls and legacy records without one
    are not bandit arms and are skipped.
    """
    arms: dict[tuple[str, str], ArmSummary] = {}
    for record in records:
        reward = record.get("reward")
        bucket = record.get("token_bucket")
        model = record.get("model")
        if reward is None or bucket is None or not model:
            continue
        if record.get("tier") == _CLASSIFIER_TIER:
            continue
        key = (model, bucket)
        existing = arms.get(key)
        if existing is None:
            arms[key] = ArmSummary(model, bucket, 1, float(reward))
        else:
            arms[key] = ArmSummary(model, bucket, existing.count + 1, existing.sum_reward + float(reward))
    return arms


def cheap_wrong_rate(records: list[dict[str, Any]], capable_tiers: set[str]) -> float:
    """Share of answer calls that served a non-capable tier and then failed.

    These are the judge's costly mistakes: a task routed cheap that should have been
    escalated. The dream step exists to drive this down.
    """
    served = [r for r in records if r.get("tier") != _CLASSIFIER_TIER and r.get("success") is not None]
    if not served:
        return 0.0
    wrong = [
        r
        for r in served
        if r.get("success") is False and r.get("tier", "") not in capable_tiers
    ]
    return len(wrong) / len(served)


def brier_score(predictions: list[tuple[float, bool]]) -> float:
    """Mean squared error of predicted solve probability against 0/1 outcomes.

    Lower is better; a perfectly calibrated teacher on certain tasks approaches 0.
    """
    if not predictions:
        return 0.0
    return sum((p - (1.0 if outcome else 0.0)) ** 2 for p, outcome in predictions) / len(
        predictions
    )


def judge_calibration(records: list[dict[str, Any]]) -> float:
    """Brier score of the *logged* capability judge's `p_solve` against observed success.

    This is the serving judge's own calibration — no teacher required. Only answer
    records that carried a verdict count. The outcome here is transport-level success,
    which misses wrong-but-200 answers, so read this as a floor on the judge's true
    error, not a measure of answer correctness.
    """
    predictions = []
    for record in records:
        if record.get("tier") == _CLASSIFIER_TIER:
            continue
        p_solve = record.get("judge_p_solve")
        success = record.get("success")
        if isinstance(p_solve, (int, float)) and isinstance(success, bool):
            predictions.append((float(p_solve), success))
    return brier_score(predictions)


def _judge_prompt(task: str) -> str:
    """The compact re-judging prompt, in the capability judge's contract."""
    rules = ", ".join(_RULES)
    boundaries = ", ".join(_BOUNDARIES)
    return (
        "You are re-judging a task a cheaper model attempted. Read the task and estimate "
        "the lowest capability level that would solve it correctly.\n\n"
        f"Task:\n{task}\n\n"
        "Return exactly one JSON object with these fields and no other text:\n"
        '- "crux": the hardest material requirement for whole-task success (string)\n'
        f'- "primary_rule": one of {rules}\n'
        f'- "capability_boundary": one of {boundaries}\n'
        '- "p_solve": probability the weakest available model solves the task (0.0-1.0)\n'
        '- "minimum_capability": the lowest capability level that still solves it (0.0-1.0)\n'
    )


def make_openai_judge(
    model: str, base_url: str, api_key: str | None
) -> Callable[[str], dict[str, Any]]:
    """Build a judge that re-emits the capability verdict via an OpenAI-compatible API.

    Uses only the standard library so the core package keeps its zero-dependency
    surface. The returned callable maps a task string to a parsed verdict dict.
    """

    def judge(task: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": _judge_prompt(task)}],
                    "response_format": {"type": "json_object"},
                }
            ).encode(),
            headers={
                "content-type": "application/json",
                **({"authorization": f"Bearer {api_key}"} if api_key else {}),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read())
        content = payload["choices"][0]["message"]["content"]
        verdict: dict[str, Any] = json.loads(content)
        return verdict

    return judge


def emit_labels(
    records: list[dict[str, Any]], judge: Callable[[str], dict[str, Any]]
) -> list[dict[str, Any]]:
    """Re-judge each logged task and pair the verdict with the observed outcome.

    The outcome (success, reward, served model) lets the trainer weight cheap-but-wrong
    samples highest. Records without a task header are skipped: there is nothing to
    re-judge. Each unique task is judged once, then reused across its records.
    """
    verdicts: dict[str, dict[str, Any]] = {}
    labels = []
    for record in records:
        task = record.get("task")
        if not task or record.get("tier") == _CLASSIFIER_TIER:
            continue
        if task not in verdicts:
            verdicts[task] = judge(task)
        labels.append(
            {
                "task": task,
                "label": verdicts[task],
                # The serving judge's own verdict, when the router logged one — lets the
                # trainer measure judge-teacher agreement, not just teacher-vs-outcome.
                "judge_verdict": {
                    "p_solve": record.get("judge_p_solve"),
                    "capability_boundary": record.get("judge_capability_boundary"),
                    "minimum_capability": record.get("judge_minimum_capability"),
                },
                "outcome": {
                    "model": record.get("model"),
                    "success": record.get("success"),
                    "reward": record.get("reward"),
                    "token_bucket": record.get("token_bucket"),
                },
            }
        )
    return labels


def teacher_calibration(labels: list[dict[str, Any]]) -> float:
    """Brier score of the teacher's p_solve against each decision's observed success.

    Gate label trust on this: a teacher whose probabilities do not track outcomes should
    not be distilling into the judge.
    """
    predictions = []
    for label in labels:
        p_solve = label.get("label", {}).get("p_solve")
        success = label.get("outcome", {}).get("success")
        if isinstance(p_solve, (int, float)) and isinstance(success, bool):
            predictions.append((float(p_solve), success))
    return brier_score(predictions)


def _build_run_bundle(*, log_path: Path, out_path: Path) -> Path:
    out_path = out_path.resolve()
    parent = out_path.parent
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    bundle_dir = parent / f"{out_path.name}.{run_id}.dream"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "log": str(log_path.resolve()),
        "out": str(out_path),
    }
    (bundle_dir / "dream_run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    try:
        shutil.copyfile(log_path, bundle_dir / "routing.jsonl")
    except Exception:
        pass

    return bundle_dir


def cmd_dream(args: argparse.Namespace) -> Path:
    """Run the dream step over a routing log.

    Returns the bundle directory used to store artifacts for later inspection.
    """
    records = read_records(args.log)
    arms = summarize_arms(records)

    bundle_dir = _build_run_bundle(log_path=args.log, out_path=args.out)

    print(f"read {len(records)} records, {len(arms)} bandit arms")
    for arm in sorted(arms.values(), key=lambda a: (a.token_bucket, a.mean)):
        print(
            f"  {arm.model} [{arm.token_bucket}]: calls={arm.count} "
            f"mean_reward={arm.mean:.3f} success_rate={arm.success_rate:.3f}"
        )
    rate = cheap_wrong_rate(records, capable_tiers=set())
    print(f"cheap-but-wrong rate: {rate:.3f}")
    judged = judge_calibration(records)
    if any(r.get("judge_p_solve") is not None for r in records):
        print(f"serving-judge calibration (Brier, lower is better): {judged:.4f}")

    mine = args.mine or args.emit_skills or args.emit_tools
    if mine:
        transcript_path = args.transcript or infer_transcript_path(args.log)
        sessions = read_transcript_intents(transcript_path)
        report = build_mining_report(sessions, transcript_path)
        write_mining_artifacts(
            bundle_dir,
            report,
            emit_skills=args.emit_skills,
            emit_tools=args.emit_tools,
        )
        print(
            f"mined {report['intent_count']} intents across {report['session_count']} sessions; "
            f"found {report['exact_duplicate_count']} exact duplicates"
        )

    if args.strong_model:
        import os

        judge = make_openai_judge(
            args.strong_model,
            args.base_url,
            args.api_key or os.environ.get("OPENAI_API_KEY"),
        )
        labels = emit_labels(records, judge)
        out = args.out
        serialized = "".join(json.dumps(label) + "\n" for label in labels)
        out.write_text(serialized)
        (bundle_dir / "labels.jsonl").write_text(serialized, encoding="utf-8")
        print(f"wrote {len(labels)} fine-tune labels to {out}")
        print(f"teacher calibration (Brier, lower is better): {teacher_calibration(labels):.4f}")

    return bundle_dir


def cmd_dream_ui(args: argparse.Namespace) -> None:
    """Run the dream step, then serve the local web UI only when ``--ui`` is set.

    Without ``--ui`` this runs the batch dream step and returns, so scheduled
    (cron) invocations terminate instead of blocking on a long-lived server.
    """
    bundle_dir = cmd_dream(args)

    if not args.ui:
        return

    from switchyard.cli.dream_ui import serve_dream_ui

    raise SystemExit(
        serve_dream_ui(
            host=args.ui_host,
            port=args.ui_port,
            initial_bundle=bundle_dir,
        )
    )
