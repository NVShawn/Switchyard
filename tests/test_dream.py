# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the offline dream-step label and calibration tooling."""

import json
from pathlib import Path

import pytest

from switchyard.cli.switchyard_cli import _build_parser, _validate_args
from switchyard.dream import (
    brier_score,
    build_classifier_dataset,
    cheap_wrong_rate,
    emit_labels,
    judge_calibration,
    learned_target_weights,
    parse_label_map,
    read_records,
    summarize_arms,
    teacher_calibration,
    write_classifier_dataset,
    write_classifier_route,
    write_learned_weights,
)
from switchyard.dream_mining import (
    build_mining_report,
    infer_transcript_path,
    read_transcript_intents,
    write_mining_artifacts,
)


def _write_log(tmp_path, records):
    path = tmp_path / "routing.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_read_records_skips_blank_and_invalid_lines(tmp_path):
    path = tmp_path / "routing.jsonl"
    path.write_text('{"model":"a","reward":0.5}\n\nnot json\n{"model":"b"}\n')
    records = read_records(path)
    assert len(records) == 2
    assert records[0]["model"] == "a"


def test_summarize_arms_aggregates_answer_calls_only():
    records = [
        {"model": "nano", "token_bucket": "small", "reward": 0.8, "success": True},
        {"model": "nano", "token_bucket": "small", "reward": 0.0, "success": False},
        {"model": "nano", "token_bucket": "large", "reward": 0.6, "success": True},
        # A classifier call is not an arm.
        {"model": "judge", "tier": "classifier", "token_bucket": "small", "reward": 0.9},
        # A legacy record with no reward is not an arm.
        {"model": "nano", "token_bucket": "small"},
    ]
    arms = summarize_arms(records)
    small = arms[("nano", "small")]
    assert small.count == 2
    assert small.sum_reward == 0.8
    # alpha = sum + 1, beta = count - sum + 1.
    assert small.alpha == 1.8
    assert small.beta == 2.2
    assert 0.0 < small.mean < 1.0
    assert arms[("nano", "large")].count == 1
    assert ("judge", "small") not in arms


def test_cheap_wrong_rate_counts_failed_non_capable_calls():
    records = [
        {"model": "nano", "tier": "", "success": True},
        {"model": "nano", "tier": "", "success": False},
        {"model": "opus", "tier": "strong", "success": True},
        {"model": "judge", "tier": "classifier", "success": True},
    ]
    # One of three answer calls was a cheap-tier failure.
    assert cheap_wrong_rate(records, capable_tiers={"strong"}) == 1 / 3


def test_brier_score_rewards_calibration():
    # Confident and correct beats hedging.
    assert brier_score([(0.9, True), (0.1, False)]) < brier_score([(0.5, True), (0.5, False)])
    assert brier_score([]) == 0.0


def _fake_judge(task):
    return {
        "crux": "bounded task",
        "primary_rule": "SUP-1",
        "capability_boundary": "supported",
        "p_solve": 0.9,
        "minimum_capability": 0.2,
    }


def test_emit_labels_rejudges_each_task_once_and_pairs_the_outcome():
    records = [
        {"model": "nano", "task": "fix the test", "success": True, "reward": 0.9, "token_bucket": "small"},
        # Same task again: the verdict is reused, not re-judged.
        {"model": "nano", "task": "fix the test", "success": True, "reward": 0.8, "token_bucket": "small"},
        # No task header: nothing to re-judge.
        {"model": "opus", "success": True, "reward": 0.5},
        # A classifier call is not re-judged.
        {"model": "judge", "tier": "classifier", "task": "ignored", "success": True},
    ]
    calls = []

    def judge(task):
        calls.append(task)
        return _fake_judge(task)

    labels = emit_labels(records, judge)
    assert calls == ["fix the test"], "each unique task is judged once"
    assert len(labels) == 2
    assert labels[0]["label"]["minimum_capability"] == 0.2
    assert labels[0]["label"]["capability_boundary"] == "supported"
    assert labels[0]["outcome"]["reward"] == 0.9


def test_teacher_calibration_scores_against_observed_outcomes():
    labels = [
        {"label": {"p_solve": 0.9}, "outcome": {"success": True}},
        {"label": {"p_solve": 0.2}, "outcome": {"success": False}},
    ]
    # (0.9-1)^2 = 0.01 and (0.2-0)^2 = 0.04, averaged.
    assert teacher_calibration(labels) == 0.025
    # Records lacking a verdict or outcome are skipped.
    assert teacher_calibration([{"label": {}, "outcome": {}}]) == 0.0


def test_judge_calibration_uses_the_logged_verdict():
    records = [
        # Answer records carrying the serving judge's verdict.
        {"model": "nano", "success": True, "judge_p_solve": 0.9},
        {"model": "nano", "success": False, "judge_p_solve": 0.2},
        # A classifier record is not an answer and is skipped.
        {"model": "judge", "tier": "classifier", "success": True, "judge_p_solve": 0.9},
        # An answer record with no verdict is skipped.
        {"model": "opus", "success": True},
    ]
    assert judge_calibration(records) == 0.025


def test_emit_labels_include_the_logged_judge_verdict():
    records = [
        {
            "model": "nano",
            "task": "fix the test",
            "success": True,
            "reward": 0.9,
            "judge_p_solve": 0.8,
            "judge_capability_boundary": "supported",
        }
    ]
    labels = emit_labels(records, _fake_judge)
    assert labels[0]["judge_verdict"]["p_solve"] == 0.8
    assert labels[0]["judge_verdict"]["capability_boundary"] == "supported"
    assert labels[0]["label"]["minimum_capability"] == 0.2


def test_dream_cli_exposes_transcript_mining_flags(tmp_path: Path) -> None:
    args = _build_parser().parse_args(
        [
            "dream",
            "--log",
            str(tmp_path / "routing.jsonl"),
            "--transcript",
            str(tmp_path / "transcript.jsonl"),
            "--mine",
            "--emit-skills",
            "--emit-tools",
        ]
    )

    assert args.transcript == tmp_path / "transcript.jsonl"
    assert args.mine is True
    assert args.emit_skills is True
    assert args.emit_tools is True


def test_learned_target_weights_favor_the_higher_reward_target():
    # weak earns high rewards, premium fails: the learned weights must rank weak first.
    records = [
        {"model": "model/weak", "reward": 0.9, "token_bucket": "small"},
        {"model": "model/weak", "reward": 0.8, "token_bucket": "medium"},
        {"model": "model/premium", "reward": 0.0, "token_bucket": "small"},
        {"model": "model/premium", "reward": 0.0, "token_bucket": "small"},
        # A classifier call is not an arm and must not contribute a weight.
        {"model": "model/classifier", "tier": "classifier", "reward": 0.0, "token_bucket": "small"},
    ]
    model_to_label = {"model/weak": "weak", "model/premium": "premium"}

    weights = learned_target_weights(records, model_to_label)

    by_label = {weight.label: weight for weight in weights}
    assert set(by_label) == {"weak", "premium"}
    assert by_label["weak"].mean > by_label["premium"].mean
    best = max(weights, key=lambda weight: weight.mean)
    assert best.label == "weak"


def test_learned_target_weights_default_label_to_model_id():
    records = [{"model": "model/weak", "reward": 0.5, "token_bucket": "small"}]

    weights = learned_target_weights(records, {})

    assert [weight.label for weight in weights] == ["model/weak"]


def test_parse_label_map_rejects_malformed_pairs():
    assert parse_label_map(["weak=model/weak"]) == {"model/weak": "weak"}
    for bad in ["weak", "=model", "weak="]:
        try:
            parse_label_map([bad])
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_write_learned_weights_emits_target_array(tmp_path: Path):
    from switchyard.dream import TargetWeight

    out = tmp_path / "weights.toml"
    write_learned_weights(
        out,
        [
            TargetWeight("weak", alpha=19.0, beta=1.0),
            TargetWeight("premium", alpha=2.0, beta=8.0),
        ],
    )

    text = out.read_text()
    assert text.count("[[target]]") == 2
    assert 'label = "weak"' in text
    assert "alpha = 19.0" in text
    assert "beta = 8.0" in text


def test_dream_cli_exposes_learned_weight_flags(tmp_path: Path) -> None:
    args = _build_parser().parse_args(
        [
            "dream",
            "--log",
            str(tmp_path / "routing.jsonl"),
            "--emit-weights",
            str(tmp_path / "weights.toml"),
            "--label-map",
            "weak=model/weak",
            "--label-map",
            "premium=model/premium",
        ]
    )

    assert args.emit_weights == tmp_path / "weights.toml"
    assert args.label == ["weak=model/weak", "premium=model/premium"]


def test_dream_cli_validates_label_mapping(capsys: pytest.CaptureFixture[str]) -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["dream", "--log", "routing.jsonl", "--label-map", "weak"])
    assert "expected TARGET=MODEL_ID" in capsys.readouterr().err

    args = parser.parse_args(
        ["dream", "--log", "routing.jsonl", "--label-map", "weak=model/weak"]
    )
    with pytest.raises(SystemExit):
        _validate_args(parser, args)
    assert "--label-map requires --emit-weights" in capsys.readouterr().err


def test_dream_cli_help_describes_learned_weight_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _build_parser().parse_args(["dream", "--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--emit-weights PATH" in help_text
    assert "--label-map TARGET=MODEL_ID" in help_text
    assert "llm_classifier target_selector route" in help_text


def test_infer_transcript_path_matches_server_naming(tmp_path: Path) -> None:
    assert infer_transcript_path(tmp_path / "routing.jsonl") == (
        tmp_path / "routing.transcript.jsonl"
    )


def test_read_transcript_intents_normalizes_tools_and_bash_commands(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    records = [
        {
            "event": "normalized_response",
            "request_id": "request-1",
            "session_id": "session-1",
            "normalized": {
                "outputs": [
                    {
                        "content": [
                            {
                                "type": "tool_call",
                                "id": "call-1",
                                "name": "shell",
                                "arguments": {"command": "rg   'needle' src"},
                            },
                            {
                                "type": "tool_call",
                                "id": "call-2",
                                "name": "read_file",
                                "arguments": {"path": "a.py", "line": 3},
                            },
                        ]
                    }
                ]
            },
        },
        {"event": "provider_response", "session_id": "session-1", "raw": {}},
    ]
    transcript.write_text("\n".join(json.dumps(record) for record in records))

    assert read_transcript_intents(transcript) == {
        "session-1": [
            {"kind": "bash_search", "tool": "shell", "command": "rg needle src"},
            {
                "kind": "tool_call",
                "tool": "read_file",
                "arguments": {"line": 3, "path": "a.py"},
            },
        ]
    }


def test_mining_report_detects_session_local_duplicates_deterministically(tmp_path: Path) -> None:
    intent = {"kind": "bash_read", "tool": "bash", "command": "cat file.py"}
    sessions = {"session-b": [intent], "session-a": [intent, intent]}

    report = build_mining_report(sessions, tmp_path / "transcript.jsonl")

    assert report["exact_duplicate_count"] == 1
    assert [session["session_id"] for session in report["sessions"]] == [
        "session-a",
        "session-b",
    ]
    assert report["sessions"][0]["duplicate_intents"] == [{"intent": intent, "count": 2}]
    assert report["sessions"][1]["duplicate_intents"] == []


def test_write_mining_artifacts_emits_versioned_drafts(tmp_path: Path) -> None:
    intent = {"kind": "tool_call", "tool": "read_file", "arguments": {"path": "a.py"}}
    report = build_mining_report({"session": [intent, intent]}, tmp_path / "transcript.jsonl")

    write_mining_artifacts(tmp_path, report, emit_skills=True, emit_tools=True)

    persisted = json.loads((tmp_path / "mining_report.json").read_text())
    assert persisted["version"] == 1
    assert "Exact duplicates: 1" in (tmp_path / "mining_report.md").read_text()
    skill = next((tmp_path / "generated_skills").glob("*.md"))
    tool = next((tmp_path / "generated_tools").glob("*.json"))
    assert "version: 1" in skill.read_text()
    assert json.loads(tool.read_text())["version"] == 1


def test_build_classifier_dataset_labels_best_target_per_task() -> None:
    records = [
        {"task": "solve X", "model": "model/weak", "reward": 0.0},
        {"task": "solve X", "model": "model/weak", "reward": 0.0},
        {"task": "solve X", "model": "model/premium", "reward": 1.0},
        {"task": "greet", "model": "model/weak", "reward": 1.0},
        {"task": "greet", "model": "model/premium", "reward": 1.0},
        {"task": "classifier call", "model": "model/weak", "reward": 1.0, "tier": "classifier"},
        {"model": "model/weak", "reward": 1.0},
    ]
    samples = build_classifier_dataset(
        records, {"model/weak": "weak", "model/premium": "premium"}
    )

    by_task = {sample.task: sample for sample in samples}
    assert set(by_task) == {"solve X", "greet"}
    assert by_task["solve X"].best_target == "premium"
    assert by_task["greet"].best_target == "premium"
    assert by_task["solve X"].stats["weak"] == (2, 0.0)


def test_write_classifier_dataset_emits_self_describing_jsonl(tmp_path: Path) -> None:
    samples = build_classifier_dataset(
        [
            {"task": "t", "model": "model/weak", "reward": 0.0},
            {"task": "t", "model": "model/premium", "reward": 1.0},
        ],
        {"model/weak": "weak", "model/premium": "premium"},
    )
    out = tmp_path / "classifier_dataset.jsonl"
    write_classifier_dataset(out, samples)

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows == [
        {
            "text": "t",
            "target": "premium",
            "stats": {
                "premium": {"count": 1, "sum_reward": 1.0, "mean": 2.0 / 3.0},
                "weak": {"count": 1, "sum_reward": 0.0, "mean": 1.0 / 3.0},
            },
        }
    ]


def test_write_classifier_route_tunes_prompt_and_tightens_schema(tmp_path: Path) -> None:
    weights = learned_target_weights(
        [
            {"model": "model/weak", "reward": 0.0},
            {"model": "model/weak", "reward": 0.0},
            {"model": "model/premium", "reward": 1.0},
            {"model": "model/premium", "reward": 1.0},
        ],
        {"model/weak": "weak", "model/premium": "premium"},
    )
    out = tmp_path / "classifier_route.toml"
    write_classifier_route(out, weights)

    import tomllib

    parsed = tomllib.loads(out.read_text())
    prompt = parsed["routes"]["auto"]["prompt"]
    assert "mean_reward" in prompt
    assert prompt.index("premium: mean_reward") < prompt.index("weak: mean_reward")

    schema = json.loads(parsed["routes"]["auto"]["response_schema"])
    enum = schema["properties"]["decision"]["properties"]["target"]["enum"]
    assert enum == ["premium", "weak"]
    assert schema["properties"]["decision"]["additionalProperties"] is False


def test_dream_cli_exposes_emit_classifier_flag(tmp_path: Path) -> None:
    args = _build_parser().parse_args(
        [
            "dream",
            "--log",
            str(tmp_path / "routing.jsonl"),
            "--emit-classifier",
            str(tmp_path / "classifier_route.toml"),
            "--label-map",
            "weak=model/weak",
        ]
    )
    assert args.emit_classifier == tmp_path / "classifier_route.toml"
    parser = _build_parser()
    _validate_args(parser, args)
