# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the offline dream-step label and calibration tooling."""

import json

from switchyard.dream import (
    brier_score,
    cheap_wrong_rate,
    emit_labels,
    judge_calibration,
    read_records,
    summarize_arms,
    teacher_calibration,
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
