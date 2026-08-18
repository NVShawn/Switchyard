# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transcript mining for offline dream runs."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPORT_VERSION = 1
_SEARCH_COMMANDS = {"find", "fd", "grep", "rg", "locate"}
_READ_COMMANDS = {"cat", "head", "less", "more", "sed", "tail"}
_SHELL_TOOLS = {"bash", "run_shell", "shell"}


def infer_transcript_path(log_path: Path) -> Path:
    """Return the server's default transcript path for a routing log."""
    return log_path.with_name("routing.transcript.jsonl")


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _command_kind(command: str) -> str | None:
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    while words and ("=" in words[0] and not words[0].startswith(("=", "-"))):
        words.pop(0)
    if not words:
        return None
    executable = Path(words[0]).name
    if executable in _SEARCH_COMMANDS:
        return "bash_search"
    if executable in _READ_COMMANDS:
        return "bash_read"
    return None


def _normalize_command(command: str) -> str:
    try:
        return shlex.join(shlex.split(command))
    except ValueError:
        return " ".join(command.split())


def _intent(block: dict[str, Any]) -> dict[str, Any] | None:
    name = block.get("name")
    arguments = block.get("arguments", {})
    if not isinstance(name, str):
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"input": arguments}
    arguments = _canonical(arguments)
    intent: dict[str, Any] = {"kind": "tool_call", "tool": name, "arguments": arguments}
    if name.lower() in _SHELL_TOOLS and isinstance(arguments, dict):
        command = arguments.get("command", arguments.get("cmd"))
        if isinstance(command, str):
            kind = _command_kind(command)
            if kind is not None:
                intent = {"kind": kind, "tool": name, "command": _normalize_command(command)}
    return intent


def _tool_blocks(value: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("type") == "tool_call":
            blocks.append(value)
        for item in value.values():
            blocks.extend(_tool_blocks(item))
    elif isinstance(value, list):
        for item in value:
            blocks.extend(_tool_blocks(item))
    return blocks


def read_transcript_intents(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read normalized response tool calls grouped by transcript session."""
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("event") != "normalized_response":
            continue
        session_id = record.get("session_id") or record.get("request_id")
        if not isinstance(session_id, str):
            continue
        for block in _tool_blocks(record.get("normalized")):
            intent = _intent(block)
            if intent is not None:
                sessions[session_id].append(intent)
    return dict(sessions)


def _intent_key(intent: dict[str, Any]) -> str:
    return json.dumps(intent, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_mining_report(
    sessions: dict[str, list[dict[str, Any]]], transcript_path: Path
) -> dict[str, Any]:
    """Build a deterministic report of exact repeated intents within sessions."""
    session_reports = []
    duplicate_total = 0
    for session_id in sorted(sessions):
        counts = Counter(_intent_key(intent) for intent in sessions[session_id])
        duplicates = [
            {"intent": json.loads(key), "count": count}
            for key, count in sorted(counts.items())
            if count > 1
        ]
        duplicate_total += sum(item["count"] - 1 for item in duplicates)
        session_reports.append(
            {
                "session_id": session_id,
                "intent_count": len(sessions[session_id]),
                "duplicate_intents": duplicates,
            }
        )
    return {
        "version": _REPORT_VERSION,
        "transcript": str(transcript_path.resolve()),
        "session_count": len(sessions),
        "intent_count": sum(len(intents) for intents in sessions.values()),
        "exact_duplicate_count": duplicate_total,
        "sessions": session_reports,
    }


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Transcript Mining Report",
        "",
        f"- Sessions: {report['session_count']}",
        f"- Intents: {report['intent_count']}",
        f"- Exact duplicates: {report['exact_duplicate_count']}",
        "",
    ]
    for session in report["sessions"]:
        lines.extend([f"## Session `{session['session_id']}`", ""])
        duplicates = session["duplicate_intents"]
        if not duplicates:
            lines.extend(["No exact duplicate intents.", ""])
            continue
        for duplicate in duplicates:
            rendered = json.dumps(duplicate["intent"], sort_keys=True, ensure_ascii=False)
            lines.append(f"- {duplicate['count']}× `{rendered}`")
        lines.append("")
    return "\n".join(lines)


def _duplicate_intents(report: dict[str, Any]) -> list[dict[str, Any]]:
    intents = {
        _intent_key(duplicate["intent"]): duplicate["intent"]
        for session in report["sessions"]
        for duplicate in session["duplicate_intents"]
    }
    return [intents[key] for key in sorted(intents)]


def _draft_name(intent: dict[str, Any]) -> str:
    digest = hashlib.sha256(_intent_key(intent).encode()).hexdigest()[:12]
    return f"intent-{digest}"


def write_mining_artifacts(
    bundle_dir: Path,
    report: dict[str, Any],
    *,
    emit_skills: bool,
    emit_tools: bool,
) -> None:
    """Write mining reports and optional generated draft artifacts."""
    (bundle_dir / "mining_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "mining_report.md").write_text(_report_markdown(report), encoding="utf-8")
    for intent in _duplicate_intents(report):
        name = _draft_name(intent)
        rendered = json.dumps(intent, indent=2, sort_keys=True, ensure_ascii=False)
        if emit_skills:
            directory = bundle_dir / "generated_skills"
            directory.mkdir(exist_ok=True)
            (directory / f"{name}.md").write_text(
                f"---\nversion: 1\nstatus: draft\nname: {name}\n---\n\n"
                f"# {name}\n\nAutomate this repeated transcript intent:\n\n```json\n{rendered}\n```\n",
                encoding="utf-8",
            )
        if emit_tools:
            directory = bundle_dir / "generated_tools"
            directory.mkdir(exist_ok=True)
            draft = {"version": 1, "status": "draft", "name": name, "intent": intent}
            (directory / f"{name}.json").write_text(
                json.dumps(draft, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
