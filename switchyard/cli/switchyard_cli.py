#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Switchyard command-line entry point."""

import argparse
from pathlib import Path

from switchyard import __version__
from switchyard.cli.launch_command import (
    cmd_launch_claude,
    cmd_launch_codex,
    cmd_launch_openclaw,
)
from switchyard.dream import cmd_dream_ui


def _label_mapping(value: str) -> str:
    label, separator, model = value.partition("=")
    if not separator or not label.strip() or not model.strip():
        raise argparse.ArgumentTypeError("expected TARGET=MODEL_ID")
    return f"{label.strip()}={model.strip()}"


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if (
        getattr(args, "label", [])
        and getattr(args, "emit_weights", None) is None
        and getattr(args, "emit_classifier", None) is None
    ):
        parser.error("--label-map requires --emit-weights or --emit-classifier")


def _add_launch_parser(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    launch = subparsers.add_parser(
        "launch",
        help="Launch a coding agent through the native server",
    )
    launch_sub = launch.add_subparsers(dest="launch_target", help="Agent to launch")
    launcher_parsers = (
        ("claude", "Launch Claude Code", "claude_args", cmd_launch_claude),
        ("codex", "Launch Codex CLI", "codex_args", cmd_launch_codex),
        ("openclaw", "Launch OpenClaw", "openclaw_args", cmd_launch_openclaw),
    )
    for name, help_text, args_dest, command in launcher_parsers:
        agent = launch_sub.add_parser(name, help=help_text)
        agent.add_argument("--model", required=True, help="Route ID from the deployment.")
        agent.add_argument(
            "--config",
            metavar="PATH",
            help="TOML deployment (default: packaged OpenRouter deployment).",
        )
        agent.add_argument(
            args_dest,
            nargs=argparse.REMAINDER,
            help="Arguments forwarded to the coding agent after --.",
        )
        agent.set_defaults(func=command)

    def _launch_help(args: argparse.Namespace) -> None:  # noqa: ARG001
        launch.print_help()
        raise SystemExit(1)

    launch.set_defaults(func=_launch_help)


def _add_dream_parser(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    dream = subparsers.add_parser(
        "dream",
        help="Offline dream-step: re-judge a routing log and score calibration",
    )
    dream.add_argument(
        "--log",
        required=True,
        type=Path,
        metavar="PATH",
        help="Routing log JSONL written by the server.",
    )
    dream.add_argument(
        "--out",
        type=Path,
        default=Path("dream_labels.jsonl"),
        metavar="PATH",
        help="Where to write fine-tune labels (default: dream_labels.jsonl).",
    )
    dream.add_argument(
        "--strong-model",
        help="Re-judge each logged task with this OpenAI-compatible model.",
    )
    dream.add_argument(
        "--transcript",
        type=Path,
        metavar="PATH",
        help="Transcript event JSONL (default: inferred from --log).",
    )
    dream.add_argument(
        "--mine",
        action="store_true",
        help="Mine transcript tool-call intents and exact duplicates.",
    )
    dream.add_argument(
        "--emit-skills",
        action="store_true",
        help="Emit draft skills for repeated intents (implies --mine).",
    )
    dream.add_argument(
        "--emit-tools",
        action="store_true",
        help="Emit draft tools for repeated intents (implies --mine).",
    )
    dream.add_argument(
        "--emit-weights",
        type=Path,
        metavar="PATH",
        help=(
            "Write learned-selection TOML weights to PATH from rewarded answer calls. "
            "Use the file on a custom llm_classifier target_selector route."
        ),
    )
    dream.add_argument(
        "--emit-classifier",
        type=Path,
        metavar="PATH",
        help=(
            "Write a tuned classifier prompt/schema TOML artifact to PATH from rewarded "
            "answer calls, plus a bundle-local (text -> best target) dataset. Overlay the "
            "file on the routes.auto custom llm_classifier target_selector route."
        ),
    )
    dream.add_argument(
        "--label-map",
        "--label",
        dest="label",
        action="append",
        default=[],
        type=_label_mapping,
        metavar="TARGET=MODEL_ID",
        help=(
            "Map a route target label to its logged model id for --emit-weights. "
            "Repeatable; unmapped model ids become target labels."
        ),
    )
    dream.add_argument(
        "--base-url",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible base URL for the strong model.",
    )
    dream.add_argument(
        "--api-key",
        help="API key for the strong model (default: $OPENAI_API_KEY).",
    )
    dream.add_argument(
        "--ui",
        action="store_true",
        help="Serve a local web UI to explore the dream run (binds to 127.0.0.1).",
    )
    dream.add_argument(
        "--ui-host",
        default="127.0.0.1",
        help="Host interface for the dream UI (default: 127.0.0.1).",
    )
    dream.add_argument(
        "--ui-port",
        type=int,
        default=8008,
        help="Port for the dream UI (default: 8008).",
    )
    dream.set_defaults(func=cmd_dream_ui)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="switchyard",
        description="Switchyard LLM proxy",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    _add_launch_parser(subparsers)
    _add_dream_parser(subparsers)
    return parser


def main() -> None:
    """Run the Switchyard CLI."""

    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(1)
    args.func(args)


if __name__ == "__main__":
    main()
