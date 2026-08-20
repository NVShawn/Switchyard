#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Offline dream step, run from cron hourly. Read-only on the routing log,
# so the serving container keeps answering queries while this runs.
#
# Install in the user crontab (rights owned by the deploying user):
#   0 * * * * /home/<you>/Switchyard/deploy/dream_cron.sh >> /home/<you>/Switchyard/deploy/data/dream_cron.log 2>&1

set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load secrets from a gitignored env file so the crontab line stays clean.
if [ -f "$ROOT/deploy/.env" ]; then
  set -a
  . "$ROOT/deploy/.env"
  set +a
fi
: "${NVIDIA_API_KEY:?NVIDIA_API_KEY must be set in deploy/.env}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$NVIDIA_API_KEY}"

# Skip rather than overlap if a prior scheduled run is still re-judging tasks.
exec 9>"$ROOT/deploy/data/dream_cron.lock"
flock -n 9 || exit 0

"$ROOT/.venv/bin/switchyard" dream \
  --log "$ROOT/deploy/data/routing.jsonl" \
  --transcript "$ROOT/deploy/data/routing.transcript.jsonl" \
  --mine \
  --emit-skills \
  --emit-tools \
  --strong-model "nvidia/nvidia/nemotron-3-ultra" \
  --base-url "https://inference-api.nvidia.com/v1" \
  --api-key "$NVIDIA_API_KEY" \
  --out "$ROOT/deploy/data/dream_labels.jsonl"