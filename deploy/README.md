# Standalone Docker deployment

Serves the repo's `routes.toml` (prediction support) in Docker on port `4000`
and appends durable routing records to a host-writable data dir so the offline
`switchyard dream` step can run outside the container against the same log.

## Run the serving container

```bash
docker compose -f deploy/docker-compose.yml up -d --build
curl http://localhost:4000/v1/models
```

- Config: `routes.toml`, bind-mounted read-only. Its NVIDIA client uses
  `forward_auth = true`, so every route requires each caller to send its own
  `Authorization: Bearer <NVIDIA_API_KEY>` token. The server keeps no shared
  serving token; all callers still contribute to the same routing, transcript,
  Dream, and classifier datasets.
- Routing log: `<data dir>/routing.jsonl`, where `<data dir>` is
  `deploy/data` by default or `$SWITCHYARD_DATA_DIR`. The container runs as
  UID 1000; if the directory is owned by root, the log cannot be written.
- Transcript log: `<data dir>/routing.transcript.jsonl`, enabled by
  `--transcript-log-file`. Records are redacted (`strict`) and size-capped by
  default; they may still contain prompts and tool content, so restrict access
  and set a retention policy. The file is created owner-only.
- Data directory must exist and be owned by UID 1000 before first start
  (created for you, or `mkdir -p deploy/data`).
- Logs: `docker compose -f deploy/docker-compose.yml logs -f`.

## Run the dream step outside the container

The dream step is offline analysis; it never runs in the request path. On the
host, point it at the same log file:

```bash
# Summary only
uv run switchyard dream --log deploy/data/routing.jsonl

# Summary + re-judge each logged task with a strong model (fine-tune labels)
uv run switchyard dream --log deploy/data/routing.jsonl \
  --strong-model nvidia/nvidia/nemotron-3-ultra \
  --base-url https://inference-api.nvidia.com/v1 --out dream_labels.jsonl
```

### Picking up post-dream data without a restart

The serving process re-reads the routing log on an interval, so continuing
traffic (and any records appended while dream analyzed the file) is picked up
without a container restart:

- Bandit-enabled routes rebuild their priors from the log every five minutes,
  with the first pass on startup.
- Per-session stats (`GET /v1/routing/session-stats`) rescan the log on demand.

Dream itself writes nothing the server consumes. It is read-only on the log;
its output (`--out`) is a fine-tune label file for training, not a routing
input.

## Scheduled dream step (cron)

`deploy/dream_cron.sh` runs the same out-of-band command above on a schedule.
It reads secrets from `deploy/.env` (gitignored), so the crontab line stays
clean:

```bash
0 * * * * /home/<you>/Switchyard/deploy/dream_cron.sh >> /home/<you>/Switchyard/deploy/data/dream_cron.log 2>&1
```

- Secrets: `deploy/.env` mode `600`, e.g. `NVIDIA_API_KEY=sk-or-...`. `docker compose up` reads the same file automatically. The cron script aliases it to `OPENAI_API_KEY` for the OpenAI-compatible judge client.
- Concurrency: a non-blocking `flock` prevents overlapping scheduled runs; a new run is skipped if the prior one is still active.
- Output: labels to `deploy/data/dream_labels.jsonl`; cron stdout to `deploy/data/dream_cron.log`.
- Mining: the scheduled command passes `--mine --emit-skills --emit-tools` and
  reads `deploy/data/routing.transcript.jsonl`, writing a versioned
  `deploy/data/dream_labels.jsonl.<run>.dream/` bundle with `mining_report.{json,md}`
  and `generated_skills/`, `generated_tools/` drafts.

## Browse dream runs (local web UI)

The UI is a long-lived process, so run it under a supervisor (e.g. a systemd
user service), not from cron. It binds to `127.0.0.1:8008` and serves current
and historical run bundles next to `--out`:

```bash
uv run switchyard dream \
  --log deploy/data/routing.jsonl \
  --transcript deploy/data/routing.transcript.jsonl \
  --out deploy/data/dream_labels.jsonl \
  --mine --emit-skills --emit-tools --ui
```

Without `--ui`, the same command runs the batch dream step and exits, which is
what the cron job relies on.

## Stop

```bash
docker compose -f deploy/docker-compose.yml down
```