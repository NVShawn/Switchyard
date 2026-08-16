# Persistence and the Offline Dream Step

Switchyard keeps no database. Its only durable state is the routing log, an
append-only JSONL file. Everything else — session affinity, live `/v1/stats`
counters — is in-memory and resets on restart by design.

## The routing log

Start the server with `--routing-log-file` to record every routed call's model,
token usage, cost, latency, success, reward, token bucket, and (for cost-aware
routes) the judge's verdict:

```bash
switchyard-server --config routes.toml --routing-log-file /data/routing.jsonl
```

The log drives two things: per-session stats (`GET /v1/routing/session-stats`)
and, for bandit-enabled routes, the feedback loop.

## Surviving restarts

To keep the log across container restarts, put it on a mounted volume and point
the flag at a path on that volume:

```bash
switchyard-server --config routes.toml --routing-log-file /data/routing.jsonl
# run the container with /data backed by a volume or persistent volume claim
```

What survives a restart, and how:

- **The routing log** persists only if its path is on a mounted volume.
- **Bandit priors** survive *through* the log. The sampler holds no durable
  state of its own; at boot it replays the log to rebuild its arms, then
  re-aggregates every five minutes. A restart loses nothing the log recorded.
- **Session affinity** does not survive. It is an in-memory, per-session
  optimization with a one-hour TTL.
- **Live stats** (`/v1/stats`) do not survive. They are in-memory telemetry.

## Running the dream step out of band

`switchyard dream` is offline analysis, never part of the request path. It reads
a routing log and reports per-arm calibration and the serving judge's own
calibration; with `--strong-model` it also re-judges logged tasks into fine-tune
labels. See [CLI Reference](../cli_reference.md).

Run it outside the serving container, against the persisted log:

- from a host that mounts the same volume,
- as a separate scheduled job or sidecar that shares the log's volume, or
- against a copy of the log file.

The serving process only needs the log on the volume; the dream step only needs
to read that file and, for re-judging, reach an OpenAI-compatible model
(`--api-key` or `OPENAI_API_KEY`).

## Related Documentation

- [Context-Window Handling](context_window.md)
- [CLI Reference](../cli_reference.md)
- [TOML Schema](../reference/toml_schema.md)
