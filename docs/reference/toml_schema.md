# TOML Schema

The native deployment file defines the LLM clients, targets, and routes a
Switchyard server serves. It is read by `switchyard-server --config` and by
`switchyard launch --config`.

Validate a file without starting the server:

```bash
switchyard-server --config routes.toml --dry-run
```

## Minimal Example

```toml
schema_version = 1

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.strong]
id = "anthropic/claude-sonnet-4.5"
llm_client = "openrouter"

[routes.default]
id = "switchyard"
type = "passthrough"
target = "strong"
```

`schema_version` must be `1`. Table names under `llm_clients`, `targets`, and
`routes` are local references; clients send the route's `id` as the model name.

`schema_version`, `[targets]`, and `[routes]` must all be present, even when a
route reaches no upstream. A file without a `[targets]` table is rejected with
`missing field targets`; an empty `[targets]` table satisfies it.
`[llm_clients]` defaults to empty and may be omitted.

## `[llm_clients.<name>]`

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `format` | Yes | — | `openai_chat`, `openai_responses`, or `anthropic_messages`. |
| `base_url` | Yes | — | Upstream base URL. |
| `api_key_env` | No | unset | Name of the environment variable holding the key. Omit to send no authentication. |
| `forward_auth` | No | `false` | Forward the caller's provider credential to this upstream. |
| `extra_headers` | No | `{}` | Custom HTTP headers sent to the model server. Set credentials with `api_key_env` or `forward_auth`; the server rejects headers owned by the selected auth mode. Header names are case-insensitive. |
| `max_retries` | No | `2` | Retry budget, `0`–`10`. |

The TOML never contains the secret itself. `api_key_env` names a variable that
must exist and be non-empty when the server loads.

Set `forward_auth = true` to use each caller's credential instead of a
server-owned key:

```toml
[llm_clients.claude]
format = "anthropic_messages"
base_url = "https://api.anthropic.com"
forward_auth = true
```

`forward_auth` cannot be combined with `api_key_env`. OpenAI clients forward
`authorization`, `chatgpt-account-id`, and `x-openai-fedramp`. Anthropic clients
forward `authorization` or `x-api-key`; for Claude subscription OAuth, they also
forward `oauth-*` values from `anthropic-beta` and remove all other inbound beta
values.

This setting gives `base_url` the caller's login. Enable it only when that
upstream should receive the credential, and use HTTPS unless the upstream runs
on loopback. Forwarding clients do not follow HTTP redirects. Check every
forwarding client used by a route, including classifier and judge targets. The
server rejects an Anthropic forwarding route called through an OpenAI endpoint,
or an OpenAI forwarding route called through an Anthropic endpoint, before it
calls an upstream.

## `[targets.<name>]`

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `id` | Yes | — | Exact model ID sent upstream. |
| `llm_client` | Yes | — | Key under `[llm_clients]`. |
| `extra_body` | No | `{}` | Values merged into the upstream request when the request does not already set that key. |
| `cost` | No | unset | Unit pricing for the target, used by cost-aware routes and the routing log. A table with `input_per_1m` and `output_per_1m` (USD per 1M tokens), both non-negative and finite. |

## `[routes.<name>]`

Every route takes the common keys below, plus the keys for its type.

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `id` | Yes | — | Public model ID that callers send in requests. |
| `type` | Yes | — | Routing algorithm for this route. |
| `context_window` | No | unset | Positive token count advertised for this route by `GET /v1/models`. Unset values appear as `null`. This does not enforce a request limit. |
| `tool_calling` | No | unset | Whether `GET /v1/models` advertises tool-calling support for this route. Unset values appear as `null`. |
| `reasoning` | No | unset | Whether `GET /v1/models` advertises reasoning support to Codex direct-provider discovery. Unset routes are advertised as non-reasoning. |

### `noop`

Returns a buffered assistant response containing `OK` without calling an
upstream model. Use it for local smoke tests.

A noop-only deployment reaches no upstream but still needs the `[targets]`
table, which can be empty:

```toml
schema_version = 1

[targets]

[routes.smoke]
id = "noop-route"
type = "noop"
```

### `passthrough`

Sends every request to one target.

| Key | Required | Meaning |
|---|:---:|---|
| `target` | Yes | Target every request is sent to. |

### `random`

Splits traffic across targets. See
[Random Routing](../routing_algorithms/random_routing.md).

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `targets` | Yes | — | Target names to choose from. |
| `weights` | No | equal | Finite, non-negative relative weights in `targets` order, with at least one positive value. Invalid weights are rejected at load time. |
| `seed` | No | unset | Reproduces the selection sequence. |

### `llm_classifier`

Runs one of three judge-backed modes: `capability`, `escalation`, or `custom`.
`classifier_target` and `max_output_tokens` apply to all three.

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `mode` | No | `capability` | Classifier behavior. Set it explicitly for new configurations. |
| `classifier_target` | Yes | — | Target the judge is called through. Not a routing destination. |
| `max_output_tokens` | No | `4096` | Maximum completion tokens for the judge verdict. Must be at least `1`. |
| `response_format_type` | No | `json_schema` | Structured-output mode for capability and escalation judges. Use `json_object` when the provider does not support JSON Schema; Switchyard adds the schema to the prompt and validates the verdict locally. Custom mode always uses its configured JSON Schema. |

Capability mode classifies before serving. See
[LLM Classifier Routing](../routing_algorithms/llm_classifier_routing.md).

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `strong_target` | Yes | — | Capable tier. |
| `weak_target` | Yes | — | Efficient tier. |
| `base_threshold` | Yes | — | Lowest solve probability that routes to the weak target. In `[0, 1]`. |
| `threshold_step` | No | `0.0` | Finite, non-negative amount added once for uncertain or unmatched verdicts and twice for unsupported verdicts. `base_threshold + 2 * threshold_step` must be at most `1`. |
| `session_affinity` | No | `false` | Reuses a session's first decision on later turns. |
| `message_hash_fallback` | No | `false` | Keys affinity on the first user message. Requires `session_affinity`. |
| `recent_turn_window` | No | unset | When unset, the judge sees the opening task and latest user follow-up, when present. When set, it also sees trailing turns. |
| `prompt` | No | packaged prompt | Replaces the capability prompt. The packaged schema is sent separately as structured-output configuration. |

Capability mode can also route among more than two tiers. `capability_targets`
declares a cost-ordered ladder; each rung is `{ target, capability, context_window? }`
where `capability` is a static level in `[0, 1]` and `context_window` is the rung's
token limit for the headroom prefilter. The judge's verdict adds a
`minimum_capability` level, and Switchyard picks the cheapest rung whose declared
`capability` clears it and whose `context_window` fits the request. Every rung's
target must declare a `cost`. When `capability_targets` is set, `strong_target` and
`weak_target` default to the most and least capable rungs.

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `capability_targets` | No | unset | Ordered ladder of `{ target, capability, context_window? }` rungs for cost-aware routing. Requires a `cost` on every rung's target. |

Two optional sub-tables layer on a ladder. `[routes.<name>.zones]` turns on
three-zone routing: a high-confidence answer goes to the cheapest adequate rung
in a single call, a low-confidence or unsupported one goes straight to the most
capable rung, and the mid band fans out to `fan_out` rungs concurrently and lets
an output judge pick the best answer. `[routes.<name>.bandit]` turns on a
Thompson-sampling correction that nudges the judge's confidence from observed
per-rung rewards recorded in the routing log (`--routing-log-file`).

`[routes.<name>.zones]`:

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `low_threshold` | Yes | — | Below this solve probability, route to the most capable rung. |
| `high_threshold` | Yes | — | At or above this, answer with one cheapest-adequate call. Must satisfy `0 <= low_threshold <= high_threshold <= 1`. |
| `fan_out` | Yes | — | How many eligible rungs the mid band calls concurrently. At least `1`. |
| `output_judge_target` | Yes | — | Target that judges the fanned-out candidate answers. Not a routing destination. |
| `output_judge_prompt` | No | packaged prompt | Replaces the output-judge prompt. |
| `output_judge_max_output_tokens` | No | `4096` | Maximum completion tokens for the output-judge verdict. |

`[routes.<name>.bandit]` requires `zones`:

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `scale` | Yes | — | How far one bandit sample can move the judge's solve probability: `corrected = p_solve + (sample - 0.5) * scale`. Non-negative and finite. |

Escalation mode serves the weak target first and judges the completed turn. See
[Escalation-Router Routing](../routing_algorithms/escalation_router_routing.md).

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `strong_target` | Yes | — | Target used after the session latches. |
| `weak_target` | Yes | — | Target served before the latch. |
| `prompt` | No | packaged prompt | Replaces the trajectory-judge prompt. |
| `escalation.confirmations` | No | `2` | Consecutive escalate verdicts required to latch. Above `1` needs a session ID. |
| `escalation.recent_turn_window` | No | `28` | Trailing messages shown to the judge. |
| `escalation.window_message_chars` | No | `500` | Per-message cap inside that window. |

Existing configurations that contain `escalation` but omit `mode` remain valid.

Custom mode validates the judge's JSON against `response_schema`, resolves the
policy selector, and routes to any configured target label.

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `targets` | Yes | — | Two or more target names available to the policy. |
| `default_target` | Yes | — | Target used when the judge fails or its verdict cannot be routed. |
| `prompt` | Yes | — | Judge system prompt. The configured inner schema is sent separately as structured-output configuration. |
| `response_schema` | Yes | — | Inner JSON Schema encoded as a TOML string. Switchyard adds the provider wrapper. |
| `policy` | Yes | — | Policy table. `target_selector` accepts a JSON Pointer such as `/decision/target`. |
| `session_affinity` | No | `false` | Reuses a session's first decision on later turns. |
| `message_hash_fallback` | No | `false` | Keys affinity on the first user message. Requires `session_affinity`. |
| `recent_turn_window` | No | unset | When unset, the judge sees the opening task and latest user follow-up, when present. When set, it also sees trailing turns. |

Classifier prompts must not contain `{{RESPONSE_SCHEMA}}`. Switchyard supplies
the schema automatically: through the structured-output request in `json_schema`
mode, or in the prompt in `json_object` mode.

`[routes.<name>.learned_selection]` layers deterministic learned routing on a
custom `target_selector` route. When present, the judge's verdict is fully
replaced: Switchyard reads the weights file and routes every request to the
target with the highest observed posterior-mean reward, immediately and with no
exploration. Produce the file offline with `switchyard dream --emit-weights
weights.toml --label-map <target>=<logged-model-id>`; it holds one `[[target]]` entry per
label with `alpha`/`beta` Beta parameters.

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `weights_path` | Yes | — | Path to the learned per-target weights TOML file. Each `[[target]]` needs `label`, `alpha`, and `beta`, all positive finite. |

### `stage_router`

Scores tool signals to pick a tier per turn. See
[Stage-Router Routing](../routing_algorithms/stage_router_routing.md) for the
optional `handoff_notes` and `classifier` tables and for tuning.

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `capable_target` | Yes | — | Capable tier. |
| `efficient_target` | Yes | — | Efficient tier. |
| `picker` | Yes | — | `efficient_first`, or `capable_first` (experimental, unbenchmarked). Tier used when the signals are not confident. |
| `confidence_threshold` | Yes | — | Corroboration a decisive pick needs. In `[0, 1]`. |
| `recent_turn_window` | No | `3` | Trailing tool results the signals are computed over. |
| `capable_system_prompt` | No | unset | System prompt handed to the capable tier. |
| `efficient_system_prompt` | No | unset | System prompt handed to the efficient tier. |
| `classifier.response_format_type` | No | `json_schema` | Structured-output mode for the optional classifier judge. Use `json_object` when the classifier provider does not support JSON Schema; Switchyard adds the schema to the prompt and validates the verdict locally. |

## Validation Errors

`--dry-run` prefixes configuration failures with
`invalid server config <path>:`. Within that wrapper, TOML deserialization
errors start with `failed to parse TOML:`, while errors from validating the
built configuration retain their inner message unchanged.

## Related Documentation

- [CLI Reference](../cli_reference.md)
- [Routing Overview](../routing_algorithms/overview.md)
