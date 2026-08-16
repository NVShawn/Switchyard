# Cost-Aware Routing: Design Plan

Status: Implemented (phases 1–5). Not part of the published MkDocs site.

Implementation notes are inline per phase. The headline mechanisms:

- `capability_targets = [{ target, capability }, ...]` on a capability route declares
  the cost-ascending ladder (unit cost inherited from each target's `cost`).
- `[routes.x.zones]` turns on the three-zone fan-out policy with an output judge.
- `[routes.x.bandit]` turns on the Thompson-sampling confidence correction, refreshed
  from the routing log (`--routing-log-file`).
- `switchyard dream --log <routing.jsonl>` runs the offline dream step.

## Objective

Route each request to the **minimum-cost model that will correctly answer it**.
A judge predicts task capability per request; cost breaks ties among
judged-adequate models. If the cheap model would miss, correctness outweighs
cost, so the router must hold a high-adequacy bar rather than optimise pure
cost. Over time the judge improves from observed outcomes.

This is Switchyard's judge-backed classifier (the request-understanding half)
plus TokenHub-style cost metadata and a feedback loop (the selection/learning
half). Ideas stolen from
[`jordanhubbard/tokenhub`](https://github.com/jordanhubbard/tokenhub):

- Per-model pricing (`InputPer1K`/`OutputPer1K`) and `estimateCostUSD`.
- Eligibility prefilters: disabled models, context headroom (1.15x), health/cooldown.
- Model-hint pinning: an explicit hint routes deterministically to that target.
- Thompson-sampling bandit per (model, token bucket) and reward logging.
- Rarely used here: TokenHub's weighted cost/latency/failure blend and softmax
  exploration. Switchyard's judge is the decision-maker; blending costs into
  capability scores would reintroduce the cheap-but-wrong error this design
  exists to kill. Cost is a **tiebreaker within judged-adequate targets**, not a
  competing signal.

## Core decision function

```
eligible = prefilters(targets, request)          # context headroom, health, hint pin
             .filter(|t| t.capability >= need(request))   # judge: minimum capability level
ranked   = sort_by_cost(eligible)                # ascending
pick     = ranked.first()                        # cheapest judged-adequate model
```

The judge emits one `p_solve` plus a constant-size verdict (not per-target
scores), so the contract scales to any number of targets. Each target declares
a static `capability` level in `routes.toml` (like TokenHub's `Weight`); the
router picks the cheapest target whose level clears the judge's required
level. `capability_boundary` still prunes impossible/unsupported tiers.

Three-zone behaviour emerges from the same verdict: high `p_solve` routes
single-target cheap; mid `p_solve` (judge unsure between a few models) fans out
to candidates concurrently and an output judge picks the winner; low `p_solve`
routes to the capable target directly. A feedback loop then corrects the
judge's priors from observed outcomes (online via a bandit, offline via
strong-model distillation).

## Phases

### 1. Cost metadata on targets — DONE

Parse `cost` per target, add a `TargetCost` type and an `estimate_cost(usage)`
helper. Delivered and verified:

- `switchyard-protocol`: `TargetCost { input_per_1m, output_per_1m }` with
  `estimate_usd(&Usage)`; bills non-cached + cached + cache-creation input at
  the input rate and output at the output rate; `None` when usage lacks counts.
- `switchyard-server`: `TargetConfig.cost` (serde default, non-negative,
  finite validation) at `config.rs:269`.
- `routes.toml`: cost block on all 10 targets, cost-ordered cheap→expensive.
- Tests: protocol cost math (2), config parse + negative rejection (2).
  `switchyard-server --config routes.toml --dry-run` OK.

### 2. Cost-aware multi-target policy — DONE

Extend the capability-mode verdict and policy to pick the **cheapest eligible
target**, not just binary strong/weak.

Config surface (routes.toml, `CapabilityClassifierRouteConfig`):
- An optional `capability_targets = [{ target, capability }, ...]` list on the
  capability route declares the ladder; `cost` is inherited from each named
  target's `TargetConfig.cost` (validated to exist on every rung). `strong_target`
  /`weak_target` remain for the binary path and default to the ladder endpoints.
  The verdict schema gains an optional `minimum_capability` (0..1).
- Keep `base_threshold` / `threshold_step`; they still map
  `capability_boundary` to a required `minimum_capability` level.
  `minimum_capability` field in routes.toml sets the default required level for
  supported verdicts; capability-level math reuses `boundary_steps()`.

Verdict contract (`capability-classifier/schema.json`):
- `p_solve` stays. Add nothing per-target (keeps schema constant-size).

Policy (`TaskClassifierPolicy` in `llm_class.rs`):
- `to_classification` currently pushes one `Score` for strong or weak
  (`llm_class.rs:223`). Change it to filter the ordered target list to
  `capability >= minimum_capability` and return the whole filtered set ranked
  by cost, letting the router's `argmax`/fallback consume the cheapest first.
  Reuse `Classification::Scores(Vec<Score>)` and existing fallback plumbing.
- Context headroom: reuse the route's `context_window` capability and skip
  targets whose window cannot fit the request (TokenHub's 1.15x rule).

Verify:
- libsy unit test: 4 tiers (nano < strong < ultra < opus) in `routes.toml`
  cost order; p_solve 0.95 + supported → nano; p_solve 0.4 + uncertain → strong;
  p_solve 0.2 + unsupported → opus (default/capable).
- Server config test: `[[route.targets]]` with capability + cost parses; a
  target missing `cost` on a cost-aware route errors with a clear message.
- `--dry-run` still OK. clippy/fmt/test suites green.

### 3. Three-zone routing (fan-out on uncertainty) — DONE

Zone boundaries on the existing `p_solve` × capability machinery:

- **Zone A** (verdict supported, `p_solve` ≥ `high_threshold`): cheapest
  eligible target, single call, no fan-out.
- **Zone B** (mid `p_solve` between `low_threshold` and `high_threshold`):
  fan out to the top-k eligible targets **concurrently**, run an output judge,
  return the judge's pick. The `Driver` already serves concurrent
  `CallModel` steps (`algorithm.rs` drives them in parallel), so this is
  several `CallModel` steps issued before waiting.
- **Zone C** (verdict unsupported/uncertain, `p_solve` < `low_threshold`):
  capable target directly.

New pieces:
- `[[route.zones]]` config: `{ low_threshold, high_threshold, fan_out_k }`
  plus an `[route.output_judge]` target (the model that ranks the candidates'
  completions) and a comparison prompt/contract, mirroring the existing judge
  contract pattern (`classifier_contract.rs`, `llm_judge.rs`).
- A small `compare` policy in libsy: after the fan-out returns k completions,
  issue one judge call that returns the winning candidate index. This is the
  same `JudgeClassifier` machinery, a different verdict schema.
- Fan-out cost/latency is gated to Zone B plus a small epsilon; Zone A/C never
  pay it.

Verify:
- libsy test: route with 3 targets fans out for a mid-p_solve verdict and the
  output judge picks the intended winner (mock judge). `Driver` concurrency
  asserted via the existing in-flight barrier pattern.
- Server integration test mirrors the streaming tests already in
  `tests/server.rs`: candidates return distinguishable prose; the winner is
  returned. gated boundaries: high and low p_solve never fan out.

### 4. Feedback loop (online) — DONE

Persist a reward per routed call so the router learns the model-choice prior.

Reward record — modelled on TokenHub's `RewardLog` + `ComputeReward`:
`{ ts, request_id, session_id, model, tier, verdict, p_solve,
 input/cached/output tokens, cost_usd (via `TargetCost::estimate_usd`),
 latency_ms, success, error_class, reward }` where
`reward = success ? (1-cost_norm)*0.3 + (1-latency_norm)*0.3 + 0.4 : 0`,
mirroring TokenHub's cost/latency/success blend, applied to **actual usage**
already captured in `routing_log.rs`.

Integration:
- Extend `RoutingRecord` (`routing_log.rs:128`) with cost and outcome fields
  (serde defaults so old records still parse — the file already does this).
  Compute `cost_usd` at record time from the model's `TargetCost`.
- The reward record is written per completed request (success and failure
  alike, at the same single call site that writes `RoutingRecord`), not only
  on success — the bandit needs the denominator (count) to compute β.

Thompson sampler sub-step (separate commit, config-flippable):
- Port TokenHub's arm shape: arm = `(target, token_bucket)` with a `Beta(α,β)`
  prior; token bucket labelled from an estimate of the request's input tokens
  (`small` / `medium` / `large`), like `TokenBucketLabel`. `UpdateArm`,
  `Sample`, and Marsaglia–Tsang `betaSample` copy over directly.
- **Selection-side correction, not replacement** (the deliberate divergence
  from TokenHub): the sampler never replaces the judge. It draws one sample
  per candidate and *adjusts the judge's `p_solve` prior for that candidate* —
  e.g. `effective_p_solve = clamp(p_solve + (sample - 0.5) * scale)` — before
  the Zone classification / cost pick. Stochastic sampling is the exploration,
  so the prior cannot collapse. Removing it must be a config flip, not a
  refactor.
- **Refresh loop, not inline updates** (TokenHub's `StartRefreshLoop`, ported):
  on boot, refresh the sampler **immediately**; then on an interval (default
  5 min) re-aggregate `(target, token_bucket) → count, sum(reward)` from the
  reward log and update arms in place. This makes the log the single source
  of truth and the sampler a pure derive — cheap to rebuild, no separate
  learned-state file to keep consistent. `alpha = sum(reward) + 1`,
  `beta = max(count - sum(reward) + 1, 1.0)` (β floored so a row can never
  produce a degenerate distribution).
- Reward math (TokenHub's `ComputeReward`, adapted to Switchyard's target
  abstraction): `reward = success ? (1-cost_norm)*0.3 + (1-latency_norm)*0.3
  + 0.4 : 0.0`, with `cost_norm` and `latency_norm` normalized against the
  route's budget/latency config; **failure ⇒ zero reward** (the cost/latency
  terms are wasted, no partial credit).

Verify:
- Integration test: a routed call appends a record whose `cost_usd` equals
  `TargetCost::estimate_usd(usage)` and whose `reward` is 0-1 normalized;
  failures record `success = false`, `reward = 0.0`.
- Bandit sub-step unit test: repeated failures on one arm shift selection away
  from it given otherwise-equal priors (deterministic via an injected RNG).
- Refresh test: new reward rows change a later `Sample`'s ordering once the
  refresh loop re-aggregates them.

### 5. Dream-step judge refinement (offline) — DONE (scoped)

Implemented as the `switchyard dream` CLI subcommand (`switchyard/dream.py`), a
stdlib-only offline tool that reads the routing log and:

1. Re-derives each `(model, token bucket)` arm's Beta posterior from logged
   rewards — the offline mirror of the online bandit's priors.
2. Reports per-arm calibration (mean reward, success rate) and the
   cheap-but-wrong rate the router should drive down.
3. With `--strong-model`, re-judges each logged `task` header via an
   OpenAI-compatible endpoint and emits fine-tune labels in the judge's contract
   (`p_solve`, `capability_boundary`, `minimum_capability`, `crux`,
   `primary_rule`), each paired with the observed outcome so the trainer weights
   cheap-but-wrong samples highest.
4. Scores the teacher's calibration (Brier of its `p_solve` against observed
   outcomes) as the gate on trusting those labels.

Scope notes / deliberate deferrals:
- Re-judging keys on the logged `x-switchyard-intake-task` header; the serving
  path does **not** log conversation bodies or the judge's own verdict, so the
  dream step re-judges the task summary, not the full transcript. Logging
  (redacted) judge inputs/verdicts is a follow-up if richer labels are needed.
- Fine-tuning itself is external training infra and out of scope; the tool emits
  the labels and the calibration gate, it does not train.
- The teacher's Brier calibration gates promotion; the actual refined-judge eval
  runs wherever fine-tuning runs, not in this repo.

Verify: `tests/test_dream.py` covers arm aggregation, cheap-wrong rate, Brier
scoring, label emission (with an injected judge, no network), and teacher
calibration.

## Assumptions

- Cost is computed from **actual** reported usage (already recorded), not a
  pre-call token estimate. Estimates are only needed at select time if we add
  input-token prediction later.
- Per-target capability levels are **static, declared in routes.toml**, not
  judged per request — keeps the verdict contract constant-size.
- Zone boundaries (`low_threshold`/`high_threshold`) are config, not learned.
- Phases 1-3 are the priority; phase 4's schema is needed by phase 5.
- Truth for Phase 5 labels is the downstream conversation outcome, never the
  internal judge preference, to avoid reinforcing the judge's own biases.

## Integration surfaces

- `switchyard-protocol`: `TargetCost` (+`estimate_usd`), next to `Usage` (Phase 1).
- `switchyard-server/src/config.rs`: `TargetConfig.cost` (Phase 1);
  `capability_targets` ladder with capability + inherited cost (Phase 2);
  `[routes.x.zones]` fan-out + output judge (Phase 3); `[routes.x.bandit]`
  correction (Phase 4).
- `switchyard-libsy/src/algorithms/llm_class.rs`: `CapabilityTarget` ladder and the
  ranked `TaskClassifierPolicy` pick (Phase 2); `CostAwareClassifier` three-zone
  fan-out + `ZoneConfig` and the output-judge contract under
  `prompts/output-judge/` (Phase 3); `BanditConfig` confidence correction
  (Phase 4).
- `switchyard-libsy/src/algorithms/util/thompson.rs`: `ThompsonSampler` bandit,
  `token_bucket`, `estimate_request_tokens` (Phase 4).
- `switchyard-server/src/routing_log.rs`: `RoutingRecord` gains `cost_usd`,
  `latency_ms`, `success`, `reward`, `token_bucket`; `reward_summary` aggregates
  arms for the refresh loop (`refresh_bandit`, replayed at boot and every 5 min)
  (Phase 4).
- `switchyard/dream.py` + the `switchyard dream` subcommand: offline label and
  calibration tooling (Phase 5).
- `docs/internal/cost_aware_routing.md`: this document.