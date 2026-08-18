// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable per-request routing records and session snapshots.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use humantime::format_rfc3339_millis;
use serde::{Deserialize, Serialize};
use switchyard_protocol::{Metadata, ModelId, Usage};

use crate::usage_metrics::token_usage;
use crate::{ServerError, ServerResult};

const LEGACY_SESSION_ID_HEADER: &str = "proxy_x_session_id";
const TASK_HEADER: &str = "x-switchyard-intake-task";
const TRIAL_ID_HEADER: &str = "x-switchyard-trial-id";

/// Tier label written for judge/classifier calls, which are not bandit arms.
const CLASSIFIER_TIER: &str = "classifier";

/// Reward normalization: this USD cost maps to a fully "expensive" call.
const REWARD_COST_SCALE_USD: f64 = 0.1;
/// Reward normalization: the latency budget a call is measured against.
const REWARD_LATENCY_BUDGET_MS: f64 = 20_000.0;

/// The terminal outcome of one routed or classifier call, for reward accounting.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Outcome {
    /// Estimated USD spent, when the target has a cost and usage was reported.
    pub(crate) cost_usd: Option<f64>,
    /// Wall-clock latency of the call, when measured.
    pub(crate) latency_ms: Option<f64>,
    /// Whether the call completed without a transport or upstream error.
    pub(crate) success: bool,
    /// Normalized 0..1 reward. `None` for classifier calls, which are not bandit arms.
    pub(crate) reward: Option<f64>,
}

/// The capability judge's verdict for a routed call, surfaced so the offline dream
/// step can score the real judge's calibration against observed outcomes.
#[derive(Clone, Debug, Default)]
pub(crate) struct JudgeVerdict {
    /// The judge's solve-probability estimate.
    pub(crate) p_solve: Option<f64>,
    /// The judge's capability boundary classification.
    pub(crate) capability_boundary: Option<String>,
    /// The judge's minimum required capability level.
    pub(crate) minimum_capability: Option<f64>,
}

impl JudgeVerdict {
    /// Reads the verdict the classifier stamped into the request's `extra_metadata`.
    /// Absent keys (a non-capability route) yield a default (empty) verdict.
    pub(crate) fn from_metadata(metadata: &Metadata) -> Self {
        let Some(extra) = metadata.extra_metadata.as_ref() else {
            return Self::default();
        };
        Self {
            p_solve: extra
                .get("switchyard.judge.p_solve")
                .and_then(|value| value.parse().ok()),
            capability_boundary: extra.get("switchyard.judge.capability_boundary").cloned(),
            minimum_capability: extra
                .get("switchyard.judge.minimum_capability")
                .and_then(|value| value.parse().ok()),
        }
    }
}

/// Computes a 0..1 reward from cost and latency, blended with a success bonus.
///
/// A failed call scores 0.0 — the cost and latency were spent without a usable
/// answer, so there is no partial credit. The blend mirrors TokenHub's
/// cost/latency/success weighting.
pub(crate) fn compute_reward(cost_usd: Option<f64>, latency_ms: f64, success: bool) -> f64 {
    if !success {
        return 0.0;
    }
    let cost_norm = (cost_usd.unwrap_or(0.0) / REWARD_COST_SCALE_USD).min(1.0);
    let latency_norm = (latency_ms / REWARD_LATENCY_BUDGET_MS).min(1.0);
    (1.0 - cost_norm) * 0.3 + (1.0 - latency_norm) * 0.3 + 0.4
}

/// Coarse token bucket for bandit context, labelled from an estimated token count.
pub(crate) fn token_bucket(estimated_tokens: u64) -> &'static str {
    if estimated_tokens < 1_000 {
        "small"
    } else if estimated_tokens <= 10_000 {
        "medium"
    } else {
        "large"
    }
}

/// Append-only writer for one routing JSONL file.
pub(crate) struct RoutingLog(fs::File);

impl RoutingLog {
    pub(crate) fn new(path: impl Into<PathBuf>) -> ServerResult<Self> {
        let path = path.into();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|error| routing_log_error(&path, error))?;
        }
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|error| routing_log_error(&path, error))?;
        Ok(Self(file))
    }

    pub(crate) fn append(
        &mut self,
        context: RoutingLogContext,
        model: &str,
        tier: Option<&str>,
        usage: Option<&Usage>,
        outcome: Outcome,
        verdict: JudgeVerdict,
    ) -> std::io::Result<()> {
        let usage = usage.map(token_usage).unwrap_or_default();
        let record = RoutingRecord {
            ts: format_rfc3339_millis(SystemTime::now()).to_string().into(),
            task: context.task.map(Cow::Owned),
            trial_id: context.trial_id.map(Cow::Owned),
            session_id: context.session_id.map(Cow::Owned),
            request_id: context.request_id.map(Cow::Owned),
            model: model.into(),
            tier: tier.unwrap_or("").into(),
            prompt_tokens: usage.prompt_tokens,
            cached_tokens: usage.cached_tokens,
            cache_creation_tokens: usage.cache_creation_tokens,
            completion_tokens: usage.completion_tokens,
            reasoning_tokens: usage.reasoning_tokens,
            total_tokens: usage.prompt_tokens.saturating_add(usage.completion_tokens),
            cost_usd: outcome.cost_usd,
            latency_ms: outcome.latency_ms,
            success: Some(outcome.success),
            reward: outcome.reward,
            token_bucket: context.token_bucket.map(Cow::Owned),
            judge_p_solve: verdict.p_solve,
            judge_capability_boundary: verdict.capability_boundary.map(Cow::Owned),
            judge_minimum_capability: verdict.minimum_capability,
        };
        let mut line = serde_json::to_vec(&record).map_err(std::io::Error::other)?;
        line.push(b'\n');

        self.0.write_all(&line)
    }
}

/// Reads complete records without synchronizing with the writer.
pub(crate) fn snapshot(
    path: &Path,
    session_id: &str,
) -> std::io::Result<Option<SessionStatsSnapshot>> {
    let mut reader = BufReader::with_capacity(64 * 1024, fs::File::open(path)?);
    let mut line = Vec::new();
    let mut snapshot = SessionStatsSnapshot::new(session_id);
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        if !line.ends_with(b"\n") {
            break;
        }
        let Ok(record) = serde_json::from_slice::<RoutingRecord>(&line) else {
            continue;
        };
        snapshot.add_record(&record, session_id);
    }
    snapshot.sum_totals();
    Ok((snapshot.total_calls > 0).then_some(snapshot))
}

/// Aggregated reward for one bandit arm — a (model, token bucket) pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RewardSummary {
    /// Number of answer calls recorded for the arm.
    pub(crate) count: u64,
    /// Sum of their rewards (soft success count for the Beta prior).
    pub(crate) sum_reward: f64,
}

/// Aggregates answer-call rewards per (model, token bucket) arm from the log.
///
/// Only answer calls carry a reward, so classifier calls and legacy records are
/// skipped. This is the read side of the feedback loop: replayed at startup and
/// re-aggregated on the refresh interval to rebuild the bandit's priors.
pub(crate) fn reward_summary(
    path: &Path,
) -> std::io::Result<std::collections::BTreeMap<(String, String), RewardSummary>> {
    let mut reader = BufReader::with_capacity(64 * 1024, fs::File::open(path)?);
    let mut line = Vec::new();
    let mut arms: std::collections::BTreeMap<(String, String), RewardSummary> =
        std::collections::BTreeMap::new();
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        if !line.ends_with(b"\n") {
            break;
        }
        let Ok(record) = serde_json::from_slice::<RoutingRecord>(&line) else {
            continue;
        };
        // Only completed answer calls with a reward and a bucket are arm outcomes.
        let (Some(reward), Some(bucket)) = (record.reward, record.token_bucket.as_deref()) else {
            continue;
        };
        if record.model.is_empty() || record.tier == CLASSIFIER_TIER {
            continue;
        }
        let arm = arms
            .entry((record.model.into_owned(), bucket.to_string()))
            .or_insert(RewardSummary {
                count: 0,
                sum_reward: 0.0,
            });
        arm.count = arm.count.saturating_add(1);
        arm.sum_reward += reward;
    }
    Ok(arms)
}

/// Request fields retained until terminal usage and routing are available.
#[derive(Clone)]
pub(crate) struct RoutingLogContext {
    task: Option<String>,
    trial_id: Option<String>,
    session_id: Option<String>,
    request_id: Option<String>,
    token_bucket: Option<String>,
}

impl RoutingLogContext {
    /// Captures the normalized session ID, with the legacy log-only header as a fallback.
    pub(crate) fn from_metadata(metadata: &Metadata) -> Self {
        let headers = metadata.http_headers.as_ref();
        Self {
            task: headers
                .and_then(|headers| nonempty_header(headers, TASK_HEADER))
                .map(str::to_string),
            trial_id: headers
                .and_then(|headers| nonempty_header(headers, TRIAL_ID_HEADER))
                .map(str::to_string),
            session_id: metadata.session_id.clone().or_else(|| {
                headers
                    .and_then(|headers| nonempty_header(headers, LEGACY_SESSION_ID_HEADER))
                    .map(str::to_string)
            }),
            request_id: metadata.correlation_id.clone(),
            token_bucket: None,
        }
    }

    /// Records the bandit's token bucket, labelled from the request's estimated size.
    pub(crate) fn with_estimated_tokens(mut self, estimated_tokens: u64) -> Self {
        self.token_bucket = Some(token_bucket(estimated_tokens).to_string());
        self
    }
}

/// One appended routing record, and the read schema [`snapshot`] parses back,
/// so the written and expected shapes cannot drift apart. Missing fields
/// default so a record from an older schema still contributes what it has.
#[derive(Default, Deserialize, Serialize)]
#[serde(default)]
struct RoutingRecord<'a> {
    ts: Cow<'a, str>,
    #[serde(borrow)]
    task: Option<Cow<'a, str>>,
    #[serde(borrow)]
    trial_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    session_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    request_id: Option<Cow<'a, str>>,
    model: Cow<'a, str>,
    tier: Cow<'a, str>,
    prompt_tokens: u64,
    cached_tokens: u64,
    cache_creation_tokens: u64,
    completion_tokens: u64,
    reasoning_tokens: u64,
    total_tokens: u64,
    /// Estimated USD spent, when the target has a cost and usage was reported.
    cost_usd: Option<f64>,
    /// Wall-clock latency of the call, when measured.
    latency_ms: Option<f64>,
    /// Terminal success; `None` on legacy records written before this field existed.
    success: Option<bool>,
    /// Normalized 0..1 reward for answer calls; `None` for classifier calls and legacy records.
    reward: Option<f64>,
    /// Bandit token bucket, when the request size was estimated.
    #[serde(borrow)]
    token_bucket: Option<Cow<'a, str>>,
    /// The judge's solve-probability estimate, when a capability judge ran.
    judge_p_solve: Option<f64>,
    /// The judge's capability boundary, when a capability judge ran.
    #[serde(borrow)]
    judge_capability_boundary: Option<Cow<'a, str>>,
    /// The judge's minimum required capability level, when a capability judge ran.
    judge_minimum_capability: Option<f64>,
}

/// Session totals returned by the routing stats endpoint.
#[derive(Serialize)]
pub(crate) struct SessionStatsSnapshot {
    session_id: String,
    total_calls: u64,
    total_prompt_tokens: u64,
    total_cached_tokens: u64,
    total_cache_creation_tokens: u64,
    total_completion_tokens: u64,
    models: BTreeMap<ModelId, SessionModelStats>,
}

#[derive(Default, Serialize)]
struct SessionModelStats {
    calls: u64,
    prompt_tokens: u64,
    cached_tokens: u64,
    cache_creation_tokens: u64,
    completion_tokens: u64,
}

impl SessionStatsSnapshot {
    fn new(session_id: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            total_calls: 0,
            total_prompt_tokens: 0,
            total_cached_tokens: 0,
            total_cache_creation_tokens: 0,
            total_completion_tokens: 0,
            models: BTreeMap::new(),
        }
    }

    fn add_record(&mut self, record: &RoutingRecord<'_>, session_id: &str) {
        if record.session_id.as_deref() != Some(session_id) {
            return;
        }
        // Failed calls are bandit outcomes, not served usage: they carry no tokens, so
        // leave them out of the per-session accounting. Legacy records have no success
        // field and were all served calls, so they still count.
        if record.success == Some(false) {
            return;
        }
        let model = match record.model.as_ref() {
            "" => "unknown",
            model => model,
        };
        let stats = self.models.entry(ModelId::from(model)).or_default();
        stats.calls = stats.calls.saturating_add(1);
        stats.prompt_tokens = stats.prompt_tokens.saturating_add(record.prompt_tokens);
        stats.cached_tokens = stats.cached_tokens.saturating_add(record.cached_tokens);
        stats.cache_creation_tokens = stats
            .cache_creation_tokens
            .saturating_add(record.cache_creation_tokens);
        stats.completion_tokens = stats
            .completion_tokens
            .saturating_add(record.completion_tokens);
    }

    /// Session totals are exactly the sum of the per-model stats, so they are
    /// derived once rather than accumulated alongside them.
    fn sum_totals(&mut self) {
        for stats in self.models.values() {
            self.total_calls = self.total_calls.saturating_add(stats.calls);
            self.total_prompt_tokens = self.total_prompt_tokens.saturating_add(stats.prompt_tokens);
            self.total_cached_tokens = self.total_cached_tokens.saturating_add(stats.cached_tokens);
            self.total_cache_creation_tokens = self
                .total_cache_creation_tokens
                .saturating_add(stats.cache_creation_tokens);
            self.total_completion_tokens = self
                .total_completion_tokens
                .saturating_add(stats.completion_tokens);
        }
    }
}

fn nonempty_header<'a>(headers: &'a http::HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .filter(|value| !value.is_empty())
        .and_then(|v| v.to_str().ok())
}

fn routing_log_error(path: &Path, error: std::io::Error) -> ServerError {
    ServerError::new(format!(
        "failed to initialize routing log {}: {error}",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the requested session is counted, absent fields fall back to zero
    /// and `unknown`, and an unparseable line does not abort the scan.
    #[test]
    fn snapshot_counts_only_the_requested_session() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("routing.jsonl");
        fs::write(
            &path,
            concat!(
                r#"{"session_id":"a","model":"m1","fallback_reason":"unavailable","prompt_tokens":10,"completion_tokens":2}"#,
                "\n",
                r#"{"session_id":"b","model":"m1","prompt_tokens":99,"completion_tokens":99}"#,
                "\n",
                "not json\n",
                r#"{"session_id":"a","prompt_tokens":5}"#,
                "\n",
            ),
        )
        .expect("write log");

        let stats = snapshot(&path, "a").expect("read log").expect("session a");
        assert_eq!(stats.total_calls, 2);
        assert_eq!(stats.total_prompt_tokens, 15);
        assert_eq!(stats.total_completion_tokens, 2);
        assert_eq!(stats.models["m1"].calls, 1);
        assert_eq!(stats.models["unknown"].prompt_tokens, 5);
        assert!(snapshot(&path, "missing").expect("read log").is_none());
    }

    #[test]
    fn reward_blends_cost_latency_and_success() {
        // A cheap, fast, successful call scores near the top.
        let best = compute_reward(Some(0.0), 0.0, true);
        assert!((best - 1.0).abs() < f64::EPSILON);
        // A failure scores zero regardless of how cheap or fast it was.
        assert_eq!(compute_reward(Some(0.0), 0.0, false), 0.0);
        assert_eq!(compute_reward(Some(0.05), 5_000.0, false), 0.0);
        // Cost and latency are each capped at their scale and pull the blend down.
        let pricey_slow = compute_reward(Some(0.1), 20_000.0, true);
        assert!((pricey_slow - 0.4).abs() < 1e-9);
        // An unknown cost (no pricing configured) reads as free, not as expensive.
        let free = compute_reward(None, 0.0, true);
        assert!((free - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn token_buckets_split_at_their_boundaries() {
        assert_eq!(token_bucket(0), "small");
        assert_eq!(token_bucket(999), "small");
        assert_eq!(token_bucket(1_000), "medium");
        assert_eq!(token_bucket(10_000), "medium");
        assert_eq!(token_bucket(10_001), "large");
    }

    #[test]
    fn records_round_trip_reward_fields_and_legacy_records_default() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("routing.jsonl");
        fs::write(
            &path,
            concat!(
                // A full answer-call record with the reward fields.
                r#"{"model":"nano","prompt_tokens":5,"completion_tokens":2,"cost_usd":0.001,"latency_ms":800.0,"success":true,"reward":0.77,"token_bucket":"small"}"#,
                "\n",
                // A failed answer call.
                r#"{"model":"opus","success":false,"reward":0.0,"token_bucket":"large"}"#,
                "\n",
                // A classifier call: not an arm, no reward.
                r#"{"model":"judge","tier":"classifier","prompt_tokens":3,"completion_tokens":1}"#,
                "\n",
                // A legacy record predating the reward fields.
                r#"{"session_id":"a","model":"m1","prompt_tokens":10,"completion_tokens":2}"#,
                "\n",
            ),
        )
        .expect("write log");

        let arms = reward_summary(&path).expect("read rewards");
        // nano: one small-bucket answer with reward 0.77. The legacy record has no reward,
        // the classifier call carries no reward, so neither becomes an arm.
        assert_eq!(
            arms.get(&("nano".to_string(), "small".to_string())),
            Some(&RewardSummary {
                count: 1,
                sum_reward: 0.77,
            })
        );
        assert_eq!(
            arms.get(&("opus".to_string(), "large".to_string())),
            Some(&RewardSummary {
                count: 1,
                sum_reward: 0.0,
            })
        );
        assert!(!arms.contains_key(&("judge".to_string(), "small".to_string())));
        assert!(!arms.contains_key(&("m1".to_string(), "small".to_string())));
        assert_eq!(arms.len(), 2);
    }
}
