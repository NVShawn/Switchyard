// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Judge-backed capability, escalation, and custom-policy routing.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use switchyard_protocol::{ContentBlock, Message, ModelId, Role};

use super::fall_through::{DefaultTarget, FallThrough};
use super::util::DEFAULT_JUDGE_MAX_OUTPUT_TOKENS;
use super::util::affinity::AffinityRouter;
use super::util::classifier_contract::{
    ClassifierContract, ClassifierContractConfig, ClassifierResponseFormat,
};
use super::util::escalation::{self, EscalationJudge, EscalationJudgeConfig, EscalationPolicy};
use super::util::llm_judge::{
    ClassifierInput, JsonSchemaDecoder, JudgeClassifier, JudgePolicy, JudgeRuntimeConfig,
    SerdeDecoder, StructuredJudge,
};
use super::util::target_selector::TargetSelectorPolicy;
use super::util::thompson::{ThompsonSampler, estimate_request_tokens, token_bucket};
use crate::core::algorithm::{self, Algorithm, Driver};
use crate::core::classifier::{Classification, Classifier, Score};
use crate::core::state::{State, StateValue};
use crate::{LibsyError, Result};
use switchyard_protocol::{
    AggLlmResponse, InstructionBlock, LlmClientError, LlmRequest, LlmResponse, OutputParams,
    Request, Response,
};

const PROMPT_TEMPLATE: &str = include_str!("../prompts/capability-classifier/prompt.md");
const SCHEMA_TEMPLATE: &str = include_str!("../prompts/capability-classifier/schema.json");
/// Output judge prompt and verdict contract for Zone B fan-out comparison.
const COMPARE_PROMPT_TEMPLATE: &str = include_str!("../prompts/output-judge/prompt.md");
const COMPARE_SCHEMA_TEMPLATE: &str = include_str!("../prompts/output-judge/schema.json");
/// Telemetry label for this algorithm's spans, metrics, and logs.
const ALGORITHM_NAME: &str = "llm_task_classifier";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskClassifierVerdict {
    crux: String,
    primary_rule: String,
    capability_boundary: String,
    p_solve: f64,
    /// Lowest capability level the judge believes will solve the task, on the
    /// same 0..1 scale as each target's declared `capability`. Optional so a
    /// judge serving a binary route can omit it; ranked routes require it.
    #[serde(default)]
    minimum_capability: Option<f64>,
}

impl TaskClassifierVerdict {
    /// Rejects malformed or internally inconsistent verdicts before policy evaluation.
    fn is_valid(&self) -> bool {
        (0.0..=1.0).contains(&self.p_solve)
            && self
                .minimum_capability
                .is_none_or(|level| (0.0..=1.0).contains(&level))
            && !self.crux.trim().is_empty()
            && matches!(
                (
                    self.primary_rule.as_str(),
                    self.capability_boundary.as_str()
                ),
                ("SUP-1" | "SUP-2" | "SUP-3" | "SUP-4" | "SUP-5", "supported")
                    | ("UNC-1" | "UNC-2", "uncertain")
                    | ("LIM-1" | "LIM-2", "unsupported")
                    | ("none", "unmatched")
            )
    }

    /// Returns the number of threshold steps assigned to this capability boundary.
    fn boundary_steps(&self) -> Option<u8> {
        match self.capability_boundary.as_str() {
            "supported" => Some(0),
            "uncertain" | "unmatched" => Some(1),
            "unsupported" => Some(2),
            _ => None,
        }
    }
}

/// Keeps the opening task and the last `recent_turn_window` turns after it. A
/// window of `0` keeps the task alone.
///
/// Inbound decoders normalize client system and developer content into
/// `LlmRequest::instructions`, so it never reaches this list.
///
/// Selects by reference and clones only what survives — a coding-agent
/// conversation carries every tool result, so cloning it whole to keep a window
/// would copy the transcript on each judged turn.
fn trim_messages(messages: &[Message], recent_turn_window: usize) -> Vec<Message> {
    let is_instruction = |message: &Message| matches!(message.role, Role::System | Role::Developer);
    let mut kept: Vec<&Message> = messages.iter().filter(|m| is_instruction(m)).collect();
    let Some(task) = messages.iter().position(|m| m.role == Role::User) else {
        return kept.into_iter().cloned().collect();
    };
    kept.push(&messages[task]);

    let tail: Vec<&Message> = messages[task + 1..]
        .iter()
        .filter(|m| !is_instruction(m))
        .collect();
    kept.extend(&tail[window_start(&tail, recent_turn_window)..]);
    kept.into_iter().cloned().collect()
}

/// The first index of the trailing window.
///
/// Counting messages alone can start the window between an assistant tool call and the
/// result answering it, leaving the judge a result whose call id was never introduced. The
/// start therefore moves back to the nearest one that keeps every tool pair whole.
///
/// One newest-to-oldest pass carries the ids still waiting for a call. Direction is what
/// makes it correct: ids repeat across a conversation, and in this order a call is only
/// ever seen after the results it could answer, so a later call — already passed — clears
/// nothing. A result whose call sits before the opening task, which trimming never reaches,
/// keeps the set non-empty to the end and falls back to the counted start, so an unpairable
/// result costs one pass and cannot widen the window to the whole conversation.
fn window_start(tail: &[&Message], recent_turn_window: usize) -> usize {
    let counted = tail.len().saturating_sub(recent_turn_window);
    // An empty window holds no result to pair, and the loop below never visits its start.
    if counted == tail.len() {
        return counted;
    }
    let mut unpaired: HashSet<&str> = HashSet::new();
    for (start, message) in tail.iter().enumerate().rev() {
        // Blocks reverse too, so a call answers a result only when it precedes it inside
        // one message as well as across messages.
        for block in message.content.iter().rev() {
            match block {
                ContentBlock::ToolResult(result) => {
                    unpaired.insert(result.tool_call_id.as_str());
                }
                ContentBlock::ToolCall(call) => {
                    unpaired.remove(call.id.as_str());
                }
                _ => {}
            }
        }
        if start <= counted && unpaired.is_empty() {
            return start;
        }
    }
    counted
}

/// Keeps the opening task and the latest user follow-up when they differ.
///
/// This path carries only task text, so tool blocks are stripped: keeping just
/// two user turns cannot preserve tool pairs, and an orphaned call or result
/// leads upstreams such as Bedrock to reject the judge request. A message left
/// with no content after stripping is dropped.
fn task_messages(messages: &[Message]) -> Vec<Message> {
    let strip_tools = |message: &Message| -> Option<Message> {
        let content: Vec<ContentBlock> = message
            .content
            .iter()
            .filter(|block| {
                !matches!(
                    block,
                    ContentBlock::ToolCall(_) | ContentBlock::ToolResult(_)
                )
            })
            .cloned()
            .collect();
        if content.is_empty() {
            return None;
        }
        Some(Message {
            role: message.role,
            content,
        })
    };
    let mut user_messages = messages.iter().filter(|message| message.role == Role::User);
    let Some(opening_task) = user_messages.next() else {
        return Vec::new();
    };
    let latest_follow_up = user_messages.next_back();
    [Some(opening_task), latest_follow_up]
        .into_iter()
        .flatten()
        .filter_map(strip_tools)
        .collect()
}

/// Selects the task messages shown to capability and custom-schema classifiers.
struct TaskInput {
    recent_turn_window: Option<usize>,
}

impl ClassifierInput for TaskInput {
    fn build_messages(&self, _state: &State, request: &Request) -> Vec<Message> {
        // The default preserves the whole-task anchor and latest user update. A
        // configured window widens that to the surrounding conversation.
        match self.recent_turn_window {
            Some(window) => trim_messages(&request.llm_request.messages, window),
            None => task_messages(&request.llm_request.messages),
        }
    }
}

type CapabilityJudge = StructuredJudge<TaskInput, SerdeDecoder<TaskClassifierVerdict>>;

/// One rung of a cost-aware capability ladder: a routing target with its declared
/// static capability level and inherited unit cost.
///
/// `capability` is on the same 0..1 scale as the judge's `minimum_capability`
/// verdict; `cost` orders adequate targets cheapest-first. Constructed by the host
/// from deployment config and handed to the capability route.
#[derive(Clone, Debug)]
pub struct CapabilityTarget {
    /// The routing target this rung selects.
    pub target: ModelId,
    /// Declared capability level in `[0.0, 1.0]`.
    pub capability: f64,
    /// Unit pricing inherited from the target's deployment config.
    pub cost: switchyard_protocol::TargetCost,
    /// The rung's context window in tokens; `None` means unknown (never prefiltered).
    pub context_window: Option<u32>,
}

/// Context headroom factor: a rung is only eligible if its window fits the request
/// with 15% to spare, so the estimate's slack does not push a call over the limit.
const CONTEXT_HEADROOM_PERCENT: u64 = 115;

impl CapabilityTarget {
    /// Validates the declared capability level.
    fn is_valid(&self) -> bool {
        self.capability.is_finite() && (0.0..=1.0).contains(&self.capability)
    }

    /// Scalar used to order rungs cheapest-first. Input and output unit prices are
    /// summed because the request's input/output split is not known at select time.
    fn cost_key(&self) -> f64 {
        self.cost.input_per_1m + self.cost.output_per_1m
    }

    /// Whether this rung's context window fits the request. Unknown on either side
    /// means do not prefilter — an unknown window is never a reason to skip a rung.
    fn fits_context(&self, needed_tokens: Option<u64>) -> bool {
        match (self.context_window, needed_tokens) {
            (Some(window), Some(needed)) => u64::from(window) >= needed,
            _ => true,
        }
    }
}

struct TaskClassifierPolicy {
    efficient_target: ModelId,
    capable_target: ModelId,
    base_threshold: f64,
    threshold_step: f64,
    /// Cost-ascending capability ladder; empty selects the binary efficient/capable policy.
    ranked: Vec<CapabilityTarget>,
}

impl TaskClassifierPolicy {
    fn new(
        efficient_target: impl Into<ModelId>,
        capable_target: impl Into<ModelId>,
        config: &TaskClassifierConfig,
    ) -> Self {
        Self {
            efficient_target: efficient_target.into(),
            capable_target: capable_target.into(),
            base_threshold: config.base_threshold,
            threshold_step: config.threshold_step,
            ranked: Vec::new(),
        }
    }

    /// Cost-aware form: routes among `ranked` instead of the binary tier pair.
    fn with_ranked_targets(mut self, ranked: Vec<CapabilityTarget>) -> Self {
        self.ranked = ranked;
        self
    }

    /// Returns the required solve probability for one validated verdict.
    fn threshold(&self, verdict: &TaskClassifierVerdict) -> Option<f64> {
        Some(self.base_threshold + f64::from(verdict.boundary_steps()?) * self.threshold_step)
    }

    /// The most capable rung, used when the judge judges the task beyond the cheap
    /// tiers or names no usable level.
    fn most_capable(&self) -> &ModelId {
        self.ranked
            .iter()
            .max_by(|a, b| {
                a.capability
                    .partial_cmp(&b.capability)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|rung| &rung.target)
            .unwrap_or(&self.capable_target)
    }

    /// Picks the cheapest rung whose declared capability clears the judge's required
    /// level and whose context window fits the request. The ladder is cost-ascending,
    /// so the first adequate rung is the pick. `needed_tokens` is the request's
    /// estimated size including headroom; `None` skips the context prefilter.
    fn ranked_pick(&self, verdict: &TaskClassifierVerdict, needed_tokens: Option<u64>) -> ModelId {
        // An unsupported boundary means the cheap tiers are judged out; go straight to
        // the strongest rung rather than trusting a level the judge already flagged.
        if verdict.capability_boundary == "unsupported" {
            return self.most_capable().clone();
        }
        // The ranked pick needs the judge's required level; without one it cannot
        // tell the rungs apart, so fall back to the strongest rung.
        let Some(required) = verdict.minimum_capability else {
            return self.most_capable().clone();
        };
        self.ranked
            .iter()
            .find(|rung| rung.capability >= required && rung.fits_context(needed_tokens))
            .map(|rung| rung.target.clone())
            .unwrap_or_else(|| self.most_capable().clone())
    }

    /// The full pick as a classification, with an optional context size for the prefilter.
    fn classify(
        &self,
        verdict: &TaskClassifierVerdict,
        needed_tokens: Option<u64>,
    ) -> Classification {
        if self.ranked.is_empty() {
            // Binary mode has no ladder; the context prefilter does not apply.
            let Some(threshold) = self.threshold(verdict) else {
                return Classification::Ambiguous(vec![]);
            };
            let target = if verdict.p_solve >= threshold
                || (threshold - verdict.p_solve).abs() <= f64::EPSILON
            {
                &self.efficient_target
            } else {
                &self.capable_target
            };
            return Classification::Scores(vec![Score {
                target: target.clone(),
                confidence: 1.0,
            }]);
        }
        Classification::Scores(vec![Score {
            target: self.ranked_pick(verdict, needed_tokens),
            confidence: 1.0,
        }])
    }
}

impl JudgePolicy for TaskClassifierPolicy {
    type Verdict = TaskClassifierVerdict;

    fn to_classification(&self, verdict: Option<&Self::Verdict>) -> Classification {
        // Judge output is untrusted. An absent, invalid, or inconsistent verdict is
        // ambiguous so the surrounding router applies its configured fallback.
        let Some(verdict) = verdict.filter(|verdict| verdict.is_valid()) else {
            return Classification::Ambiguous(vec![]);
        };
        // No request context here, so the context prefilter does not apply; the
        // request-aware classifier calls `classify` with the estimate instead.
        self.classify(verdict, None)
    }
}

#[derive(Clone, Debug)]
/// Settings that control capability classifier prompting and routing.
pub struct TaskClassifierConfig {
    /// Lowest solve probability that routes a supported task to the efficient target.
    pub base_threshold: f64,
    /// Amount added per capability-boundary step.
    ///
    /// Supported verdicts use `base_threshold`, uncertain and unmatched verdicts use one
    /// step, and unsupported verdicts use two steps.
    pub threshold_step: f64,
    /// Enables session affinity before the judge-backed classifier.
    pub session_affinity: bool,
    /// Uses the first user message as the SessionKey for sticky routing when session metadata is unavailable.
    pub message_hash_fallback: bool,
    /// Trailing conversation turns the judge sees on top of the client
    /// instructions and the opening task.
    ///
    /// `None` (the default) judges the opening task and latest user follow-up.
    /// `Some(n)` widens that to the client instructions, the opening task, and
    /// the last `n` turns after it.
    pub recent_turn_window: Option<usize>,
    /// Prompt and verdict contract settings for the classifier judge.
    pub contract: ClassifierContractConfig,
    /// Maximum completion tokens available to the classifier verdict.
    pub max_output_tokens: u64,
}

/// Flat serialized shape that maps prompt settings into the runtime contract.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskClassifierConfigWire {
    base_threshold: f64,
    #[serde(default)]
    threshold_step: f64,
    #[serde(default)]
    session_affinity: bool,
    #[serde(default)]
    message_hash_fallback: bool,
    #[serde(default)]
    recent_turn_window: Option<usize>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    response_format_type: ClassifierResponseFormat,
    #[serde(default = "default_judge_max_output_tokens")]
    max_output_tokens: u64,
}

impl<'de> Deserialize<'de> for TaskClassifierConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = TaskClassifierConfigWire::deserialize(deserializer)?;
        let mut contract = ClassifierContractConfig::default();
        if let Some(prompt) = wire.prompt {
            contract = contract.with_prompt(prompt);
        }
        contract = contract.with_response_format_type(wire.response_format_type);
        Ok(Self {
            base_threshold: wire.base_threshold,
            threshold_step: wire.threshold_step,
            session_affinity: wire.session_affinity,
            message_hash_fallback: wire.message_hash_fallback,
            recent_turn_window: wire.recent_turn_window,
            contract,
            max_output_tokens: wire.max_output_tokens,
        })
    }
}

const fn default_judge_max_output_tokens() -> u64 {
    DEFAULT_JUDGE_MAX_OUTPUT_TOKENS
}

impl Default for TaskClassifierConfig {
    fn default() -> Self {
        Self {
            base_threshold: 0.0,
            threshold_step: 0.0,
            session_affinity: false,
            message_hash_fallback: false,
            recent_turn_window: None,
            contract: ClassifierContractConfig::default(),
            max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
        }
    }
}

impl TaskClassifierConfig {
    /// Validates routing thresholds before the classifier is constructed.
    fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.base_threshold) {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "base_threshold must be between 0 and 1, got {}",
                    self.base_threshold
                ),
            });
        }
        if !self.threshold_step.is_finite() || self.threshold_step < 0.0 {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "threshold_step must be finite and greater than or equal to 0, got {}",
                    self.threshold_step
                ),
            });
        }
        let unsupported_threshold = self.base_threshold + 2.0 * self.threshold_step;
        if unsupported_threshold > 1.0 && unsupported_threshold - 1.0 > f64::EPSILON {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "base_threshold + 2 * threshold_step must be at most 1, got {unsupported_threshold}"
                ),
            });
        }
        if self.max_output_tokens == 0 {
            return Err(LibsyError::AlgorithmError {
                message: "max_output_tokens must be at least 1".to_string(),
            });
        }
        if self.message_hash_fallback && !self.session_affinity {
            return Err(LibsyError::AlgorithmError {
                message: "message_hash_fallback requires session_affinity".to_string(),
            });
        }
        Ok(())
    }
}

/// Policy that maps a custom classifier verdict to a routing target.
#[derive(Clone, Debug)]
pub enum CustomClassifierPolicy {
    /// Resolves a JSON Pointer and treats its string value as a configured target label.
    TargetSelector {
        /// JSON Pointer evaluated against each schema-validated verdict.
        selector: String,
    },
}

impl CustomClassifierPolicy {
    /// Creates a policy that selects a target label through a JSON Pointer.
    pub fn target_selector(selector: impl Into<String>) -> Self {
        Self::TargetSelector {
            selector: selector.into(),
        }
    }
}

/// Settings for a classifier whose JSON Schema and target-selection policy are user supplied.
#[derive(Clone, Debug)]
pub struct CustomClassifierConfig {
    /// System prompt sent to the classifier judge.
    pub prompt: String,
    /// Inner JSON Schema placed inside the provider's structured-output wrapper.
    pub response_schema: Value,
    /// Deterministic policy applied after the verdict passes schema validation.
    pub policy: CustomClassifierPolicy,
    /// Enables session affinity before the judge-backed classifier.
    pub session_affinity: bool,
    /// Uses the first user message when session metadata is unavailable.
    pub message_hash_fallback: bool,
    /// Trailing conversation turns shown to the classifier judge.
    pub recent_turn_window: Option<usize>,
    /// Maximum completion tokens available to the classifier verdict.
    pub max_output_tokens: u64,
}

impl CustomClassifierConfig {
    /// Creates a custom-schema classifier contract with conservative runtime defaults.
    pub fn new(
        prompt: impl Into<String>,
        response_schema: Value,
        policy: CustomClassifierPolicy,
    ) -> Self {
        Self {
            prompt: prompt.into(),
            response_schema,
            policy,
            session_affinity: false,
            message_hash_fallback: false,
            recent_turn_window: None,
            max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.max_output_tokens == 0 {
            return Err(LibsyError::AlgorithmError {
                message: "max_output_tokens must be at least 1".to_string(),
            });
        }
        if self.message_hash_fallback && !self.session_affinity {
            return Err(LibsyError::AlgorithmError {
                message: "message_hash_fallback requires session_affinity".to_string(),
            });
        }
        Ok(())
    }
}

enum CustomPolicyRuntime {
    TargetSelector(TargetSelectorPolicy),
}

impl JudgePolicy for CustomPolicyRuntime {
    type Verdict = Value;

    fn to_classification(&self, verdict: Option<&Self::Verdict>) -> Classification {
        match self {
            Self::TargetSelector(policy) => policy.to_classification(verdict),
        }
    }
}

struct TaskClassifier {
    classifier: JudgeClassifier<CapabilityJudge, TaskClassifierPolicy>,
    efficient_target: ModelId,
    capable_target: ModelId,
}

// ── Cost-aware classifier (capability ladder + confidence zones) ─────────────

/// The output judge's pick among the fanned-out candidate answers.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompareVerdict {
    winner: usize,
    /// Stating the case sharpens the verdict; routing reads only the index.
    #[allow(dead_code)]
    reason: String,
}

/// Confidence-zone and fan-out settings for cost-aware routing.
///
/// Zones partition the judge's `p_solve`: high confidence answers with a single
/// cheapest-adequate call (Zone A), low confidence routes straight to the most
/// capable rung (Zone C), and the mid band fans out to several rungs and lets an
/// output judge pick the best answer (Zone B). Fan-out is the explicit price of
/// uncertainty: Zone B spends `fan_out` answer calls plus one judge call.
#[derive(Clone, Debug)]
pub struct ZoneConfig {
    /// `p_solve` at or above this answers with one cheapest-adequate call (Zone A).
    pub high_threshold: f64,
    /// `p_solve` below this routes straight to the most capable rung (Zone C).
    pub low_threshold: f64,
    /// How many eligible rungs Zone B calls concurrently before judging their answers.
    pub fan_out: usize,
    /// Target that judges the fanned-out candidate answers.
    pub output_judge_target: ModelId,
    /// Prompt and verdict contract settings for the output judge.
    pub output_judge_contract: ClassifierContractConfig,
    /// Maximum completion tokens available to the output judge verdict.
    pub output_judge_max_output_tokens: u64,
}

impl ZoneConfig {
    fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.low_threshold) || !(0.0..=1.0).contains(&self.high_threshold)
        {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "zone thresholds must be between 0 and 1, got low {} high {}",
                    self.low_threshold, self.high_threshold
                ),
            });
        }
        if self.low_threshold > self.high_threshold {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "zone low_threshold {} must not exceed high_threshold {}",
                    self.low_threshold, self.high_threshold
                ),
            });
        }
        if self.fan_out == 0 {
            return Err(LibsyError::AlgorithmError {
                message: "zone fan_out must be at least 1".to_string(),
            });
        }
        if self.output_judge_max_output_tokens == 0 {
            return Err(LibsyError::AlgorithmError {
                message: "output judge max_output_tokens must be at least 1".to_string(),
            });
        }
        Ok(())
    }
}

/// Zone boundaries with the output-judge contract compiled, ready to serve.
struct ZoneRuntime {
    low_threshold: f64,
    high_threshold: f64,
    fan_out: usize,
    output_judge_target: ModelId,
    output_contract: ClassifierContract,
    output_max_output_tokens: u64,
}

impl ZoneConfig {
    fn build_runtime(&self) -> Result<ZoneRuntime> {
        self.validate()?;
        let output_contract = ClassifierContract::from_config(
            &self.output_judge_contract,
            COMPARE_PROMPT_TEMPLATE,
            COMPARE_SCHEMA_TEMPLATE,
        )?;
        Ok(ZoneRuntime {
            low_threshold: self.low_threshold,
            high_threshold: self.high_threshold,
            fan_out: self.fan_out,
            output_judge_target: self.output_judge_target.clone(),
            output_contract,
            output_max_output_tokens: self.output_judge_max_output_tokens,
        })
    }
}

/// Which zone a validated verdict falls into.
enum Zone {
    /// High confidence: one cheapest-adequate call.
    Answer,
    /// Low confidence or an unsupported boundary: the most capable rung.
    Capable,
    /// Mid confidence: fan out and judge the candidates' answers.
    FanOut,
}

/// Optional Thompson-sampling correction to the judge's confidence, learned from
/// observed per-arm rewards. The sampler is shared with the host's refresh loop.
///
/// The bandit never replaces the judge: it only shifts `p_solve` before zone
/// classification, so removing it is a config flip, not a refactor.
pub struct BanditConfig {
    /// Shared bandit state over `(target, token bucket)` arms.
    pub sampler: Arc<ThompsonSampler>,
    /// How far one sample can move `p_solve`: `corrected = p_solve + (sample - 0.5) * scale`.
    pub scale: f64,
}

/// Routes among a cost-ascending capability ladder, optionally fanning out over the
/// mid-confidence zone and judging the candidates' answers with an output judge.
struct CostAwareClassifier {
    classifier: JudgeClassifier<CapabilityJudge, TaskClassifierPolicy>,
    /// Cost-ascending capability ladder.
    ranked: Vec<CapabilityTarget>,
    /// The most capable rung: Zone C's target and the cascade's fallback.
    capable_target: ModelId,
    zones: Option<ZoneRuntime>,
    /// Optional bandit correction to the judge's confidence.
    bandit: Option<BanditConfig>,
}

#[async_trait]
impl Classifier<State> for CostAwareClassifier {
    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        let Some(driver) = driver else {
            return Err(LibsyError::AlgorithmError {
                message: "cost-aware classifier requires a driver".to_string(),
            });
        };
        let verdict = self.classifier.verdict(state, request, driver).await;
        let Some(verdict) = verdict.filter(|verdict| verdict.is_valid()) else {
            return Ok((Classification::Ambiguous(vec![]), None));
        };
        // Estimated request size with headroom, for the context prefilter.
        let needed = Some(Self::needed_tokens(request));
        let Some(zones) = &self.zones else {
            // No zones: the ranked pick is the whole decision, a single call.
            return Ok((self.classifier.policy().classify(&verdict, needed), None));
        };
        // The bandit nudges the judge's confidence from the cheapest adequate rung's
        // observed rewards before the zone decision — a failing cheap tier drops toward
        // fan-out or the capable rung.
        let p_solve = self.corrected_p_solve(&verdict, request);
        match Self::zone(p_solve, &verdict, zones) {
            Zone::Answer => Ok((self.classifier.policy().classify(&verdict, needed), None)),
            Zone::Capable => Ok((decisive(&self.capable_target), None)),
            Zone::FanOut => self.fan_out(request, driver, &verdict, zones).await,
        }
    }
}

impl CostAwareClassifier {
    /// The request's estimated token size with context headroom (the 15% rule).
    fn needed_tokens(request: &Request) -> u64 {
        estimate_request_tokens(request) * CONTEXT_HEADROOM_PERCENT / 100
    }

    /// Maps a (possibly bandit-corrected) confidence to its zone. An unsupported boundary
    /// is always Zone C: the judge already distrusts the cheap tiers, whatever p_solve says.
    fn zone(p_solve: f64, verdict: &TaskClassifierVerdict, zones: &ZoneRuntime) -> Zone {
        if verdict.capability_boundary == "unsupported" {
            return Zone::Capable;
        }
        if p_solve >= zones.high_threshold {
            Zone::Answer
        } else if p_solve < zones.low_threshold {
            Zone::Capable
        } else {
            Zone::FanOut
        }
    }

    /// The cheapest rung clearing the judge's level — the Zone A candidate whose arm
    /// corrects the confidence. Without a level, the cheapest rung overall.
    fn cheapest_eligible(&self, verdict: &TaskClassifierVerdict) -> Option<&ModelId> {
        match verdict.minimum_capability {
            Some(required) => self
                .ranked
                .iter()
                .find(|rung| rung.capability >= required)
                .map(|rung| &rung.target),
            None => self.ranked.first().map(|rung| &rung.target),
        }
    }

    /// The judge's `p_solve`, nudged by the bandit arm of the cheapest adequate rung.
    fn corrected_p_solve(&self, verdict: &TaskClassifierVerdict, request: &Request) -> f64 {
        let Some(bandit) = &self.bandit else {
            return verdict.p_solve;
        };
        let Some(cheapest) = self.cheapest_eligible(verdict) else {
            return verdict.p_solve;
        };
        let sample = bandit
            .sampler
            .sample(cheapest, token_bucket(estimate_request_tokens(request)));
        (verdict.p_solve + (sample - 0.5) * bandit.scale).clamp(0.0, 1.0)
    }

    /// Zone B: call the cheapest eligible rungs concurrently, buffer their answers, and
    /// return the output judge's pick. The losers' cost is the price of uncertainty.
    async fn fan_out(
        &self,
        request: &mut Request,
        driver: &Driver,
        verdict: &TaskClassifierVerdict,
        zones: &ZoneRuntime,
    ) -> Result<(Classification, Option<Response>)> {
        // Eligible rungs clear the judged level and fit the request's context, cheapest
        // first; without a level the whole ladder is eligible. The ladder is cost-ordered.
        let needed = Some(Self::needed_tokens(request));
        let eligible: Vec<&CapabilityTarget> = self
            .ranked
            .iter()
            .filter(|rung| {
                verdict
                    .minimum_capability
                    .is_none_or(|required| rung.capability >= required)
                    && rung.fits_context(needed)
            })
            .collect();
        let candidates: Vec<&CapabilityTarget> = eligible.into_iter().take(zones.fan_out).collect();
        match candidates.len() {
            // No rung clears the level: this is Zone C's job, not a fan-out.
            0 => return Ok((decisive(&self.capable_target), None)),
            // One rung needs no judge and no second call.
            1 => {
                return Ok((decisive(&candidates[0].target), None));
            }
            _ => {}
        }

        // Call the candidates concurrently; the driver serves them in parallel.
        let results = futures::future::join_all(
            candidates
                .iter()
                .map(|rung| driver.call_model(request.clone(), vec![rung.target.clone()], true)),
        )
        .await;

        // Buffer the answers that came back; a failed or unbufferable candidate drops out.
        let stream = request.llm_request.stream;
        let mut buffered: Vec<(
            ModelId,
            AggLlmResponse,
            Option<switchyard_protocol::Metadata>,
        )> = Vec::new();
        for (rung, result) in candidates.iter().zip(results) {
            let Ok(response) = result else { continue };
            let Response {
                llm_response,
                metadata,
            } = response;
            match llm_response.into_agg().await {
                Ok(aggregate) => buffered.push((rung.target.clone(), aggregate, metadata)),
                Err(_) => continue,
            }
        }
        match buffered.len() {
            // Every candidate failed: abstain so the cascade's fallback decides.
            0 => return Ok((Classification::Ambiguous(vec![]), None)),
            // A lone survivor needs no comparison.
            1 => {
                return match buffered.into_iter().next() {
                    Some((target, aggregate, metadata)) => Ok((
                        decisive(&target),
                        Some(into_response(aggregate, metadata, stream)),
                    )),
                    None => Ok((Classification::Ambiguous(vec![]), None)),
                };
            }
            _ => {}
        }

        // Ask the output judge which candidate answer is best. A judge failure or an
        // out-of-range index falls back to the cheapest candidate, already paid for.
        let candidate_texts: Vec<String> = buffered
            .iter()
            .map(|(_, aggregate, _)| switchyard_protocol::completion_text(aggregate))
            .collect();
        let compare_request = Self::compare_request(request, &candidate_texts, zones);
        let winner = match driver
            .call_model(
                compare_request,
                vec![zones.output_judge_target.clone()],
                false,
            )
            .await
        {
            Ok(response) => match response.llm_response.into_agg().await {
                Ok(aggregate) => {
                    super::util::llm_judge::parse_json_verdict::<CompareVerdict>(&aggregate)
                        .ok()
                        .filter(|verdict| verdict.winner < buffered.len())
                        .map(|verdict| verdict.winner)
                }
                Err(_) => None,
            },
            Err(_) => None,
        }
        .unwrap_or(0);

        let (target, aggregate, metadata) = buffered.swap_remove(winner);
        Ok((
            decisive(&target),
            Some(into_response(aggregate, metadata, stream)),
        ))
    }

    /// Builds the output-judge request: the trimmed task, then the numbered candidates.
    fn compare_request(request: &Request, candidates: &[String], zones: &ZoneRuntime) -> Request {
        let mut messages = task_messages(&request.llm_request.messages);
        let mut comparison = String::from("Candidate answers, numbered from 0:\n");
        for (index, candidate) in candidates.iter().enumerate() {
            comparison.push_str(&format!("\n--- candidate {index} ---\n{candidate}\n"));
        }
        messages.push(Message::text(Role::User, comparison));
        Request {
            llm_request: LlmRequest {
                model: request.llm_request.model.clone(),
                instructions: vec![InstructionBlock {
                    role: Role::System,
                    content: Message::text(
                        Role::System,
                        zones.output_contract.system_prompt().to_string(),
                    )
                    .content,
                }],
                messages,
                output: OutputParams {
                    max_output_tokens: Some(zones.output_max_output_tokens),
                    response_format: Some(zones.output_contract.response_format().clone()),
                },
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: request.metadata.clone(),
        }
    }
}

/// Rebuilds a buffered candidate answer into a response, re-streaming it when the
/// caller asked for a stream.
fn into_response(
    aggregate: AggLlmResponse,
    metadata: Option<switchyard_protocol::Metadata>,
    stream: bool,
) -> Response {
    Response {
        llm_response: if stream {
            LlmResponse::Stream(aggregate.into_stream())
        } else {
            LlmResponse::Agg(aggregate)
        },
        metadata,
    }
}


// ── Escalation classifier ──────────────────────────────────────────────────

/// Session-state key holding the consecutive-escalate streak.
const STREAK_KEY: &str = "escalation_streak";

fn streak(state: &State) -> u32 {
    match state.extra.get(STREAK_KEY) {
        Some(StateValue::Count(n)) => *n,
        _ => 0,
    }
}

fn decisive(target: &ModelId) -> Classification {
    Classification::Scores(vec![Score {
        target: target.clone(),
        confidence: 1.0,
    }])
}

fn assistant_message(response: &AggLlmResponse) -> Message {
    Message {
        role: Role::Assistant,
        content: response
            .first_output()
            .map(|output| output.content.clone())
            .unwrap_or_default(),
    }
}

/// Calls the efficient model, judges its response, and latches to capable once the streak
/// confirms. Returns the efficient response directly when not escalating so the caller does
/// not pay for a second model call.
struct EscalationClassifier {
    judge: JudgeClassifier<EscalationJudge, EscalationPolicy>,
    capable: ModelId,
    efficient: ModelId,
    /// Consecutive escalate verdicts required to latch.
    confirmations: u32,
}

#[async_trait]
impl Classifier<State> for EscalationClassifier {
    fn routing_tier(&self, selected_model_id: &ModelId) -> Option<&'static str> {
        if self.capable == self.efficient {
            None
        } else if *selected_model_id == self.capable {
            Some("strong")
        } else if *selected_model_id == self.efficient {
            Some("weak")
        } else {
            None
        }
    }

    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        let Some(driver) = driver else {
            return Err(LibsyError::AlgorithmError {
                message: "escalation classifier requires a driver".into(),
            });
        };

        // A confirmed session stays capable without a judge call.
        if streak(state) >= self.confirmations {
            return Ok((decisive(&self.capable), None));
        }

        // Call efficient model and buffer the response so the judge can read it.
        //
        // If the efficient model exceeds its context window, fall through to capable. This call
        // deliberately has one candidate so the classifier sees the efficient model's error.
        tracing::info!(
            target = %self.efficient,
            "escalation classifier selected efficient tier"
        );
        let efficient_response = match driver
            .call_model(request.clone(), vec![self.efficient.clone()], true)
            .await
        {
            Ok(r) => r,
            Err(LibsyError::ClientCall {
                source: LlmClientError::ContextWindowExceeded { .. },
                ..
            }) => return Ok((decisive(&self.capable), None)),
            Err(e) => return Err(e),
        };
        // The call resolves when its stream handle arrives; transport can still fail while
        // buffering. Fall back only for that availability failure and keep other errors typed.
        let agg = match efficient_response.llm_response.into_agg().await {
            Ok(agg) => agg,
            Err(LlmClientError::Transport { .. }) => {
                return Ok((decisive(&self.capable), None));
            }
            Err(source) => {
                return Err(LibsyError::client_call(self.efficient.clone(), source));
            }
        };
        // Append the efficient reply so the judge reads this turn's completed trajectory.
        let mut judge_request = request.clone();
        judge_request
            .llm_request
            .messages
            .push(assistant_message(&agg));
        let efficient_response = Response {
            llm_response: if request.llm_request.stream {
                LlmResponse::Stream(agg.into_stream())
            } else {
                LlmResponse::Agg(agg)
            },
            metadata: efficient_response.metadata,
        };

        let (classification, _) = self
            .judge
            .score(state, &mut judge_request, Some(driver))
            .await?;

        let held = streak(state);
        let best = classification.argmax(false)?;
        let (escalate, pending) = match &best {
            Some(score) if score.target == self.capable => (true, held + 1),
            Some(_) => (false, 0),
            None => (false, held),
        };
        state
            .extra
            .insert(STREAK_KEY.to_string(), StateValue::Count(pending));

        if escalate && pending >= self.confirmations {
            // Streak confirmed: drop the efficient response, caller will serve capable.
            return Ok((decisive(&self.capable), None));
        }

        Ok((decisive(&self.efficient), Some(efficient_response)))
    }
}

/// Routes requests through a capability, escalation, or custom classifier mode.
pub struct LlmTaskClassifier {
    route: FallThrough<State>,
    /// Classifier used when this router is embedded in another cascade.
    inner: Arc<dyn Classifier<State>>,
}

struct ClassifierRouteConfig {
    default_target: ModelId,
    session_affinity: bool,
    message_hash_fallback: bool,
}

/// Complete construction settings for one LLM classifier mode.
#[non_exhaustive]
pub enum LlmClassifierConfig {
    /// Routes between efficient and capable targets from a task-level verdict.
    Capability {
        /// Target that produces classifier verdicts.
        judge_target: ModelId,
        /// Target used when the efficient tier can handle the task.
        efficient_target: ModelId,
        /// Target used when the task needs the capable tier.
        capable_target: ModelId,
        /// Cost-ascending capability ladder for cost-aware multi-target routing.
        ///
        /// Empty selects the binary efficient/capable policy. Two or more rungs switch
        /// to the cost-aware pick: the cheapest rung whose declared capability clears
        /// the judge's `minimum_capability`.
        #[doc = ""]
        capability_targets: Vec<CapabilityTarget>,
        /// Confidence zones and fan-out. Only valid with a `capability_targets` ladder;
        /// `None` makes every ranked pick a single call.
        capability_zones: Option<ZoneConfig>,
        /// Optional Thompson-sampling correction to the judge's confidence, learned from
        /// observed rewards. Only applies when `capability_zones` is set.
        bandit: Option<BanditConfig>,
        /// Capability classifier settings.
        config: TaskClassifierConfig,
    },
    /// Judges efficient responses and escalates after a confirmed streak.
    Escalation {
        /// Target that produces escalation verdicts.
        judge_target: ModelId,
        /// Target called before each escalation decision.
        efficient_target: ModelId,
        /// Target used after escalation is confirmed.
        capable_target: ModelId,
        /// Prompt and verdict contract settings for the escalation judge.
        contract: ClassifierContractConfig,
        /// Escalation policy settings.
        config: EscalationJudgeConfig,
        /// Maximum completion tokens available to the escalation verdict.
        max_output_tokens: u64,
    },
    /// Routes among named targets using a user-supplied schema and policy.
    Custom {
        /// Target that produces classifier verdicts.
        judge_target: ModelId,
        /// User-facing labels paired with their resolved routing targets.
        targets: Vec<(String, ModelId)>,
        /// Label selected when the judge does not produce a usable verdict.
        default_target: String,
        /// Custom classifier settings.
        config: CustomClassifierConfig,
    },
}

impl LlmTaskClassifier {
    /// Builds the classifier mode described by `config`.
    ///
    /// # Errors
    ///
    /// Returns an error when the selected mode's targets, contract, policy, or runtime
    /// settings are invalid.
    pub fn new(config: LlmClassifierConfig) -> Result<Self> {
        match config {
            LlmClassifierConfig::Capability {
                judge_target,
                efficient_target,
                capable_target,
                capability_targets,
                capability_zones,
                bandit,
                config,
            } => Self::build_capability(
                judge_target,
                efficient_target,
                capable_target,
                capability_targets,
                capability_zones,
                bandit,
                config,
            ),
            LlmClassifierConfig::Escalation {
                judge_target,
                efficient_target,
                capable_target,
                contract,
                config,
                max_output_tokens,
            } => Self::build_escalation(
                judge_target,
                efficient_target,
                capable_target,
                contract,
                config,
                max_output_tokens,
            ),
            LlmClassifierConfig::Custom {
                judge_target,
                targets,
                default_target,
                config,
            } => Self::build_custom(judge_target, targets, default_target, config),
        }
    }

    fn build_capability(
        judge_target: ModelId,
        efficient_target: ModelId,
        capable_target: ModelId,
        capability_targets: Vec<CapabilityTarget>,
        capability_zones: Option<ZoneConfig>,
        bandit: Option<BanditConfig>,
        config: TaskClassifierConfig,
    ) -> Result<Self> {
        config.validate()?;
        // Zones need a ladder to fan out over; they are meaningless for the binary pair.
        // The bandit only corrects zone confidence, so it needs zones to act on.
        if capability_zones.is_some() && capability_targets.is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "capability_zones requires a capability_targets ladder".to_string(),
            });
        }
        if bandit.is_some() && capability_zones.is_none() {
            return Err(LibsyError::AlgorithmError {
                message: "bandit correction requires capability_zones".to_string(),
            });
        }
        // Validate and order the ladder cheapest-first. An empty ladder keeps the binary
        // efficient/capable policy; one rung is not a ladder, so reject it.
        let ranked = if capability_targets.is_empty() {
            Vec::new()
        } else {
            if capability_targets.len() < 2 {
                return Err(LibsyError::AlgorithmError {
                    message: "capability_targets requires at least two rungs".to_string(),
                });
            }
            if let Some(rung) = capability_targets.iter().find(|rung| !rung.is_valid()) {
                return Err(LibsyError::AlgorithmError {
                    message: format!(
                        "capability target {:?} capability must be between 0 and 1, got {}",
                        rung.target, rung.capability
                    ),
                });
            }
            let mut ranked = capability_targets;
            ranked.sort_by(|a, b| {
                a.cost_key()
                    .partial_cmp(&b.cost_key())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            ranked
        };
        let zones = capability_zones
            .map(|zones| zones.build_runtime())
            .transpose()?;
        let contract = Self::load_capability_contract(&config.contract)?;
        let session_affinity = config.session_affinity;
        let message_hash_fallback = config.message_hash_fallback;
        let judge_classifier = JudgeClassifier::new(
            StructuredJudge::new(
                TaskInput {
                    recent_turn_window: config.recent_turn_window,
                },
                contract,
                SerdeDecoder::new(),
                JudgeRuntimeConfig::new(config.max_output_tokens)?,
            ),
            judge_target.clone(),
            TaskClassifierPolicy::new(efficient_target.clone(), capable_target.clone(), &config)
                .with_ranked_targets(ranked.clone()),
        );
        // The cascade's last-resort target is the strongest rung on a ladder, else the
        // capable tier as before.
        let most_capable = |ranked: &[CapabilityTarget]| {
            ranked
                .iter()
                .max_by(|a, b| {
                    a.capability
                        .partial_cmp(&b.capability)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|rung| rung.target.clone())
                .unwrap_or_else(|| capable_target.clone())
        };
        let (targets, default_target, inner): (Vec<ModelId>, ModelId, Arc<dyn Classifier<State>>) =
            if ranked.is_empty() {
                (
                    vec![efficient_target.clone(), capable_target.clone()],
                    capable_target.clone(),
                    Arc::new(TaskClassifier {
                        classifier: judge_classifier,
                        efficient_target: efficient_target.clone(),
                        capable_target: capable_target.clone(),
                    }),
                )
            } else {
                let default = most_capable(&ranked);
                (
                    ranked.iter().map(|rung| rung.target.clone()).collect(),
                    default.clone(),
                    Arc::new(CostAwareClassifier {
                        classifier: judge_classifier,
                        ranked,
                        capable_target: default,
                        zones,
                        bandit,
                    }),
                )
            };
        Self::from_classifier(
            targets,
            inner,
            ClassifierRouteConfig {
                default_target,
                session_affinity,
                message_hash_fallback,
            },
        )
    }

    fn build_custom(
        judge_target: ModelId,
        targets: Vec<(String, ModelId)>,
        default_target: String,
        config: CustomClassifierConfig,
    ) -> Result<Self> {
        config.validate()?;
        if targets.len() < 2 {
            return Err(LibsyError::AlgorithmError {
                message: "custom classifier requires at least two targets".to_string(),
            });
        }

        let mut labels = BTreeSet::new();
        let mut resolved_names = BTreeSet::new();
        let mut target_map = BTreeMap::new();
        let mut resolved_targets = Vec::with_capacity(targets.len());
        for (label, target) in targets {
            if label.trim().is_empty() || label.trim() != label {
                return Err(LibsyError::AlgorithmError {
                    message: "custom classifier target labels must be non-empty and have no surrounding whitespace"
                        .to_string(),
                });
            }
            if !labels.insert(label.clone()) {
                return Err(LibsyError::AlgorithmError {
                    message: format!("custom classifier target label {label:?} is duplicated"),
                });
            }
            if !resolved_names.insert(target.clone()) {
                return Err(LibsyError::AlgorithmError {
                    message: format!("custom classifier resolved target {target:?} is duplicated"),
                });
            }
            target_map.insert(label, target.clone());
            resolved_targets.push(target);
        }
        let default_name =
            target_map
                .get(&default_target)
                .cloned()
                .ok_or_else(|| LibsyError::AlgorithmError {
                    message: format!(
                        "default_target {default_target:?} must be one of the configured targets"
                    ),
                })?;

        let CustomClassifierConfig {
            prompt,
            response_schema,
            policy,
            session_affinity,
            message_hash_fallback,
            recent_turn_window,
            max_output_tokens,
        } = config;
        let contract = ClassifierContract::from_inner_schema(&prompt, response_schema)?;
        let policy = match policy {
            CustomClassifierPolicy::TargetSelector { selector } => {
                CustomPolicyRuntime::TargetSelector(TargetSelectorPolicy::new(
                    selector, target_map,
                )?)
            }
        };
        let classifier: Arc<dyn Classifier<State>> = Arc::new(JudgeClassifier::new(
            StructuredJudge::new(
                TaskInput { recent_turn_window },
                contract,
                JsonSchemaDecoder::new(),
                JudgeRuntimeConfig::new(max_output_tokens)?,
            ),
            judge_target,
            policy,
        ));

        Self::from_classifier(
            resolved_targets,
            classifier,
            ClassifierRouteConfig {
                default_target: default_name,
                session_affinity,
                message_hash_fallback,
            },
        )
    }

    fn build_escalation(
        judge_target: ModelId,
        efficient_target: ModelId,
        capable_target: ModelId,
        contract_config: ClassifierContractConfig,
        config: EscalationJudgeConfig,
        max_output_tokens: u64,
    ) -> Result<Self> {
        let capable_name = capable_target.clone();
        let efficient_name = efficient_target.clone();
        let confirmations = config.confirmations;
        let esc = Arc::new(EscalationClassifier {
            judge: escalation::build_judge(
                judge_target,
                capable_name,
                efficient_name,
                &contract_config,
                config,
                max_output_tokens,
            )?,
            capable: capable_target.clone(),
            efficient: efficient_target.clone(),
            confirmations,
        });
        let inner: Arc<dyn Classifier<State>> = esc.clone();
        let targets = vec![capable_target, efficient_target];
        Ok(Self {
            route: FallThrough::<State>::new_with_state(targets)
                .with_name(ALGORITHM_NAME)
                .with_classifier(esc),
            inner,
        })
    }

    /// Loads the packaged capability-classifier contract.
    fn load_capability_contract(config: &ClassifierContractConfig) -> Result<ClassifierContract> {
        ClassifierContract::from_config(config, PROMPT_TEMPLATE, SCHEMA_TEMPLATE)
    }

    /// Keeps affinity and fallback ordering identical across judge-backed modes.
    fn from_classifier(
        targets: Vec<ModelId>,
        inner: Arc<dyn Classifier<State>>,
        config: ClassifierRouteConfig,
    ) -> Result<Self> {
        algorithm::ensure_model_is_target(&targets, &config.default_target)?;
        if config.message_hash_fallback && !config.session_affinity {
            return Err(LibsyError::AlgorithmError {
                message: "message_hash_fallback requires session_affinity".to_string(),
            });
        }
        // Affinity comes first so a retained assignment short-circuits the judge call.
        // Note: when this classifier is embedded inside another cascade (e.g. StageRouter)
        // the affinity processor never fires — only the inner score() is called.
        let mut route = FallThrough::<State>::new_with_state(targets).with_name(ALGORITHM_NAME);
        if config.session_affinity {
            let affinity = if config.message_hash_fallback {
                AffinityRouter::new().with_message_hash_fallback()
            } else {
                AffinityRouter::new()
            };
            // Both roles must share one `Arc` so the classifier reads what the processor wrote.
            let affinity = Arc::new(affinity);
            route = route
                .with_processor(affinity.clone())
                .with_classifier(affinity);
        }
        let fallback = DefaultTarget::new(config.default_target);
        Ok(Self {
            route: route
                .with_classifier(inner.clone())
                .with_classifier(Arc::new(fallback)),
            inner,
        })
    }
}

#[async_trait]
impl Classifier<State> for TaskClassifier {
    fn routing_tier(&self, selected_model_id: &ModelId) -> Option<&'static str> {
        if self.efficient_target == self.capable_target {
            None
        } else if *selected_model_id == self.efficient_target {
            Some("weak")
        } else if *selected_model_id == self.capable_target {
            Some("strong")
        } else {
            None
        }
    }

    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        self.classifier.score(state, request, driver).await
    }
}

#[async_trait]
impl Classifier<State> for LlmTaskClassifier {
    fn routing_tier(&self, selected_model_id: &ModelId) -> Option<&'static str> {
        self.inner.routing_tier(selected_model_id)
    }

    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        self.inner.score(state, request, driver).await
    }
}

#[async_trait]
impl Algorithm for LlmTaskClassifier {
    fn name(&self) -> &str {
        "llm_task_classifier"
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<Response> {
        self.route.execute(driver, request).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parking_lot::Mutex;
    use serde_json::Value;

    use super::*;
    use switchyard_protocol::{
        ContentBlock, InstructionBlock, LlmClientError, LlmRequest, LlmResponseChunk, Metadata,
        ToolCall, ToolResult, completion_text, text_request, text_response,
    };

    use crate::algorithms::util::llm_judge::Judge;
    use crate::core::testing::{Serve, reply, test_drive};
    use switchyard_protocol::{LlmResponse, Response};

    const TEST_THRESHOLD: f64 = 0.5;

    fn test_config(base_threshold: f64) -> TaskClassifierConfig {
        TaskClassifierConfig {
            base_threshold,
            ..TaskClassifierConfig::default()
        }
    }

    fn policy() -> TaskClassifierPolicy {
        TaskClassifierPolicy::new("efficient", "capable", &test_config(TEST_THRESHOLD))
    }

    fn verdict(
        p_solve: f64,
        capability_boundary: &str,
        primary_rule: &str,
    ) -> TaskClassifierVerdict {
        TaskClassifierVerdict {
            crux: "test crux".to_string(),
            primary_rule: primary_rule.to_string(),
            capability_boundary: capability_boundary.to_string(),
            p_solve,
            minimum_capability: None,
        }
    }

    /// Verdict with an explicit `minimum_capability`, for ranked-ladder tests.
    fn ranked_verdict(
        p_solve: f64,
        capability_boundary: &str,
        primary_rule: &str,
        minimum_capability: f64,
    ) -> TaskClassifierVerdict {
        TaskClassifierVerdict {
            minimum_capability: Some(minimum_capability),
            ..verdict(p_solve, capability_boundary, primary_rule)
        }
    }

    fn selected(
        policy: &TaskClassifierPolicy,
        verdict: Option<&TaskClassifierVerdict>,
    ) -> Result<ModelId> {
        policy
            .to_classification(verdict)
            .argmax(false)?
            .map(|score| score.target)
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "policy abstained".to_string(),
            })
    }

    /// Records what each target received; answers the judge with a supported verdict and
    /// every other target with a plain completion.
    #[derive(Default)]
    struct Recorder {
        calls: Mutex<Vec<String>>,
        call_roles: Mutex<Vec<(String, bool)>>,
        judge_max_output_tokens: Mutex<Vec<Option<u64>>>,
        judge_system_prompts: Mutex<Vec<String>>,
    }

    impl Recorder {
        fn calls(&self) -> Vec<String> {
            self.calls.lock().clone()
        }

        fn call_roles(&self) -> Vec<(String, bool)> {
            self.call_roles.lock().clone()
        }

        fn judge_max_output_tokens(&self) -> Vec<Option<u64>> {
            self.judge_max_output_tokens.lock().clone()
        }

        fn judge_system_prompts(&self) -> Vec<String> {
            self.judge_system_prompts.lock().clone()
        }

        fn serve(self: &Arc<Self>) -> impl Serve {
            let recorder = Arc::clone(self);
            move |model: ModelId, request: Request| {
                let recorder = Arc::clone(&recorder);
                async move {
                    let model = model.to_string();
                    recorder.calls.lock().push(model.clone());
                    recorder
                        .call_roles
                        .lock()
                        .push((model.clone(), model != "judge"));
                    let completion = if model == "judge" {
                        recorder
                            .judge_max_output_tokens
                            .lock()
                            .push(request.llm_request.output.max_output_tokens);
                        recorder.judge_system_prompts.lock().extend(
                            request
                                .llm_request
                                .instructions
                                .first()
                                .and_then(|instruction| {
                                    instruction.content.iter().find_map(|b| {
                                        if let ContentBlock::Text { text } = b {
                                            Some(text.clone())
                                        } else {
                                            None
                                        }
                                    })
                                }),
                        );
                        r#"{"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.9}"#.to_string()
                    } else {
                        format!("answer from {model}")
                    };
                    Ok(Response {
                        llm_response: LlmResponse::Agg(text_response(None, completion)),
                        metadata: request.metadata,
                    })
                }
            }
        }
    }

    /// The judge times out; every other target answers normally.
    fn unreachable_judge() -> impl Serve {
        |model: ModelId, request: Request| async move {
            let model = model.to_string();
            if model == "judge" {
                return Err(LlmClientError::Timeout {
                    source: Box::new(std::io::Error::other("judge unreachable")),
                });
            }
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(None, format!("answer from {model}"))),
                metadata: request.metadata,
            })
        }
    }

    fn router() -> Result<Arc<LlmTaskClassifier>> {
        Ok(Arc::new(LlmTaskClassifier::new(
            LlmClassifierConfig::Capability {
                judge_target: ModelId::from("judge"),
                efficient_target: ModelId::from("efficient"),
                capable_target: ModelId::from("capable"),
                capability_targets: Vec::new(),
                capability_zones: None,
                bandit: None,
                config: test_config(TEST_THRESHOLD),
            },
        )?))
    }

    fn classify_request() -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "classify this task"),
            raw_request: None,
            metadata: None,
        }
    }

    fn classify_session_request() -> Request {
        Request {
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                ..Metadata::default()
            }),
            ..classify_request()
        }
    }

    fn classify_follow_up_request() -> Request {
        let mut request = classify_request();
        request
            .llm_request
            .messages
            .push(Message::text(Role::Assistant, "I will add the test."));
        request.llm_request.messages.push(Message::text(
            Role::User,
            "Now run the test suite and report the result.",
        ));
        request
    }

    #[tokio::test]
    async fn an_unreachable_judge_routes_capable_instead_of_failing_the_request() -> Result<()> {
        let router = router()?;

        let (trace, response) = test_drive(router, classify_request(), unreachable_judge()).await?;

        assert_eq!(
            trace.last().map(|d| d.selected_model_id().as_str()),
            Some("capable")
        );
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("answer from capable".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn classifier_judges_each_request_without_affinity() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        let router = router()?;
        let request = classify_request;

        test_drive(router.clone(), request(), recorder.serve()).await?;
        test_drive(router.clone(), request(), recorder.serve()).await?;

        assert_eq!(
            recorder.calls(),
            vec!["judge", "efficient", "judge", "efficient"]
        );
        assert_eq!(
            recorder.call_roles(),
            vec![
                ("judge".to_string(), false),
                ("efficient".to_string(), true),
                ("judge".to_string(), false),
                ("efficient".to_string(), true),
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn classifier_config_sets_the_judge_completion_cap() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        let router = Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Capability {
            judge_target: ModelId::from("judge"),
            efficient_target: ModelId::from("efficient"),
            capable_target: ModelId::from("capable"),
            capability_targets: Vec::new(),
            capability_zones: None,
            bandit: None,
            config: TaskClassifierConfig {
                max_output_tokens: 512,
                ..test_config(TEST_THRESHOLD)
            },
        })?);

        test_drive(router, classify_request(), recorder.serve()).await?;

        assert_eq!(recorder.judge_max_output_tokens(), vec![Some(512)]);
        Ok(())
    }

    #[tokio::test]
    async fn classifier_config_overrides_the_packaged_prompt() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        let router = Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Capability {
            judge_target: ModelId::from("judge"),
            efficient_target: ModelId::from("efficient"),
            capable_target: ModelId::from("capable"),
            capability_targets: Vec::new(),
            capability_zones: None,
            bandit: None,
            config: TaskClassifierConfig {
                contract: ClassifierContractConfig::default()
                    .with_prompt("Custom capability rubric."),
                ..test_config(TEST_THRESHOLD)
            },
        })?);

        test_drive(router, classify_request(), recorder.serve()).await?;

        let prompts = recorder.judge_system_prompts();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0], "Custom capability rubric.");
        Ok(())
    }

    #[tokio::test]
    async fn classifier_config_enables_session_affinity() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        let router = Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Capability {
            judge_target: ModelId::from("judge"),
            efficient_target: ModelId::from("efficient"),
            capable_target: ModelId::from("capable"),
            capability_targets: Vec::new(),
            capability_zones: None,
            bandit: None,
            config: TaskClassifierConfig {
                session_affinity: true,
                ..test_config(TEST_THRESHOLD)
            },
        })?);

        let session_request = classify_session_request;
        test_drive(router.clone(), session_request(), recorder.serve()).await?;
        test_drive(router.clone(), session_request(), recorder.serve()).await?;

        assert_eq!(recorder.calls(), vec!["judge", "efficient", "efficient"]);
        Ok(())
    }

    #[tokio::test]
    async fn classifier_config_reuses_message_hash_affinity_for_a_follow_up() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        let router = Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Capability {
            judge_target: ModelId::from("judge"),
            efficient_target: ModelId::from("efficient"),
            capable_target: ModelId::from("capable"),
            capability_targets: Vec::new(),
            capability_zones: None,
            bandit: None,
            config: TaskClassifierConfig {
                session_affinity: true,
                message_hash_fallback: true,
                recent_turn_window: None,
                ..test_config(TEST_THRESHOLD)
            },
        })?);

        test_drive(router.clone(), classify_request(), recorder.serve()).await?;
        test_drive(
            router.clone(),
            classify_follow_up_request(),
            recorder.serve(),
        )
        .await?;

        assert_eq!(recorder.calls(), vec!["judge", "efficient", "efficient"]);
        Ok(())
    }

    #[test]
    fn the_threshold_boundary_is_inclusive() -> Result<()> {
        let policy = policy();
        let at_threshold = verdict(0.5, "supported", "SUP-1");
        let below_threshold = verdict(0.49, "supported", "SUP-1");
        assert_eq!(selected(&policy, Some(&at_threshold))?, "efficient");
        assert_eq!(selected(&policy, Some(&below_threshold))?, "capable");
        Ok(())
    }

    #[test]
    fn the_threshold_moves_the_routing_boundary() -> Result<()> {
        let borderline = verdict(0.5, "supported", "SUP-1");
        let strict = TaskClassifierPolicy::new("efficient", "capable", &test_config(0.9));
        let lenient = TaskClassifierPolicy::new("efficient", "capable", &test_config(0.1));
        assert_eq!(selected(&strict, Some(&borderline))?, "capable");
        assert_eq!(selected(&lenient, Some(&borderline))?, "efficient");
        Ok(())
    }

    #[test]
    fn classifier_config_rejects_unknown_fields() {
        let error = serde_json::from_value::<TaskClassifierConfig>(serde_json::json!({
            "base_threshold": 0.5,
            "classifier_magic": true,
        }))
        .expect_err("unknown classifier fields must be rejected");

        assert!(
            error
                .to_string()
                .contains("unknown field `classifier_magic`"),
            "{error}"
        );
    }

    #[test]
    fn invalid_classifier_config_is_rejected() -> Result<()> {
        for bad in [1.5, -0.1, f64::NAN, f64::INFINITY] {
            assert!(
                LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                    judge_target: ModelId::from("judge"),
                    efficient_target: ModelId::from("e"),
                    capable_target: ModelId::from("c"),
                    capability_targets: Vec::new(),
                    capability_zones: None,
                    bandit: None,
                    config: test_config(bad),
                })
                .is_err(),
                "base threshold {bad} should be rejected"
            );
        }
        for config in [
            TaskClassifierConfig {
                base_threshold: 0.5,
                threshold_step: -0.1,
                ..TaskClassifierConfig::default()
            },
            TaskClassifierConfig {
                base_threshold: 0.8,
                threshold_step: 0.11,
                ..TaskClassifierConfig::default()
            },
            TaskClassifierConfig {
                base_threshold: 0.5,
                message_hash_fallback: true,
                ..TaskClassifierConfig::default()
            },
            TaskClassifierConfig {
                base_threshold: 0.5,
                max_output_tokens: 0,
                ..TaskClassifierConfig::default()
            },
        ] {
            assert!(
                LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                    judge_target: ModelId::from("judge"),
                    efficient_target: ModelId::from("e"),
                    capable_target: ModelId::from("c"),
                    capability_targets: Vec::new(),
                    capability_zones: None,
                    bandit: None,
                    config,
                })
                .is_err()
            );
        }
        for base_threshold in [0.0, 1.0] {
            LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                judge_target: ModelId::from("judge"),
                efficient_target: ModelId::from("e"),
                capable_target: ModelId::from("c"),
                capability_targets: Vec::new(),
                capability_zones: None,
                bandit: None,
                config: test_config(base_threshold),
            })?;
        }
        Ok(())
    }

    #[test]
    fn an_unusable_verdict_is_ambiguous() -> Result<()> {
        let policy = policy();
        let inconsistent_rule = TaskClassifierVerdict {
            capability_boundary: "uncertain".to_string(),
            ..verdict(1.0, "supported", "SUP-1")
        };
        let empty_crux = TaskClassifierVerdict {
            crux: "  ".to_string(),
            ..verdict(1.0, "supported", "SUP-1")
        };
        let unusable = [
            Some(verdict(1.1, "supported", "SUP-1")),
            Some(inconsistent_rule),
            Some(empty_crux),
            None,
        ];
        for verdict in unusable {
            let classification = policy.to_classification(verdict.as_ref());
            assert!(matches!(classification, Classification::Ambiguous(_)));
            assert!(classification.argmax(false)?.is_none());
            assert!(classification.argmax(true)?.is_none());
        }
        Ok(())
    }

    #[test]
    fn capability_boundaries_apply_monotonic_threshold_steps() -> Result<()> {
        let policy = TaskClassifierPolicy::new(
            "efficient",
            "capable",
            &TaskClassifierConfig {
                threshold_step: 0.1,
                ..test_config(0.4)
            },
        );

        assert_eq!(
            selected(&policy, Some(&verdict(0.4, "supported", "SUP-2")))?,
            "efficient"
        );
        assert_eq!(
            selected(&policy, Some(&verdict(0.49, "uncertain", "UNC-1")))?,
            "capable"
        );
        assert_eq!(
            selected(&policy, Some(&verdict(0.5, "uncertain", "UNC-1")))?,
            "efficient"
        );
        assert_eq!(
            selected(&policy, Some(&verdict(0.5, "unmatched", "none")))?,
            "efficient"
        );
        assert_eq!(
            selected(&policy, Some(&verdict(0.59, "unsupported", "LIM-1")))?,
            "capable"
        );
        assert_eq!(
            selected(&policy, Some(&verdict(0.6, "unsupported", "LIM-1")))?,
            "efficient"
        );
        Ok(())
    }

    /// A four-rung ladder in cost order: nano < strong < ultra < opus by price.
    fn ranked_policy() -> TaskClassifierPolicy {
        let rung = |name: &str, capability: f64, price: f64| CapabilityTarget {
            target: ModelId::from(name),
            capability,
            cost: switchyard_protocol::TargetCost {
                input_per_1m: price,
                output_per_1m: price,
            },
            context_window: None,
        };
        TaskClassifierPolicy::new("efficient", "capable", &test_config(TEST_THRESHOLD))
            .with_ranked_targets(vec![
                rung("nano", 0.2, 0.1),
                rung("strong", 0.5, 0.5),
                rung("ultra", 0.8, 1.0),
                rung("opus", 1.0, 3.0),
            ])
    }

    #[test]
    fn ranked_pick_is_the_cheapest_rung_that_clears_the_judged_level() -> Result<()> {
        let policy = ranked_policy();
        assert_eq!(
            selected(
                &policy,
                Some(&ranked_verdict(0.9, "supported", "SUP-1", 0.1))
            )?,
            "nano"
        );
        assert_eq!(
            selected(
                &policy,
                Some(&ranked_verdict(0.5, "supported", "SUP-1", 0.5))
            )?,
            "strong"
        );
        assert_eq!(
            selected(
                &policy,
                Some(&ranked_verdict(0.3, "uncertain", "UNC-1", 0.9))
            )?,
            "opus"
        );
        Ok(())
    }

    #[test]
    fn ranked_pick_falls_through_to_the_most_capable_rung() -> Result<()> {
        let policy = ranked_policy();
        // An unsupported boundary distrusts the cheap tiers regardless of level.
        assert_eq!(
            selected(
                &policy,
                Some(&ranked_verdict(0.9, "unsupported", "LIM-1", 0.1))
            )?,
            "opus"
        );
        // A required level above every rung lands on the strongest rung too.
        assert_eq!(
            selected(
                &policy,
                Some(&ranked_verdict(0.9, "supported", "SUP-1", 1.0))
            )?,
            "opus"
        );
        // A verdict without a usable level cannot tell the rungs apart.
        assert_eq!(
            selected(&policy, Some(&verdict(0.9, "supported", "SUP-1")))?,
            "opus"
        );
        Ok(())
    }

    #[test]
    fn ranked_pick_orders_by_cost_not_declaration_order() -> Result<()> {
        // Declare the rungs dearest-first; the ladder must still pick the cheapest
        // adequate rung, so the policy orders by cost, not config order.
        let rung = |name: &str, capability: f64, price: f64| CapabilityTarget {
            target: ModelId::from(name),
            capability,
            cost: switchyard_protocol::TargetCost {
                input_per_1m: price,
                output_per_1m: price,
            },
            context_window: None,
        };
        let router = LlmTaskClassifier::new(LlmClassifierConfig::Capability {
            judge_target: ModelId::from("judge"),
            efficient_target: ModelId::from("cheap"),
            capable_target: ModelId::from("opus"),
            capability_targets: vec![
                rung("opus", 1.0, 3.0),
                rung("cheap", 0.5, 0.1),
                rung("mid", 0.5, 0.4),
            ],
            capability_zones: None,
            bandit: None,
            config: test_config(TEST_THRESHOLD),
        })?;
        let _ = router;
        // The build sorts by cost: cheap (0.2 total) < mid (0.8) < opus (6.0). A 0.5
        // requirement clears both cheap and mid; the cheapest adequate rung wins.
        let policy = TaskClassifierPolicy::new("cheap", "opus", &test_config(TEST_THRESHOLD))
            .with_ranked_targets({
                let mut rungs = vec![
                    rung("opus", 1.0, 3.0),
                    rung("cheap", 0.5, 0.1),
                    rung("mid", 0.5, 0.4),
                ];
                rungs.sort_by(|a, b| {
                    a.cost_key()
                        .partial_cmp(&b.cost_key())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                rungs
            });
        assert_eq!(
            selected(
                &policy,
                Some(&ranked_verdict(0.9, "supported", "SUP-1", 0.5))
            )?,
            "cheap"
        );
        Ok(())
    }

    #[test]
    fn a_ladder_of_one_or_an_invalid_level_is_rejected() {
        let rung = |name: &str, capability: f64| CapabilityTarget {
            target: ModelId::from(name),
            capability,
            cost: switchyard_protocol::TargetCost::default(),
            context_window: None,
        };
        // One rung is not a ladder.
        assert!(
            LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                judge_target: ModelId::from("judge"),
                efficient_target: ModelId::from("e"),
                capable_target: ModelId::from("c"),
                capability_targets: vec![rung("only", 0.5)],
                capability_zones: None,
                bandit: None,
                config: test_config(TEST_THRESHOLD),
            })
            .is_err()
        );
        // Capability levels outside 0..=1 are rejected.
        assert!(
            LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                judge_target: ModelId::from("judge"),
                efficient_target: ModelId::from("e"),
                capable_target: ModelId::from("c"),
                capability_targets: vec![rung("a", 0.5), rung("b", 1.5)],
                capability_zones: None,
                bandit: None,
                config: test_config(TEST_THRESHOLD),
            })
            .is_err()
        );
    }

    #[test]
    fn the_context_prefilter_skips_rungs_that_cannot_fit_the_request() -> Result<()> {
        let rung =
            |name: &str, capability: f64, price: f64, window: Option<u32>| CapabilityTarget {
                target: ModelId::from(name),
                capability,
                cost: switchyard_protocol::TargetCost {
                    input_per_1m: price,
                    output_per_1m: price,
                },
                context_window: window,
            };
        // Two same-capability rungs: a cheap one with a tiny window and a pricier roomy one.
        let policy =
            TaskClassifierPolicy::new("efficient", "capable", &test_config(TEST_THRESHOLD))
                .with_ranked_targets(vec![
                    rung("tiny", 0.5, 0.1, Some(1_000)),
                    rung("roomy", 0.5, 0.5, Some(1_000_000)),
                ]);
        let verdict = ranked_verdict(0.9, "supported", "SUP-1", 0.5);

        // A 5_000-token request overflows the tiny window, so the roomy rung serves it.
        let big = policy
            .classify(&verdict, Some(5_000))
            .argmax(false)?
            .map(|score| score.target);
        assert_eq!(big, Some(ModelId::from("roomy")));

        // A small request fits both, so the cheaper one wins.
        let small = policy
            .classify(&verdict, Some(100))
            .argmax(false)?
            .map(|score| score.target);
        assert_eq!(small, Some(ModelId::from("tiny")));
        Ok(())
    }

    #[test]
    fn an_unknown_context_window_is_never_a_reason_to_skip_a_rung() -> Result<()> {
        let rung =
            |name: &str, capability: f64, price: f64, window: Option<u32>| CapabilityTarget {
                target: ModelId::from(name),
                capability,
                cost: switchyard_protocol::TargetCost {
                    input_per_1m: price,
                    output_per_1m: price,
                },
                context_window: window,
            };
        let policy =
            TaskClassifierPolicy::new("efficient", "capable", &test_config(TEST_THRESHOLD))
                .with_ranked_targets(vec![rung("unknown", 0.5, 0.1, None)]);
        let verdict = ranked_verdict(0.9, "supported", "SUP-1", 0.5);
        // Even a huge estimated size cannot prefilter a rung with no declared window.
        let pick = policy
            .classify(&verdict, Some(u64::MAX))
            .argmax(false)?
            .map(|score| score.target);
        assert_eq!(pick, Some(ModelId::from("unknown")));
        Ok(())
    }

    // ── Confidence zones and fan-out ────────────────────────────────────────

    fn cost_rung(name: &str, capability: f64, price: f64) -> CapabilityTarget {
        CapabilityTarget {
            target: ModelId::from(name),
            capability,
            cost: switchyard_protocol::TargetCost {
                input_per_1m: price,
                output_per_1m: price,
            },
            context_window: None,
        }
    }

    fn test_zones() -> ZoneConfig {
        ZoneConfig {
            low_threshold: 0.3,
            high_threshold: 0.7,
            fan_out: 3,
            output_judge_target: ModelId::from("output_judge"),
            output_judge_contract: ClassifierContractConfig::default(),
            output_judge_max_output_tokens: 256,
        }
    }

    /// A four-rung ladder (nano < strong < ultra < opus by cost) with zones enabled.
    fn zoned_router() -> Result<Arc<LlmTaskClassifier>> {
        Ok(Arc::new(LlmTaskClassifier::new(
            LlmClassifierConfig::Capability {
                judge_target: ModelId::from("judge"),
                efficient_target: ModelId::from("nano"),
                capable_target: ModelId::from("opus"),
                capability_targets: vec![
                    cost_rung("nano", 0.2, 0.1),
                    cost_rung("strong", 0.5, 0.5),
                    cost_rung("ultra", 0.8, 1.0),
                    cost_rung("opus", 1.0, 3.0),
                ],
                capability_zones: Some(test_zones()),
                bandit: None,
                config: test_config(TEST_THRESHOLD),
            },
        )?))
    }

    /// Serves the capability judge with a fixed verdict, the output judge with a fixed
    /// winner, and every answer target with distinguishable prose. Records each call.
    fn recording_zone_serve(
        calls: Arc<Mutex<Vec<String>>>,
        p_solve: f64,
        boundary: &str,
        minimum_capability: f64,
        winner: usize,
    ) -> impl Serve {
        let boundary = boundary.to_string();
        move |model: ModelId, _request: Request| {
            let calls = Arc::clone(&calls);
            let boundary = boundary.clone();
            let model = model.to_string();
            async move {
                calls.lock().push(model.clone());
                match model.as_str() {
                    "judge" => Ok(reply(format!(
                        r#"{{"crux":"crux","primary_rule":"SUP-1","capability_boundary":"{boundary}","p_solve":{p_solve},"minimum_capability":{minimum_capability}}}"#
                    ))),
                    "output_judge" => {
                        Ok(reply(format!(r#"{{"winner":{winner},"reason":"best"}}"#)))
                    }
                    other => Ok(reply(format!("answer from {other}"))),
                }
            }
        }
    }

    #[tokio::test]
    async fn zone_a_answers_with_a_single_cheapest_adequate_call() -> Result<()> {
        let router = zoned_router()?;
        let calls = Arc::new(Mutex::new(Vec::new()));
        let serve = recording_zone_serve(Arc::clone(&calls), 0.9, "supported", 0.2, 0);

        let (_trace, response) = test_drive(router, classify_request(), serve).await?;

        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("answer from nano".to_string())
        );
        // No fan-out, no output judge: judge then the one cheap answer call.
        assert_eq!(*calls.lock(), vec!["judge".to_string(), "nano".to_string()]);
        Ok(())
    }

    #[tokio::test]
    async fn zone_c_routes_to_the_most_capable_rung_without_fanning_out() -> Result<()> {
        let router = zoned_router()?;
        let calls = Arc::new(Mutex::new(Vec::new()));
        let serve = recording_zone_serve(Arc::clone(&calls), 0.1, "supported", 0.2, 0);

        let (_trace, response) = test_drive(router, classify_request(), serve).await?;

        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("answer from opus".to_string())
        );
        assert_eq!(*calls.lock(), vec!["judge".to_string(), "opus".to_string()]);
        Ok(())
    }

    #[tokio::test]
    async fn an_unsupported_boundary_forces_zone_c_even_at_high_confidence() -> Result<()> {
        let router = zoned_router()?;
        let calls = Arc::new(Mutex::new(Vec::new()));
        let serve = recording_zone_serve(Arc::clone(&calls), 0.95, "unsupported", 0.2, 0);

        let (_trace, response) = test_drive(router, classify_request(), serve).await?;

        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("answer from opus".to_string())
        );
        assert_eq!(*calls.lock(), vec!["judge".to_string(), "opus".to_string()]);
        Ok(())
    }

    #[tokio::test]
    async fn zone_b_fans_out_and_the_output_judge_picks_the_winner() -> Result<()> {
        let router = zoned_router()?;
        let calls = Arc::new(Mutex::new(Vec::new()));
        // Mid confidence, level 0.2: every rung is eligible, so the cheapest three fan out.
        let serve = recording_zone_serve(Arc::clone(&calls), 0.5, "supported", 0.2, 1);

        let (_trace, response) = test_drive(router, classify_request(), serve).await?;

        // Winner index 1 is "strong".
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("answer from strong".to_string())
        );
        let calls = calls.lock().clone();
        assert_eq!(calls.first().map(String::as_str), Some("judge"));
        assert_eq!(calls.last().map(String::as_str), Some("output_judge"));
        let mut fanned = calls[1..calls.len() - 1].to_vec();
        fanned.sort();
        assert_eq!(fanned, vec!["nano", "strong", "ultra"]);
        Ok(())
    }

    #[tokio::test]
    async fn zone_b_fans_out_only_over_the_rungs_that_clear_the_level() -> Result<()> {
        let router = zoned_router()?;
        let calls = Arc::new(Mutex::new(Vec::new()));
        // Level 0.6 leaves ultra and opus eligible; only those two fan out.
        let serve = recording_zone_serve(Arc::clone(&calls), 0.5, "supported", 0.6, 1);

        let (_trace, response) = test_drive(router, classify_request(), serve).await?;

        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("answer from opus".to_string())
        );
        let calls = calls.lock().clone();
        assert_eq!(calls.first().map(String::as_str), Some("judge"));
        assert_eq!(calls.last().map(String::as_str), Some("output_judge"));
        let mut fanned = calls[1..calls.len() - 1].to_vec();
        fanned.sort();
        assert_eq!(fanned, vec!["opus", "ultra"]);
        Ok(())
    }

    #[tokio::test]
    async fn zone_b_falls_back_to_the_cheapest_candidate_when_the_judge_misfires() -> Result<()> {
        let router = zoned_router()?;
        let calls = Arc::new(Mutex::new(Vec::new()));
        // Winner index 9 is out of range for three candidates: fall back to index 0.
        let serve = recording_zone_serve(Arc::clone(&calls), 0.5, "supported", 0.2, 9);

        let (_trace, response) = test_drive(router, classify_request(), serve).await?;

        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("answer from nano".to_string())
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn zone_b_calls_its_candidates_concurrently() -> Result<()> {
        // Each candidate blocks on a barrier of three: without concurrent service the
        // barrier never fills and the run times out, so passing proves concurrency.
        let router = zoned_router()?;
        let barrier = Arc::new(tokio::sync::Barrier::new(3));
        let serve = {
            let barrier = Arc::clone(&barrier);
            move |model: ModelId, _request: Request| {
                let barrier = Arc::clone(&barrier);
                let model = model.to_string();
                async move {
                    match model.as_str() {
                        "judge" => Ok(reply(
                            r#"{"crux":"c","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.5,"minimum_capability":0.2}"#
                                .to_string(),
                        )),
                        "output_judge" => Ok(reply(r#"{"winner":0,"reason":"best"}"#.to_string())),
                        other => {
                            barrier.wait().await;
                            Ok(reply(format!("answer from {other}")))
                        }
                    }
                }
            }
        };

        let run = test_drive(router, classify_request(), serve);
        let (_trace, response) = tokio::time::timeout(std::time::Duration::from_secs(5), run)
            .await
            .map_err(|error| LibsyError::external("zone B fan-out deadlocked", error))??;

        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("answer from nano".to_string())
        );
        Ok(())
    }

    /// A zoned router with a bandit over the shared `sampler`.
    fn bandit_router(sampler: Arc<ThompsonSampler>, scale: f64) -> Result<Arc<LlmTaskClassifier>> {
        let mut zones = test_zones();
        // Keep Zone B out of the way so the correction's effect is a clean A↔C shift.
        zones.low_threshold = 0.3;
        zones.high_threshold = 0.7;
        Ok(Arc::new(LlmTaskClassifier::new(
            LlmClassifierConfig::Capability {
                judge_target: ModelId::from("judge"),
                efficient_target: ModelId::from("nano"),
                capable_target: ModelId::from("opus"),
                capability_targets: vec![
                    cost_rung("nano", 0.2, 0.1),
                    cost_rung("strong", 0.5, 0.5),
                    cost_rung("ultra", 0.8, 1.0),
                    cost_rung("opus", 1.0, 3.0),
                ],
                capability_zones: Some(zones),
                bandit: Some(BanditConfig { sampler, scale }),
                config: test_config(TEST_THRESHOLD),
            },
        )?))
    }

    #[tokio::test]
    async fn a_failing_cheap_arm_pushes_the_route_off_zone_a() -> Result<()> {
        // The judge is confident (Zone A territory), but nano's arm has only failures,
        // so the corrected confidence drops below the low threshold and the route lands
        // on the most capable rung instead.
        let sampler = Arc::new(ThompsonSampler::new());
        // A hundred failures: Beta(1, 101) samples sit near 0.01, deterministically
        // dragging 0.75 + (0.01 - 0.5) * 1.0 below the 0.3 low threshold.
        sampler.update_arm(&ModelId::from("nano"), "small", 1.0, 101.0);
        let router = bandit_router(sampler, 1.0)?;
        let calls = Arc::new(Mutex::new(Vec::new()));
        let serve = recording_zone_serve(Arc::clone(&calls), 0.75, "supported", 0.2, 0);

        let (_trace, response) = test_drive(router, classify_request(), serve).await?;

        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("answer from opus".to_string())
        );
        assert_eq!(*calls.lock(), vec!["judge".to_string(), "opus".to_string()]);
        Ok(())
    }

    #[tokio::test]
    async fn a_succeeding_cheap_arm_keeps_the_route_cheap() -> Result<()> {
        // A hundred successes: Beta(101, 1) samples near 0.99, pushing the corrected
        // confidence up, so the route stays on the cheap rung.
        let sampler = Arc::new(ThompsonSampler::new());
        sampler.update_arm(&ModelId::from("nano"), "small", 101.0, 1.0);
        let router = bandit_router(sampler, 1.0)?;
        let calls = Arc::new(Mutex::new(Vec::new()));
        let serve = recording_zone_serve(Arc::clone(&calls), 0.75, "supported", 0.2, 0);

        let (_trace, response) = test_drive(router, classify_request(), serve).await?;

        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("answer from nano".to_string())
        );
        assert_eq!(*calls.lock(), vec!["judge".to_string(), "nano".to_string()]);
        Ok(())
    }

    /// The text of each message a judge with `recent_turn_window` would be sent.
    /// The no-window case is covered by `capability_judge_builds_a_structured_request`.
    fn capability_judge(recent_turn_window: Option<usize>) -> Result<CapabilityJudge> {
        Ok(StructuredJudge::new(
            TaskInput { recent_turn_window },
            LlmTaskClassifier::load_capability_contract(&ClassifierContractConfig::default())?,
            SerdeDecoder::new(),
            JudgeRuntimeConfig::new(DEFAULT_JUDGE_MAX_OUTPUT_TOKENS)?,
        ))
    }

    fn judged_contents(recent_turn_window: usize) -> Result<Vec<String>> {
        let judge = capability_judge(Some(recent_turn_window))?;
        let request = Request {
            llm_request: LlmRequest {
                messages: vec![
                    Message::text(Role::System, "client instructions"),
                    Message::text(Role::User, "initial task"),
                    Message::text(Role::Assistant, "old response"),
                    Message::text(Role::User, "old follow-up"),
                    Message::text(Role::Assistant, "recent 1"),
                    Message::text(Role::User, "recent 2"),
                ],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        };
        Ok(judge
            .build_request(&State::default(), &request)
            .llm_request
            .messages
            .iter()
            .filter_map(|message| message.text_content("\n"))
            .collect())
    }

    #[test]
    fn a_window_widens_the_judge_to_the_surrounding_conversation() -> Result<()> {
        // Client instructions and the opening task, plus the last two turns.
        let contents = judged_contents(2)?;
        assert!(contents.contains(&"client instructions".to_string()));
        assert!(contents.contains(&"initial task".to_string()));
        assert!(contents.contains(&"recent 1".to_string()));
        assert!(contents.contains(&"recent 2".to_string()));
        assert!(!contents.contains(&"old response".to_string()));
        Ok(())
    }

    #[test]
    fn a_zero_window_keeps_only_the_instructions_and_the_task() -> Result<()> {
        let contents = judged_contents(0)?;
        assert!(contents.contains(&"client instructions".to_string()));
        assert!(contents.contains(&"initial task".to_string()));
        assert!(!contents.contains(&"recent 2".to_string()));
        Ok(())
    }

    fn tool_call(id: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: id.to_string(),
                name: "search".to_string(),
                arguments: Value::Null,
            })],
        }
    }

    fn tool_result(id: &str) -> Message {
        Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: id.to_string(),
                content: vec![ContentBlock::Text {
                    text: "tool output".to_string(),
                }],
                is_error: None,
            })],
        }
    }

    /// A count-based window can begin on a tool result, which leaves the call that
    /// introduced its id outside the window and the classifier history invalid.
    #[test]
    fn trimming_keeps_the_call_that_introduced_a_kept_tool_result() {
        let messages = vec![
            Message::text(Role::System, "client instructions"),
            Message::text(Role::User, "initial task"),
            Message::text(Role::Assistant, "old response"),
            tool_call("call-1"),
            tool_result("call-1"),
            Message::text(Role::Assistant, "recent 1"),
            Message::text(Role::User, "recent 2"),
            Message::text(Role::Assistant, "recent 3"),
            Message::text(Role::User, "recent 4"),
        ];

        // The five-message tail begins exactly on the tool result.
        let kept = trim_messages(&messages, 5);

        assert_eq!(
            kept,
            vec![
                Message::text(Role::System, "client instructions"),
                Message::text(Role::User, "initial task"),
                tool_call("call-1"),
                tool_result("call-1"),
                Message::text(Role::Assistant, "recent 1"),
                Message::text(Role::User, "recent 2"),
                Message::text(Role::Assistant, "recent 3"),
                Message::text(Role::User, "recent 4"),
            ]
        );
    }

    /// Ids repeat across a conversation, so a later call must not stand in for the one that
    /// answers an earlier result.
    #[test]
    fn trimming_pairs_a_repeated_id_with_the_call_that_precedes_it() {
        let messages = vec![
            Message::text(Role::System, "client instructions"),
            Message::text(Role::User, "initial task"),
            tool_call("x"),
            tool_result("x"),
            Message::text(Role::Assistant, "later"),
            tool_call("x"),
            tool_result("x"),
        ];

        // The four-message tail begins on the first result, whose own call sits one earlier.
        let kept = trim_messages(&messages, 4);

        assert_eq!(
            kept,
            vec![
                Message::text(Role::System, "client instructions"),
                Message::text(Role::User, "initial task"),
                tool_call("x"),
                tool_result("x"),
                Message::text(Role::Assistant, "later"),
                tool_call("x"),
                tool_result("x"),
            ]
        );
    }

    /// A result whose call precedes the opening task can never be paired, because trimming
    /// never reaches behind the task. The window must not widen hunting for it.
    #[test]
    fn trimming_keeps_the_counted_window_when_a_result_cannot_be_paired() {
        let messages = vec![
            Message::text(Role::System, "client instructions"),
            tool_call("orphan"),
            Message::text(Role::User, "initial task"),
            Message::text(Role::Assistant, "old response"),
            tool_result("orphan"),
            Message::text(Role::Assistant, "recent 1"),
            Message::text(Role::User, "recent 2"),
        ];

        let kept = trim_messages(&messages, 3);

        assert_eq!(
            kept,
            vec![
                Message::text(Role::System, "client instructions"),
                Message::text(Role::User, "initial task"),
                tool_result("orphan"),
                Message::text(Role::Assistant, "recent 1"),
                Message::text(Role::User, "recent 2"),
            ]
        );
    }

    /// The no-window path keeps only task text: tool blocks are stripped so the
    /// judge never receives an orphaned call or result, which Bedrock rejects.
    #[test]
    fn task_messages_strip_tool_blocks() {
        let messages = vec![
            Message::text(Role::User, "initial task"),
            tool_call("call-1"),
            Message {
                role: Role::User,
                content: vec![
                    ContentBlock::ToolResult(ToolResult {
                        tool_call_id: "call-1".to_string(),
                        content: vec![ContentBlock::Text {
                            text: "tool output".to_string(),
                        }],
                        is_error: None,
                    }),
                    ContentBlock::Text {
                        text: "latest follow-up".to_string(),
                    },
                ],
            },
        ];

        let kept = task_messages(&messages);

        assert_eq!(
            kept,
            vec![
                Message::text(Role::User, "initial task"),
                Message::text(Role::User, "latest follow-up"),
            ]
        );
    }

    /// A follow-up carrying only a tool result has no task text and is dropped,
    /// leaving the opening task alone rather than an orphaned result.
    #[test]
    fn task_messages_drop_a_follow_up_left_empty() {
        let messages = vec![
            Message::text(Role::User, "initial task"),
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult(ToolResult {
                    tool_call_id: "call-1".to_string(),
                    content: vec![ContentBlock::Text {
                        text: "tool output".to_string(),
                    }],
                    is_error: None,
                })],
            },
        ];

        let kept = task_messages(&messages);

        assert_eq!(kept, vec![Message::text(Role::User, "initial task")]);
    }

    #[test]
    fn capability_judge_builds_a_structured_request() -> Result<()> {
        let judge = capability_judge(None)?;
        let request = Request {
            llm_request: LlmRequest {
                model: Some("inbound".to_string()),
                messages: vec![
                    Message::text(Role::System, "client instructions"),
                    Message::text(Role::Developer, "client developer instructions"),
                    Message::text(Role::User, "initial task"),
                    Message::text(Role::Assistant, "old response"),
                    Message::text(Role::User, "old follow-up"),
                    Message::text(Role::Assistant, "recent 1"),
                    Message::text(Role::User, "recent 2"),
                    Message::text(Role::Assistant, "recent 3"),
                    Message::text(Role::User, "recent 4"),
                    Message::text(Role::Assistant, "recent 5"),
                ],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        };
        let judge_request = judge.build_request(&State::default(), &request);

        assert_eq!(judge_request.llm_request.model, request.llm_request.model);
        assert_eq!(judge_request.llm_request.instructions.len(), 1);
        assert_eq!(judge_request.llm_request.instructions[0].role, Role::System);
        assert_eq!(
            judge_request.llm_request.instructions[0].content,
            InstructionBlock {
                role: Role::System,
                content: Message::text(Role::System, judge.contract().system_prompt()).content,
            }
            .content,
        );
        assert_eq!(judge_request.llm_request.messages.len(), 2);
        let contents = judge_request
            .llm_request
            .messages
            .iter()
            .filter_map(|message| message.text_content("\n"))
            .collect::<Vec<_>>();
        assert!(contents.contains(&"recent 4".to_string()));
        assert!(contents.contains(&"initial task".to_string()));
        assert!(!contents.contains(&"recent 5".to_string()));
        assert!(!contents.contains(&"client instructions".to_string()));
        assert_eq!(
            judge_request.llm_request.output.response_format,
            Some(judge.contract().response_format().clone())
        );
        assert_eq!(
            judge_request.llm_request.output.max_output_tokens,
            Some(DEFAULT_JUDGE_MAX_OUTPUT_TOKENS)
        );
        Ok(())
    }

    fn sample_value(spec: &Value) -> Value {
        if let Some(first) = spec
            .get("enum")
            .and_then(Value::as_array)
            .and_then(|values| values.first())
        {
            return first.clone();
        }
        match spec.get("type").and_then(Value::as_str) {
            Some("number") => serde_json::json!(0.5),
            Some("boolean") => serde_json::json!(false),
            _ => serde_json::json!("sample"),
        }
    }

    fn schema_shaped_verdict(schema: &Value) -> Result<String> {
        let properties = schema
            .pointer("/json_schema/schema/properties")
            .and_then(Value::as_object)
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "packaged schema declares no properties".to_string(),
            })?;
        Ok(Value::Object(
            properties
                .iter()
                .map(|(name, spec)| (name.clone(), sample_value(spec)))
                .collect(),
        )
        .to_string())
    }

    /// Built from the schema so a property added there fails here rather than silently
    /// rejecting every production verdict.
    #[test]
    fn every_schema_property_round_trips_through_the_judge_parser() -> Result<()> {
        let contract =
            LlmTaskClassifier::load_capability_contract(&ClassifierContractConfig::default())?;
        let schema = contract.response_format();
        let reply = schema_shaped_verdict(schema)?;
        let judge: CapabilityJudge = StructuredJudge::new(
            TaskInput {
                recent_turn_window: None,
            },
            contract,
            SerdeDecoder::new(),
            JudgeRuntimeConfig::new(DEFAULT_JUDGE_MAX_OUTPUT_TOKENS)?,
        );

        let verdict = judge.parse(&text_response(None, reply))?;

        assert!(verdict.is_valid());
        assert!((0.0..=1.0).contains(&verdict.p_solve));
        Ok(())
    }

    #[test]
    fn packaged_prompt_keeps_the_schema_in_the_structured_request() -> Result<()> {
        let contract =
            LlmTaskClassifier::load_capability_contract(&ClassifierContractConfig::default())?;
        let prompt = contract.system_prompt();
        let schema_name = contract
            .response_format()
            .pointer("/json_schema/name")
            .and_then(Value::as_str)
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "packaged response schema has no name".to_string(),
            })?;
        assert_eq!(schema_name, "CapabilityClassifierDecision");
        assert!(prompt.contains("SUP-1 [supported]"));
        assert!(prompt.contains("SUP-5 [supported]"));
        assert!(!prompt.contains("{{RESPONSE_SCHEMA}}"));
        assert!(!prompt.contains("\"type\": \"object\""));
        assert!(!prompt.contains("\"json_schema\""));
        assert!(!prompt.contains(schema_name));
        let rule_values = contract
            .response_format()
            .pointer("/json_schema/schema/properties/primary_rule/enum")
            .and_then(Value::as_array)
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "rendered response schema has no primary rule enum".to_string(),
            })?;
        assert!(
            rule_values
                .iter()
                .any(|value| value.as_str() == Some("SUP-1"))
        );
        assert!(
            rule_values
                .iter()
                .any(|value| value.as_str() == Some("none"))
        );
        Ok(())
    }

    // ── with_escalation tests ──────────────────────────────────────────────

    use std::collections::VecDeque;

    /// A queue of replies, drained in order.
    struct Queue(Mutex<VecDeque<String>>);

    impl Queue {
        fn new(replies: impl IntoIterator<Item = &'static str>) -> Arc<Self> {
            Arc::new(Self(Mutex::new(
                replies.into_iter().map(String::from).collect(),
            )))
        }

        fn take(&self) -> String {
            self.0
                .lock()
                .pop_front()
                .unwrap_or_else(|| "unexpected call".to_string())
        }
    }

    /// Serves the judge target from `judge` and every other target from `model`, each with
    /// its next queued reply.
    fn queued(model: Arc<Queue>, judge: Arc<Queue>) -> impl Serve {
        move |target: ModelId, request: Request| {
            let queue = if target == "judge" {
                Arc::clone(&judge)
            } else {
                Arc::clone(&model)
            };
            async move {
                Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(None, queue.take())),
                    metadata: request.metadata,
                })
            }
        }
    }

    /// Returns a stream that emits partial content before failing during aggregation.
    fn streamed_then_error(error: LlmClientError) -> Response {
        Response {
            llm_response: LlmResponse::Stream(Box::pin(futures::stream::iter([
                Ok(LlmResponseChunk::TextDelta {
                    index: 0,
                    text: "partial".to_string(),
                }
                .into()),
                Err(error),
            ]))),
            metadata: None,
        }
    }

    /// Builds a router with escalation enabled (`confirmations=1` latches on the first verdict).
    fn escalation_router() -> Result<Arc<LlmTaskClassifier>> {
        Ok(Arc::new(LlmTaskClassifier::new(
            LlmClassifierConfig::Escalation {
                judge_target: ModelId::from("judge"),
                efficient_target: ModelId::from("efficient"),
                capable_target: ModelId::from("capable"),
                contract: ClassifierContractConfig::default(),
                config: EscalationJudgeConfig {
                    confirmations: 1,
                    ..EscalationJudgeConfig::default()
                },
                max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
            },
        )?))
    }

    #[tokio::test]
    async fn escalation_router_serves_efficient_when_judge_declines() -> Result<()> {
        // Judge: no escalation. Expect the efficient response to be returned directly.
        let judge = Queue::new([r#"{"escalate":false,"reason":"progressing"}"#]);
        let model = Queue::new(["efficient answer"]);
        let router = escalation_router()?;

        let (trace, response) =
            test_drive(router, classify_request(), queued(model, judge)).await?;

        // The efficient model is the serving target, and the response comes from its call.
        assert_eq!(
            trace.last().map(|d| d.selected_model_id().as_str()),
            Some("efficient")
        );
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("efficient answer".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn escalation_config_overrides_the_packaged_prompt() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        let router = Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Escalation {
            judge_target: ModelId::from("judge"),
            efficient_target: ModelId::from("efficient"),
            capable_target: ModelId::from("capable"),
            contract: ClassifierContractConfig::default().with_prompt("Custom trajectory rubric."),
            config: EscalationJudgeConfig {
                confirmations: 1,
                ..EscalationJudgeConfig::default()
            },
            max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
        })?);

        test_drive(router, classify_request(), recorder.serve()).await?;

        let prompts = recorder.judge_system_prompts();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0], "Custom trajectory rubric.");
        Ok(())
    }

    #[tokio::test]
    async fn escalation_router_upgrades_to_capable_when_judge_escalates() -> Result<()> {
        // Judge: escalate. After the efficient call, the streak confirms and capable is served.
        let judge = Queue::new([r#"{"escalate":true,"reason":"stuck in a loop"}"#]);
        // Efficient is called first (by the classifier), then capable is called by FallThrough.
        let model = Queue::new(["efficient draft", "capable answer"]);
        let router = escalation_router()?;

        let (trace, response) =
            test_drive(router, classify_request(), queued(model, judge)).await?;

        assert_eq!(
            trace.last().map(|d| d.selected_model_id().as_str()),
            Some("capable")
        );
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("capable answer".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn escalation_router_stays_capable_after_latch() -> Result<()> {
        // First turn: judge escalates and the streak latches.
        // Second turn: judge is not called again; capable is served directly.
        let judge = Queue::new([r#"{"escalate":true,"reason":"stuck"}"#]);
        let model = Queue::new(["efficient draft", "capable t1", "capable t2"]);
        let router = escalation_router()?;

        let session_request = classify_session_request();
        test_drive(
            router.clone(),
            session_request.clone(),
            queued(Arc::clone(&model), Arc::clone(&judge)),
        )
        .await?;
        let (trace, _) = test_drive(router.clone(), session_request, queued(model, judge)).await?;

        assert_eq!(
            trace.last().map(|d| d.selected_model_id().as_str()),
            Some("capable")
        );
        Ok(())
    }

    #[tokio::test]
    async fn escalation_classifier_falls_back_to_capable_when_efficient_overflows() -> Result<()> {
        // When the efficient model exceeds its context window inside score(), the classifier
        // must return capable rather than propagating the error — otherwise the client sees
        // HTTP 400 instead of a response from the strong model.
        let router = escalation_router()?;

        // Efficient overflows, capable answers, and the judge must never be called.
        let serve = |target: ModelId, _request: Request| async move {
            match target.as_str() {
                "efficient" => Err(LlmClientError::ContextWindowExceeded {
                    model: target,
                    message: "prompt is too long".to_string(),
                }),
                "judge" => panic!("the judge must not be consulted when efficient overflows"),
                _ => Ok(reply("capable answer")),
            }
        };

        let (trace, response) = test_drive(router, classify_request(), serve).await?;

        assert_eq!(
            trace.last().map(|d| d.selected_model_id().as_str()),
            Some("capable")
        );
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("capable answer".to_string())
        );
        Ok(())
    }

    /// A transport failure while buffering efficient must bypass the judge and serve capable.
    #[tokio::test]
    async fn escalation_classifier_falls_back_when_efficient_stream_transport_fails() -> Result<()>
    {
        let router = escalation_router()?;
        let calls = Arc::new(Mutex::new(Vec::new()));
        let serve = {
            let calls = Arc::clone(&calls);
            move |model: ModelId, _request: Request| {
                let calls = Arc::clone(&calls);
                async move {
                    let model = model.to_string();
                    calls.lock().push(model.clone());
                    match model.as_str() {
                        "efficient" => Ok(streamed_then_error(LlmClientError::Transport {
                            source: Box::new(std::io::Error::other("stream disconnected")),
                        })),
                        "judge" => {
                            panic!("the judge must not be consulted after a transport failure")
                        }
                        _ => Ok(reply("capable answer")),
                    }
                }
            }
        };
        let mut request = classify_request();
        request.llm_request.stream = true;

        let result = test_drive(router, request, serve).await;

        assert_eq!(&*calls.lock(), &["efficient", "capable"]);
        let (_, response) = result?;
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("capable answer".to_string())
        );
        Ok(())
    }

    /// Non-transport aggregation failures remain typed and do not silently change targets.
    #[tokio::test]
    async fn escalation_classifier_preserves_non_transport_stream_errors() -> Result<()> {
        let router = escalation_router()?;
        let serve = |target: ModelId, _request: Request| async move {
            match target.as_str() {
                "efficient" => Ok(streamed_then_error(LlmClientError::InvalidResponse {
                    source: Box::new(std::io::Error::other("invalid stream event")),
                })),
                other => panic!("unexpected call to {other}"),
            }
        };
        let mut request = classify_request();
        request.llm_request.stream = true;

        match test_drive(router, request, serve).await {
            Err(LibsyError::ClientCall {
                target,
                source: LlmClientError::InvalidResponse { .. },
            }) => {
                assert_eq!(target, "efficient");
                Ok(())
            }
            Err(other) => panic!("expected InvalidResponse client error, got {other:?}"),
            Ok(_) => panic!("expected stream aggregation to fail"),
        }
    }
}
