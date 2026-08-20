// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic target selection from a validated JSON classifier verdict.
//!
//! When learned per-target statistics are supplied, selection becomes a full
//! replacement of the judge's verdict: the target with the highest observed
//! posterior-mean reward wins outright. Unlike the Thompson-sampling confidence
//! correction, this is deterministic — no exploration, no sampling — so identical
//! learned stats always produce the same pick.

use std::collections::BTreeMap;

use jsonptr::PointerBuf;
use serde_json::Value;

use super::llm_judge::JudgePolicy;
use crate::core::classifier::{Classification, Score};
use crate::{LibsyError, Result};
use switchyard_protocol::ModelId;

/// Aggregated reward for one routing target, learned offline from the routing log.
///
/// `alpha` is the soft success count (sum of rewards) plus one; `beta` the soft
/// failure count plus one. The posterior mean `alpha / (alpha + beta)` is the
/// target's estimated reward and the sole basis for the deterministic pick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LearnedTargetStat {
    /// Beta prior's alpha: soft success count plus one.
    pub alpha: f64,
    /// Beta prior's beta: soft failure count plus one.
    pub beta: f64,
}

impl LearnedTargetStat {
    /// Posterior mean reward `alpha / (alpha + beta)`, the target's estimate.
    fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }
}

/// Maps one string field in a validated verdict to a configured routing target.
///
/// With learned stats attached, the verdict is only a fallback: the highest
/// mean-reward configured target is selected deterministically instead.
pub(crate) struct TargetSelectorPolicy {
    selector: PointerBuf,
    targets: BTreeMap<String, ModelId>,
    /// Learned per-label reward stats. Empty keeps the verdict-driven behavior.
    learned: BTreeMap<String, LearnedTargetStat>,
}

impl TargetSelectorPolicy {
    /// Parses a JSON Pointer used to read validated verdicts.
    pub(crate) fn new(
        selector: impl Into<String>,
        targets: BTreeMap<String, ModelId>,
    ) -> Result<Self> {
        let selector =
            PointerBuf::parse(selector.into()).map_err(|error| LibsyError::AlgorithmError {
                message: format!("policy selector is not a valid JSON Pointer: {error}"),
            })?;
        if selector.is_root() {
            return Err(LibsyError::AlgorithmError {
                message: "policy selector must identify a response field".to_string(),
            });
        }
        Ok(Self {
            selector,
            targets,
            learned: BTreeMap::new(),
        })
    }

    /// Attaches learned per-label reward stats, enabling deterministic replacement.
    ///
    /// Only stats whose label maps to a configured target are kept; the rest cannot
    /// name a routing destination and are dropped.
    pub(crate) fn with_learned(mut self, learned: BTreeMap<String, LearnedTargetStat>) -> Self {
        self.learned = learned
            .into_iter()
            .filter(|(label, _)| self.targets.contains_key(label))
            .collect();
        self
    }

    /// The learned best target: the highest mean-reward label that maps to a target.
    ///
    /// Ties break on label order (the map is sorted) so the pick is deterministic.
    fn learned_best(&self) -> Option<&ModelId> {
        self.learned
            .iter()
            .filter_map(|(label, stat)| self.targets.get(label).map(|target| (stat, target)))
            .max_by(|(a, _), (b, _)| {
                a.mean()
                    .partial_cmp(&b.mean())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, target)| target)
    }
}

impl JudgePolicy for TargetSelectorPolicy {
    type Verdict = Value;

    fn to_classification(&self, verdict: Option<&Self::Verdict>) -> Classification {
        // Learned stats fully replace the verdict: the best observed target wins,
        // deterministically, without consulting the judge's field at all.
        if let Some(best) = self.learned_best() {
            return Classification::Scores(vec![Score {
                target: best.clone(),
                confidence: 1.0,
            }]);
        }
        let target = verdict
            .and_then(|verdict| self.selector.resolve(verdict).ok())
            .and_then(Value::as_str)
            .and_then(|label| self.targets.get(label));
        match target {
            Some(target) => Classification::Scores(vec![Score {
                target: target.clone(),
                confidence: 1.0,
            }]),
            None => Classification::Ambiguous(vec![]),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::Result;

    #[test]
    fn a_verdict_selects_its_mapped_target() -> Result<()> {
        let policy = TargetSelectorPolicy::new(
            "/decision/target",
            BTreeMap::from([
                ("opus".to_string(), ModelId::from("model/opus")),
                ("sonnet".to_string(), ModelId::from("model/sonnet")),
            ]),
        )?;
        let classification = policy.to_classification(Some(&json!({
            "decision": {"target": "sonnet"}
        })));

        assert_eq!(
            classification.argmax(false)?.map(|score| score.target),
            Some(ModelId::from("model/sonnet"))
        );
        Ok(())
    }

    #[test]
    fn a_missing_or_unknown_target_abstains() -> Result<()> {
        let policy = TargetSelectorPolicy::new(
            "/target",
            BTreeMap::from([("sonnet".to_string(), ModelId::from("model/sonnet"))]),
        )?;

        assert_eq!(
            policy
                .to_classification(Some(&json!({"target": "unknown"})))
                .argmax(false)?,
            None
        );
        assert_eq!(
            policy
                .to_classification(Some(&json!({"reason": "missing"})))
                .argmax(false)?,
            None
        );
        Ok(())
    }

    #[test]
    fn an_invalid_json_pointer_is_rejected() {
        let result = TargetSelectorPolicy::new("/target~2name", BTreeMap::new());
        assert!(matches!(result, Err(LibsyError::AlgorithmError { message })
                if message.contains("valid JSON Pointer")));
    }

    #[test]
    fn learned_stats_switch_selection_to_the_stronger_target() -> Result<()> {
        // The verdict names "sonnet", but learned stats show "opus" outperforming it,
        // so the deterministic pick fully replaces the verdict with the winner.
        let policy = TargetSelectorPolicy::new(
            "/decision/target",
            BTreeMap::from([
                ("opus".to_string(), ModelId::from("model/opus")),
                ("sonnet".to_string(), ModelId::from("model/sonnet")),
            ]),
        )?
        .with_learned(BTreeMap::from([
            (
                "sonnet".to_string(),
                LearnedTargetStat {
                    alpha: 2.0,
                    beta: 8.0,
                },
            ),
            (
                "opus".to_string(),
                LearnedTargetStat {
                    alpha: 9.0,
                    beta: 1.0,
                },
            ),
        ]));

        let classification = policy.to_classification(Some(&json!({
            "decision": {"target": "sonnet"}
        })));
        assert_eq!(
            classification.argmax(false)?.map(|score| score.target),
            Some(ModelId::from("model/opus"))
        );
        Ok(())
    }

    #[test]
    fn learned_stats_ignore_labels_without_a_target() -> Result<()> {
        // A learned stat for an unmapped label cannot name a destination, so it is
        // dropped: the remaining mapped label decides the pick.
        let policy = TargetSelectorPolicy::new(
            "/target",
            BTreeMap::from([("sonnet".to_string(), ModelId::from("model/sonnet"))]),
        )?
        .with_learned(BTreeMap::from([
            (
                "ghost".to_string(),
                LearnedTargetStat {
                    alpha: 99.0,
                    beta: 1.0,
                },
            ),
            (
                "sonnet".to_string(),
                LearnedTargetStat {
                    alpha: 3.0,
                    beta: 3.0,
                },
            ),
        ]));

        assert_eq!(
            policy
                .to_classification(Some(&json!({"target": "unknown"})))
                .argmax(false)?
                .map(|score| score.target),
            Some(ModelId::from("model/sonnet"))
        );
        Ok(())
    }

    #[test]
    fn without_learned_stats_the_verdict_still_decides() -> Result<()> {
        // An empty learned map keeps the verdict-driven behavior unchanged.
        let policy = TargetSelectorPolicy::new(
            "/target",
            BTreeMap::from([
                ("opus".to_string(), ModelId::from("model/opus")),
                ("sonnet".to_string(), ModelId::from("model/sonnet")),
            ]),
        )?
        .with_learned(BTreeMap::new());

        assert_eq!(
            policy
                .to_classification(Some(&json!({"target": "opus"})))
                .argmax(false)?
                .map(|score| score.target),
            Some(ModelId::from("model/opus"))
        );
        Ok(())
    }
}
