// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contextual Thompson sampling for the cost-aware router's confidence correction.
//!
//! Each `(target, token bucket)` pair is an arm with a `Beta(alpha, beta)` prior over
//! the reward its calls earn. The host refreshes arm parameters from the routing log;
//! the router draws one sample per request to nudge the judge's `p_solve` before zone
//! classification. Sampling is the exploration — no separate epsilon is needed.
//!
//! This is a port of TokenHub's `thompson.go` bandit. It deliberately *corrects* the
//! judge's prior rather than replacing the routing decision: the judge stays the
//! decision-maker, and the bandit only shifts how much the cheap tier is trusted.

use std::collections::BTreeMap;

use parking_lot::{Mutex, RwLock};
use rand::RngExt;
#[cfg(test)]
use rand::SeedableRng;
use rand::rngs::StdRng;
use switchyard_protocol::{ModelId, Request};

/// Coarse token bucket for bandit context, labelled from an estimated token count.
///
/// The same three labels gate selection and record rewards, so both sides must agree
/// on the boundaries — this is the single definition they share.
pub fn token_bucket(estimated_tokens: u64) -> &'static str {
    if estimated_tokens < 1_000 {
        "small"
    } else if estimated_tokens <= 10_000 {
        "medium"
    } else {
        "large"
    }
}

/// Rough input-token estimate for bandit bucketing: message text chars / 4.
///
/// A coarse proxy is fine — the bucket only separates small, medium, and large
/// requests, not exact counts. Tool-call arguments are not text, so tool-heavy
/// conversations read low; the bucket boundaries absorb that.
pub fn estimate_request_tokens(request: &Request) -> u64 {
    let chars: usize = request
        .llm_request
        .messages
        .iter()
        .filter_map(|message| message.text_content("\n"))
        .map(|text| text.len())
        .sum();
    (chars / 4) as u64
}

/// `Beta(alpha, beta)` parameters for one arm. `alpha` is the soft success count
/// (sum of rewards) plus one; `beta` the soft failure count plus one.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Arm {
    alpha: f64,
    beta: f64,
}

impl Default for Arm {
    /// The uniform prior: nothing observed yet.
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }
}

/// A contextual bandit over `(target, token bucket)` arms.
///
/// Thread-safe: the router shares one sampler across concurrent requests, and the
/// host's refresh loop rewrites arm parameters under the same lock.
pub struct ThompsonSampler {
    arms: RwLock<BTreeMap<(ModelId, String), Arm>>,
    rng: Mutex<StdRng>,
}

impl ThompsonSampler {
    /// A sampler with uniform priors and an entropy-seeded RNG.
    pub fn new() -> Self {
        Self {
            arms: RwLock::new(BTreeMap::new()),
            rng: Mutex::new(rand::make_rng()),
        }
    }

    /// A deterministically seeded sampler, for tests.
    #[cfg(test)]
    fn with_seed(seed: u64) -> Self {
        Self {
            arms: RwLock::new(BTreeMap::new()),
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }

    /// Set an arm's Beta parameters from its aggregated reward summary.
    pub fn update_arm(&self, target: &ModelId, token_bucket: &str, alpha: f64, beta: f64) {
        self.arms.write().insert(
            (target.clone(), token_bucket.to_string()),
            Arm { alpha, beta },
        );
    }

    /// Draw one Thompson sample for an arm, using the uniform prior when unseen.
    pub fn sample(&self, target: &ModelId, token_bucket: &str) -> f64 {
        let arm = self
            .arms
            .read()
            .get(&(target.clone(), token_bucket.to_string()))
            .copied()
            .unwrap_or_default();
        let mut rng = self.rng.lock();
        beta_sample(&mut rng, arm.alpha, arm.beta)
    }

    /// The arm's posterior mean, `alpha / (alpha + beta)`. Deterministic, for tests.
    #[cfg(test)]
    fn mean(&self, target: &ModelId, token_bucket: &str) -> f64 {
        let arm = self
            .arms
            .read()
            .get(&(target.clone(), token_bucket.to_string()))
            .copied()
            .unwrap_or_default();
        arm.alpha / (arm.alpha + arm.beta)
    }
}

impl Default for ThompsonSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Draw from `Beta(alpha, beta)` as `x / (x + y)` of two gamma draws.
fn beta_sample(rng: &mut StdRng, alpha: f64, beta: f64) -> f64 {
    let x = gamma_sample(rng, alpha.max(1e-10));
    let y = gamma_sample(rng, beta.max(1e-10));
    if x + y == 0.0 { 0.5 } else { x / (x + y) }
}

/// Draw from `Gamma(shape, 1)` via Marsaglia–Tsang, boosting shapes below 1.
fn gamma_sample(rng: &mut StdRng, shape: f64) -> f64 {
    if shape < 1.0 {
        // Gamma(shape) = Gamma(shape + 1) * U^(1/shape).
        return gamma_sample(rng, shape + 1.0) * rng.random::<f64>().powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = standard_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.random::<f64>();
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Box–Muller standard normal from two uniforms, avoiding a distribution dependency.
fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unseen_arm_samples_from_the_uniform_prior() {
        let sampler = ThompsonSampler::with_seed(7);
        let target = ModelId::from("m");
        // Beta(1,1) is uniform; samples stay in [0,1] and are not degenerate.
        for _ in 0..100 {
            let s = sampler.sample(&target, "small");
            assert!((0.0..=1.0).contains(&s));
        }
        assert_eq!(sampler.mean(&target, "small"), 0.5);
    }

    #[test]
    fn repeated_failures_shift_an_arm_down() {
        let sampler = ThompsonSampler::with_seed(7);
        let target = ModelId::from("m");
        // Ten failures: alpha = 0 + 1, beta = 10 + 1.
        sampler.update_arm(&target, "small", 1.0, 11.0);
        assert!(sampler.mean(&target, "small") < 0.1);
        // Samples cluster near the low mean rather than the uniform 0.5.
        let sum: f64 = (0..200).map(|_| sampler.sample(&target, "small")).sum();
        assert!(sum / 200.0 < 0.2, "mean sample {}", sum / 200.0);
    }

    #[test]
    fn repeated_successes_shift_an_arm_up() {
        let sampler = ThompsonSampler::with_seed(7);
        let target = ModelId::from("m");
        sampler.update_arm(&target, "small", 21.0, 1.0);
        assert!(sampler.mean(&target, "small") > 0.9);
        let sum: f64 = (0..200).map(|_| sampler.sample(&target, "small")).sum();
        assert!(sum / 200.0 > 0.8, "mean sample {}", sum / 200.0);
    }

    #[test]
    fn arms_are_independent_per_bucket() {
        let sampler = ThompsonSampler::with_seed(7);
        let target = ModelId::from("m");
        sampler.update_arm(&target, "small", 1.0, 11.0);
        sampler.update_arm(&target, "large", 21.0, 1.0);
        assert!(sampler.mean(&target, "small") < 0.1);
        assert!(sampler.mean(&target, "large") > 0.9);
    }
}
