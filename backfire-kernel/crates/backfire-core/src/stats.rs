// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — core::stats
//! Statistical primitives: conformal quantile, EMA, Beta posterior, Wilson
//! interval, percentile rank, reductions, normal quantile, and confusion
//! counts.
//!
//! Pure computations on pre-validated inputs — argument validation lives at
//! the FFI boundary (the pattern used by the rest of the crate, e.g.
//! [`crate::compute`]): callers guarantee finiteness/range invariants and
//! these functions guarantee the numbers.

/// Split-conformal quantile of *residuals* at *coverage*.
///
/// Sorts a copy and returns the `ceil((n + 1) * coverage)`-th order statistic
/// (1-indexed), clamped to the observed range. Caller guarantees a non-empty
/// slice of finite, non-negative residuals and `coverage` in `(0, 1)`.
pub fn conformal_quantile(residuals: &[f64], coverage: f64) -> f64 {
    let mut sorted = residuals.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let q_idx = (((n + 1) as f64 * coverage).ceil() as isize - 1).clamp(0, (n - 1) as isize);
    sorted[q_idx as usize]
}

/// Exponential moving average step; a missing *previous* seeds with *value*.
///
/// Caller guarantees finite inputs and `alpha` in `(0, 1]`.
pub fn ema_update(previous: Option<f64>, value: f64, alpha: f64) -> f64 {
    match previous {
        Some(prev) => alpha * value + (1.0 - alpha) * prev,
        None => value,
    }
}

/// Posterior mean of a Beta(alpha_prior, beta_prior) after Bernoulli pulls.
///
/// Caller guarantees finite positive priors and `successes <= pulls`.
pub fn beta_posterior_mean(
    alpha_prior: f64,
    beta_prior: f64,
    successes: usize,
    pulls: usize,
) -> f64 {
    let alpha = alpha_prior + successes as f64;
    let beta = beta_prior + (pulls - successes) as f64;
    alpha / (alpha + beta)
}

/// Wilson score interval for a proportion at the fixed 95 % z.
///
/// Returns `(0, 0)` for `n == 0` by contract. The z is fixed at the 97.5 %
/// normal quantile regardless of the caller's requested confidence — the
/// deterministic bounded output the FFI surface has always produced.
pub fn wilson_score_interval(p_hat: f64, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }
    let z = 1.959_963_984_540_054_f64; // 95 % default approximation
    let nf = n as f64;
    let denominator = 1.0 + z * z / nf;
    let centre = (p_hat + z * z / (2.0 * nf)) / denominator;
    let halfwidth =
        (z * ((p_hat * (1.0 - p_hat) / nf + z * z / (4.0 * nf * nf)).sqrt())) / denominator;
    ((centre - halfwidth).max(0.0), (centre + halfwidth).min(1.0))
}

/// Fraction of *values* less than or equal to *value*; an empty slice ranks 1.
///
/// Caller guarantees finite inputs.
pub fn percentile_rank(values: &[f64], value: f64) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    let below = values.iter().filter(|v| **v <= value).count();
    below as f64 / values.len() as f64
}

/// Arithmetic mean. Caller guarantees a non-empty slice of finite values.
pub fn mean(values: &[f64]) -> f64 {
    let sum: f64 = values.iter().sum();
    sum / values.len() as f64
}

/// Acklam's rational approximation of the standard normal quantile.
///
/// Caller guarantees `p` finite and strictly inside `(0, 1)`.
pub fn standard_normal_quantile(p: f64) -> f64 {
    let a = [
        -3.969_683_028_665_376e+01,
        2.209_460_984_245_205e+02,
        -2.759_285_104_469_687e+02,
        1.383_577_518_672_69e+02,
        -3.066_479_806_614_716e+01,
        2.506_628_277_459_239e+00,
    ];
    let b = [
        -5.447_609_879_822_406e+01,
        1.615_858_368_580_409e+02,
        -1.556_989_798_598_866e+02,
        6.680_131_188_771_972e+01,
        -1.328_068_155_288_572e+01,
    ];
    let c = [
        -7.784_894_002_430_293e-03,
        -3.223_964_580_411_365e-01,
        -2.400_758_277_161_838e+00,
        -2.549_732_539_343_734e+00,
        4.374_664_141_464_968e+00,
        2.938_163_982_698_783e+00,
    ];
    let d = [
        7.784_695_709_041_462e-03,
        3.224_671_290_700_398e-01,
        2.445_134_137_142_996e+00,
        3.754_408_661_907_416e+00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }
    let q = (-2.0 * (1.0 - p).ln()).sqrt();
    -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
}

/// Sum of finite f64 values (caller-validated).
pub fn sum_f64(values: &[f64]) -> f64 {
    values.iter().sum()
}

/// Sum of i64 values.
pub fn sum_i64(values: &[i64]) -> i64 {
    values.iter().sum()
}

/// Product of finite f64 values (caller-validated).
pub fn product_f64(values: &[f64]) -> f64 {
    values.iter().product()
}

/// Confusion counts `(tp, tn, fp, fn)` of *scores* against *labels* at
/// *threshold*; `score >= threshold` predicts positive.
///
/// Caller guarantees equal lengths and finite scores/threshold.
pub fn confusion_counts_threshold(
    scores: &[f64],
    labels: &[bool],
    threshold: f64,
) -> (usize, usize, usize, usize) {
    let mut tp = 0usize;
    let mut tn = 0usize;
    let mut fp = 0usize;
    let mut fnn = 0usize;
    for (score, label) in scores.iter().zip(labels.iter()) {
        let predicted_positive = *score >= threshold;
        match (predicted_positive, *label) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fnn += 1,
            (false, false) => tn += 1,
        }
    }
    (tp, tn, fp, fnn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conformal_quantile_orders_and_clamps() {
        // n=5: index = ceil((n+1)*coverage) - 1 = ceil(4.8) - 1 = 4 -> 0.5.
        let q = conformal_quantile(&[0.3, 0.1, 0.2, 0.4, 0.5], 0.8);
        assert!((q - 0.5).abs() < 1e-12);
        // Quantile index clamps to the largest residual at high coverage.
        let top = conformal_quantile(&[0.1, 0.9], 0.99);
        assert!((top - 0.9).abs() < 1e-12);
        // And to the smallest at very low coverage.
        let bottom = conformal_quantile(&[0.1, 0.9], 0.01);
        assert!((bottom - 0.1).abs() < 1e-12);
    }

    #[test]
    fn ema_update_seeds_and_blends() {
        // No previous value: the observation seeds the average.
        assert!((ema_update(None, 0.6, 0.5) - 0.6).abs() < 1e-12);
        // Blend: 0.5*0.0 + 0.5*1.0.
        assert!((ema_update(Some(1.0), 0.0, 0.5) - 0.5).abs() < 1e-12);
        // alpha=1 forgets the past entirely.
        assert!((ema_update(Some(0.2), 0.9, 1.0) - 0.9).abs() < 1e-12);
    }

    #[test]
    fn beta_posterior_mean_matches_closed_form() {
        // (1 + 3) / (1 + 3 + 1 + 1) with alpha=beta=1, 3/4 successes.
        let mean = beta_posterior_mean(1.0, 1.0, 3, 4);
        assert!((mean - 4.0 / 6.0).abs() < 1e-12);
        // Prior-only mean with zero pulls.
        let prior = beta_posterior_mean(2.0, 6.0, 0, 0);
        assert!((prior - 0.25).abs() < 1e-12);
    }

    #[test]
    fn wilson_interval_is_bounded() {
        let (lo, hi) = wilson_score_interval(0.9, 100);
        assert!((0.0..0.9).contains(&lo));
        assert!(hi > 0.9 && hi <= 1.0);
        // Zero samples collapse to (0, 0) by contract.
        assert_eq!(wilson_score_interval(0.5, 0), (0.0, 0.0));
        // Extremes stay clamped inside [0, 1].
        let (lo0, hi1) = wilson_score_interval(0.0, 3);
        assert!(lo0 >= 0.0 && hi1 <= 1.0);
    }

    #[test]
    fn rank_mean_sums_and_products() {
        assert!((percentile_rank(&[1.0, 2.0, 3.0], 2.0) - 2.0 / 3.0).abs() < 1e-12);
        assert!((percentile_rank(&[], 5.0) - 1.0).abs() < 1e-12);

        assert!((mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-12);

        assert!((sum_f64(&[0.5, 0.25]) - 0.75).abs() < 1e-12);
        assert!((sum_f64(&[]) - 0.0).abs() < 1e-12);
        assert_eq!(sum_i64(&[1, -2, 4]), 3);
        assert_eq!(sum_i64(&[]), 0);
        assert!((product_f64(&[2.0, 3.0]) - 6.0).abs() < 1e-12);
        assert!((product_f64(&[]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn standard_normal_quantile_hits_known_points() {
        // Symmetric around the median and matches the 97.5 % point used by
        // the Wilson interval.
        assert!(standard_normal_quantile(0.5).abs() < 1e-9);
        let z975 = standard_normal_quantile(0.975);
        assert!((z975 - 1.959_963_984_540_054).abs() < 1e-6);
        let z025 = standard_normal_quantile(0.025);
        assert!((z975 + z025).abs() < 1e-6);
        // Tail branches on both sides.
        let low = standard_normal_quantile(0.001);
        let high = standard_normal_quantile(0.999);
        assert!((low + high).abs() < 1e-6);
        assert!(low < -3.0 && high > 3.0);
    }

    #[test]
    fn confusion_counts_split_by_threshold() {
        let (tp, tn, fp, fnn) =
            confusion_counts_threshold(&[0.9, 0.8, 0.4, 0.2], &[true, false, true, false], 0.5);
        assert_eq!((tp, tn, fp, fnn), (1, 1, 1, 1));
        // Empty inputs count nothing.
        assert_eq!(confusion_counts_threshold(&[], &[], 0.5), (0, 0, 0, 0));
    }
}
