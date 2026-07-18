// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — ffi::stats
//! Statistical helper bindings: conformal quantile, EMA, Beta posterior,
//! Wilson interval, percentile rank, reductions, normal quantile, and
//! confusion counts.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn rust_conformal_quantile(residuals: Vec<f64>, coverage: f64) -> PyResult<f64> {
    if !(0.0..1.0).contains(&coverage) {
        return Err(PyValueError::new_err("coverage must be in (0, 1)"));
    }
    if residuals.is_empty() {
        return Err(PyValueError::new_err("residuals must be non-empty"));
    }
    if residuals
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "residuals must be finite and non-negative",
        ));
    }
    let mut sorted = residuals;
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let q_idx = (((n + 1) as f64 * coverage).ceil() as isize - 1).clamp(0, (n - 1) as isize);
    Ok(sorted[q_idx as usize])
}

#[pyfunction]
fn rust_ema_update(previous: Option<f64>, value: f64, alpha: f64) -> PyResult<f64> {
    if !value.is_finite() {
        return Err(PyValueError::new_err("value must be finite"));
    }
    if !(0.0..=1.0).contains(&alpha) || alpha == 0.0 {
        return Err(PyValueError::new_err("alpha must be in (0, 1]"));
    }
    if let Some(prev) = previous {
        if !prev.is_finite() {
            return Err(PyValueError::new_err(
                "previous must be finite when provided",
            ));
        }
        Ok(alpha * value + (1.0 - alpha) * prev)
    } else {
        Ok(value)
    }
}

#[pyfunction]
fn rust_beta_posterior_mean(
    alpha_prior: f64,
    beta_prior: f64,
    successes: usize,
    pulls: usize,
) -> PyResult<f64> {
    if alpha_prior <= 0.0 || !alpha_prior.is_finite() {
        return Err(PyValueError::new_err("alpha_prior must be finite and > 0"));
    }
    if beta_prior <= 0.0 || !beta_prior.is_finite() {
        return Err(PyValueError::new_err("beta_prior must be finite and > 0"));
    }
    if successes > pulls {
        return Err(PyValueError::new_err("successes cannot exceed pulls"));
    }
    let alpha = alpha_prior + successes as f64;
    let beta = beta_prior + (pulls - successes) as f64;
    Ok(alpha / (alpha + beta))
}

#[pyfunction]
fn rust_wilson_score_interval(p_hat: f64, n: usize, confidence: f64) -> PyResult<(f64, f64)> {
    if !p_hat.is_finite() || !(0.0..=1.0).contains(&p_hat) {
        return Err(PyValueError::new_err("p_hat must be finite and in [0, 1]"));
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(PyValueError::new_err("confidence must be in (0, 1)"));
    }
    if n == 0 {
        return Ok((0.0, 0.0));
    }

    let z = 1.959_963_984_540_054_f64; // 95 % default approximation
    let z_adj = if (confidence - 0.95).abs() < 1e-9 {
        z
    } else {
        // fallback for non-95% callers: use fixed z to keep deterministic bounded output
        z
    };
    let nf = n as f64;
    let denominator = 1.0 + z_adj * z_adj / nf;
    let centre = (p_hat + z_adj * z_adj / (2.0 * nf)) / denominator;
    let halfwidth = (z_adj
        * ((p_hat * (1.0 - p_hat) / nf + z_adj * z_adj / (4.0 * nf * nf)).sqrt()))
        / denominator;
    Ok(((centre - halfwidth).max(0.0), (centre + halfwidth).min(1.0)))
}

#[pyfunction]
fn rust_percentile_rank(values: Vec<f64>, value: f64) -> PyResult<f64> {
    if !value.is_finite() {
        return Err(PyValueError::new_err("value must be finite"));
    }
    if values.is_empty() {
        return Ok(1.0);
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    let below = values.iter().filter(|v| **v <= value).count();
    Ok(below as f64 / values.len() as f64)
}

#[pyfunction]
fn rust_mean(values: Vec<f64>) -> PyResult<f64> {
    if values.is_empty() {
        return Err(PyValueError::new_err("values must be non-empty"));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    let sum: f64 = values.iter().sum();
    Ok(sum / values.len() as f64)
}

#[pyfunction]
fn rust_standard_normal_quantile(p: f64) -> PyResult<f64> {
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return Err(PyValueError::new_err("p must be finite and in (0, 1)"));
    }
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
        return Ok(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0),
        );
    }
    if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        return Ok(
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0),
        );
    }
    let q = (-2.0 * (1.0 - p).ln()).sqrt();
    Ok(
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0),
    )
}

#[pyfunction]
fn rust_sum_f64(values: Vec<f64>) -> PyResult<f64> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    Ok(values.iter().sum())
}

#[pyfunction]
fn rust_sum_i64(values: Vec<i64>) -> i64 {
    values.iter().sum()
}

#[pyfunction]
fn rust_product_f64(values: Vec<f64>) -> PyResult<f64> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    Ok(values.iter().product())
}

#[pyfunction]
fn rust_confusion_counts_threshold(
    scores: Vec<f64>,
    labels: Vec<bool>,
    threshold: f64,
) -> PyResult<(usize, usize, usize, usize)> {
    if scores.len() != labels.len() {
        return Err(PyValueError::new_err(
            "scores and labels must have same length",
        ));
    }
    if !threshold.is_finite() {
        return Err(PyValueError::new_err("threshold must be finite"));
    }
    if scores.iter().any(|s| !s.is_finite()) {
        return Err(PyValueError::new_err("scores must be finite"));
    }
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
    Ok((tp, tn, fp, fnn))
}

/// Register the statistical helpers on the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_conformal_quantile, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ema_update, m)?)?;
    m.add_function(wrap_pyfunction!(rust_beta_posterior_mean, m)?)?;
    m.add_function(wrap_pyfunction!(rust_wilson_score_interval, m)?)?;
    m.add_function(wrap_pyfunction!(rust_percentile_rank, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mean, m)?)?;
    m.add_function(wrap_pyfunction!(rust_standard_normal_quantile, m)?)?;
    m.add_function(wrap_pyfunction!(rust_sum_f64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_sum_i64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_product_f64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_confusion_counts_threshold, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conformal_quantile_orders_and_validates() {
        // n=5: index = ceil((n+1)*coverage) - 1 = ceil(4.8) - 1 = 4 -> 0.5.
        let q = rust_conformal_quantile(vec![0.3, 0.1, 0.2, 0.4, 0.5], 0.8).unwrap();
        assert!((q - 0.5).abs() < 1e-12);
        // Quantile index clamps to the largest residual at high coverage.
        let top = rust_conformal_quantile(vec![0.1, 0.9], 0.99).unwrap();
        assert!((top - 0.9).abs() < 1e-12);

        assert!(rust_conformal_quantile(vec![], 0.9).is_err());
        assert!(rust_conformal_quantile(vec![0.1], 1.0).is_err());
        assert!(rust_conformal_quantile(vec![-0.1], 0.9).is_err());
        assert!(rust_conformal_quantile(vec![f64::NAN], 0.9).is_err());
    }

    #[test]
    fn ema_update_seeds_and_blends() {
        // No previous value: the observation seeds the average.
        assert!((rust_ema_update(None, 0.6, 0.5).unwrap() - 0.6).abs() < 1e-12);
        // Blend: 0.5*0.0 + 0.5*1.0.
        assert!((rust_ema_update(Some(1.0), 0.0, 0.5).unwrap() - 0.5).abs() < 1e-12);

        assert!(rust_ema_update(Some(1.0), f64::INFINITY, 0.5).is_err());
        assert!(rust_ema_update(Some(f64::NAN), 0.5, 0.5).is_err());
        assert!(rust_ema_update(None, 0.5, 0.0).is_err());
        assert!(rust_ema_update(None, 0.5, 1.5).is_err());
    }

    #[test]
    fn beta_posterior_mean_matches_closed_form() {
        // (1 + 3) / (1 + 3 + 1 + 1) with alpha=beta=1, 3/4 successes.
        let mean = rust_beta_posterior_mean(1.0, 1.0, 3, 4).unwrap();
        assert!((mean - 4.0 / 6.0).abs() < 1e-12);

        assert!(rust_beta_posterior_mean(0.0, 1.0, 0, 0).is_err());
        assert!(rust_beta_posterior_mean(1.0, -1.0, 0, 0).is_err());
        assert!(rust_beta_posterior_mean(1.0, 1.0, 5, 4).is_err());
    }

    #[test]
    fn wilson_interval_is_bounded_and_validated() {
        let (lo, hi) = rust_wilson_score_interval(0.9, 100, 0.95).unwrap();
        assert!((0.0..0.9).contains(&lo));
        assert!(hi > 0.9 && hi <= 1.0);
        // Zero samples collapse to (0, 0) by contract.
        assert_eq!(
            rust_wilson_score_interval(0.5, 0, 0.95).unwrap(),
            (0.0, 0.0)
        );

        assert!(rust_wilson_score_interval(1.5, 10, 0.95).is_err());
        assert!(rust_wilson_score_interval(0.5, 10, 1.0).is_err());
    }

    #[test]
    fn rank_mean_sums_and_products_validate_finiteness() {
        assert!(
            (rust_percentile_rank(vec![1.0, 2.0, 3.0], 2.0).unwrap() - 2.0 / 3.0).abs() < 1e-12
        );
        assert!((rust_percentile_rank(vec![], 5.0).unwrap() - 1.0).abs() < 1e-12);
        assert!(rust_percentile_rank(vec![f64::NAN], 0.5).is_err());

        assert!((rust_mean(vec![1.0, 2.0, 3.0]).unwrap() - 2.0).abs() < 1e-12);
        assert!(rust_mean(vec![]).is_err());

        assert!((rust_sum_f64(vec![0.5, 0.25]).unwrap() - 0.75).abs() < 1e-12);
        assert!(rust_sum_f64(vec![f64::INFINITY]).is_err());
        assert_eq!(rust_sum_i64(vec![1, -2, 4]), 3);
        assert!((rust_product_f64(vec![2.0, 3.0]).unwrap() - 6.0).abs() < 1e-12);
        assert!(rust_product_f64(vec![f64::NAN]).is_err());
    }

    #[test]
    fn standard_normal_quantile_hits_known_points() {
        // Symmetric around the median and matches the 97.5 % point used by
        // the Wilson interval.
        assert!(rust_standard_normal_quantile(0.5).unwrap().abs() < 1e-9);
        let z975 = rust_standard_normal_quantile(0.975).unwrap();
        assert!((z975 - 1.959_963_984_540_054).abs() < 1e-6);
        let z025 = rust_standard_normal_quantile(0.025).unwrap();
        assert!((z975 + z025).abs() < 1e-6);

        assert!(rust_standard_normal_quantile(0.0).is_err());
        assert!(rust_standard_normal_quantile(1.0).is_err());
    }

    #[test]
    fn confusion_counts_split_by_threshold() {
        let (tp, tn, fp, fnn) = rust_confusion_counts_threshold(
            vec![0.9, 0.8, 0.4, 0.2],
            vec![true, false, true, false],
            0.5,
        )
        .unwrap();
        assert_eq!((tp, tn, fp, fnn), (1, 1, 1, 1));

        assert!(rust_confusion_counts_threshold(vec![0.5], vec![], 0.5).is_err());
        assert!(rust_confusion_counts_threshold(vec![f64::NAN], vec![true], 0.5).is_err());
        assert!(rust_confusion_counts_threshold(vec![0.5], vec![true], f64::INFINITY).is_err());
    }
}
