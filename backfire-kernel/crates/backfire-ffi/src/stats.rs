// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — ffi::stats
//! Statistical helper bindings: argument validation at the Python boundary,
//! computation delegated to [`backfire_core::stats`] (the crate-wide
//! pattern — pure maths lives in core, PyO3 owns the contract checks).

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
    Ok(backfire_core::stats::conformal_quantile(
        &residuals, coverage,
    ))
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
    }
    Ok(backfire_core::stats::ema_update(previous, value, alpha))
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
    Ok(backfire_core::stats::beta_posterior_mean(
        alpha_prior,
        beta_prior,
        successes,
        pulls,
    ))
}

#[pyfunction]
fn rust_wilson_score_interval(p_hat: f64, n: usize, confidence: f64) -> PyResult<(f64, f64)> {
    if !p_hat.is_finite() || !(0.0..=1.0).contains(&p_hat) {
        return Err(PyValueError::new_err("p_hat must be finite and in [0, 1]"));
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(PyValueError::new_err("confidence must be in (0, 1)"));
    }
    Ok(backfire_core::stats::wilson_score_interval(p_hat, n))
}

#[pyfunction]
fn rust_percentile_rank(values: Vec<f64>, value: f64) -> PyResult<f64> {
    if !value.is_finite() {
        return Err(PyValueError::new_err("value must be finite"));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    Ok(backfire_core::stats::percentile_rank(&values, value))
}

#[pyfunction]
fn rust_mean(values: Vec<f64>) -> PyResult<f64> {
    if values.is_empty() {
        return Err(PyValueError::new_err("values must be non-empty"));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    Ok(backfire_core::stats::mean(&values))
}

#[pyfunction]
fn rust_standard_normal_quantile(p: f64) -> PyResult<f64> {
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return Err(PyValueError::new_err("p must be finite and in (0, 1)"));
    }
    Ok(backfire_core::stats::standard_normal_quantile(p))
}

#[pyfunction]
fn rust_sum_f64(values: Vec<f64>) -> PyResult<f64> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    Ok(backfire_core::stats::sum_f64(&values))
}

#[pyfunction]
fn rust_sum_i64(values: Vec<i64>) -> i64 {
    backfire_core::stats::sum_i64(&values)
}

#[pyfunction]
fn rust_product_f64(values: Vec<f64>) -> PyResult<f64> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    Ok(backfire_core::stats::product_f64(&values))
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
    Ok(backfire_core::stats::confusion_counts_threshold(
        &scores, &labels, threshold,
    ))
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
    fn conformal_quantile_validates_and_delegates() {
        // Happy path delegates to core: n=5, coverage 0.8 -> 0.5.
        let q = rust_conformal_quantile(vec![0.3, 0.1, 0.2, 0.4, 0.5], 0.8).unwrap();
        assert!((q - 0.5).abs() < 1e-12);

        assert!(rust_conformal_quantile(vec![], 0.9).is_err());
        assert!(rust_conformal_quantile(vec![0.1], 1.0).is_err());
        assert!(rust_conformal_quantile(vec![-0.1], 0.9).is_err());
        assert!(rust_conformal_quantile(vec![f64::NAN], 0.9).is_err());
    }

    #[test]
    fn ema_update_validates_and_delegates() {
        assert!((rust_ema_update(Some(1.0), 0.0, 0.5).unwrap() - 0.5).abs() < 1e-12);

        assert!(rust_ema_update(Some(1.0), f64::INFINITY, 0.5).is_err());
        assert!(rust_ema_update(Some(f64::NAN), 0.5, 0.5).is_err());
        assert!(rust_ema_update(None, 0.5, 0.0).is_err());
        assert!(rust_ema_update(None, 0.5, 1.5).is_err());
    }

    #[test]
    fn beta_posterior_mean_validates_and_delegates() {
        let mean = rust_beta_posterior_mean(1.0, 1.0, 3, 4).unwrap();
        assert!((mean - 4.0 / 6.0).abs() < 1e-12);

        assert!(rust_beta_posterior_mean(0.0, 1.0, 0, 0).is_err());
        assert!(rust_beta_posterior_mean(1.0, -1.0, 0, 0).is_err());
        assert!(rust_beta_posterior_mean(1.0, 1.0, 5, 4).is_err());
    }

    #[test]
    fn wilson_interval_validates_and_delegates() {
        let (lo, hi) = rust_wilson_score_interval(0.9, 100, 0.95).unwrap();
        assert!((0.0..0.9).contains(&lo));
        assert!(hi > 0.9 && hi <= 1.0);
        assert_eq!(
            rust_wilson_score_interval(0.5, 0, 0.95).unwrap(),
            (0.0, 0.0)
        );

        assert!(rust_wilson_score_interval(1.5, 10, 0.95).is_err());
        assert!(rust_wilson_score_interval(0.5, 10, 1.0).is_err());
    }

    #[test]
    fn rank_mean_sums_and_products_validate_and_delegate() {
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
    fn standard_normal_quantile_validates_and_delegates() {
        let z975 = rust_standard_normal_quantile(0.975).unwrap();
        assert!((z975 - 1.959_963_984_540_054).abs() < 1e-6);

        assert!(rust_standard_normal_quantile(0.0).is_err());
        assert!(rust_standard_normal_quantile(1.0).is_err());
    }

    #[test]
    fn confusion_counts_validate_and_delegate() {
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
