// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — compute::nli_math
//! NLI probability math: softmax, divergence/confidence mapping, and
//! chunk-score aggregation/attribution reductions.
//!
//! Mirrors `_softmax_np()` / `_probs_to_divergence()` /
//! `_probs_to_confidence()` and the aggregation semantics of
//! `NLIScorer._score_chunked_with_counts()` from `nli.py`.

/// Row-wise softmax for a 2D array (flattened as rows × cols).
///
/// Mirrors `_softmax_np()` from `nli.py`.
pub fn softmax(logits: &[f64], cols: usize) -> Vec<f64> {
    if cols == 0 || logits.is_empty() {
        return vec![];
    }
    let rows = logits.len() / cols;
    let mut result = vec![0.0; logits.len()];

    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        let row = &logits[start..end];

        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for (i, &v) in row.iter().enumerate() {
            let e = (v - max_val).exp();
            result[start + i] = e;
            sum += e;
        }
        if sum > 0.0 {
            for i in start..end {
                result[i] /= sum;
            }
        }
    }

    result
}

/// Convert NLI softmax probabilities to divergence scores.
///
/// 2-class: divergence = 1 - P(supported).
/// 3-class: divergence = P(contradiction) + 0.5 * P(neutral).
/// Mirrors `_probs_to_divergence()` from `nli.py`.
pub fn probs_to_divergence(
    probs: &[f64],
    cols: usize,
    contradiction_idx: usize,
    neutral_idx: usize,
) -> Vec<f64> {
    if cols == 0 || probs.is_empty() {
        return vec![];
    }
    let rows = probs.len() / cols;
    let mut result = Vec::with_capacity(rows);

    for r in 0..rows {
        let start = r * cols;
        let row = &probs[start..start + cols];

        if cols == 2 {
            // 2-class: 1 - P(supported), where supported is class 1
            result.push(1.0 - row.get(1).copied().unwrap_or(0.5));
        } else {
            // 3-class: P(contradiction) + 0.5 * P(neutral)
            let p_contra = row.get(contradiction_idx).copied().unwrap_or(0.0);
            let p_neutral = row.get(neutral_idx).copied().unwrap_or(0.0);
            result.push(p_contra + 0.5 * p_neutral);
        }
    }

    result
}

/// Compute confidence from softmax probabilities (1 - normalised entropy).
///
/// Mirrors `_probs_to_confidence()` from `nli.py`.
pub fn probs_to_confidence(probs: &[f64], cols: usize) -> Vec<f64> {
    if cols == 0 || probs.is_empty() {
        return vec![];
    }
    let rows = probs.len() / cols;
    let log_k = (cols as f64).ln();
    let mut result = Vec::with_capacity(rows);

    for r in 0..rows {
        let start = r * cols;
        let row = &probs[start..start + cols];

        let entropy: f64 = row
            .iter()
            .filter(|&&p| p > 1e-12)
            .map(|&p| -p * p.ln())
            .sum();

        let normalised = if log_k > 0.0 { entropy / log_k } else { 0.0 };

        result.push((1.0 - normalised).clamp(0.0, 1.0));
    }

    result
}

/// Aggregate chunk-level score matrix into per-hypothesis and global scores.
///
/// `flat_scores` is row-major with shape `(n_prem, n_hyp)` where index is
/// `p * n_hyp + h`.
///
/// Mirrors aggregation semantics in `NLIScorer._score_chunked_with_counts()`.
pub fn aggregate_chunk_scores(
    flat_scores: &[f64],
    n_prem: usize,
    n_hyp: usize,
    inner_agg: &str,
    outer_agg: &str,
) -> (f64, Vec<f64>) {
    if n_prem == 0 || n_hyp == 0 || flat_scores.is_empty() {
        return (0.5, vec![0.5]);
    }

    let mut per_hyp: Vec<f64> = Vec::with_capacity(n_hyp);
    for h_idx in 0..n_hyp {
        let mut scores_h: Vec<f64> = Vec::with_capacity(n_prem);
        for p in 0..n_prem {
            let idx = p * n_hyp + h_idx;
            if idx < flat_scores.len() {
                scores_h.push(flat_scores[idx]);
            }
        }
        if scores_h.is_empty() {
            per_hyp.push(0.5);
            continue;
        }
        let value = if inner_agg == "min" {
            scores_h.iter().copied().fold(f64::INFINITY, f64::min)
        } else if inner_agg == "mean" {
            scores_h.iter().sum::<f64>() / scores_h.len() as f64
        } else {
            scores_h.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        };
        per_hyp.push(value);
    }

    if per_hyp.is_empty() {
        return (0.5, vec![0.5]);
    }

    let agg = if outer_agg == "max" {
        per_hyp.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    } else if outer_agg == "trimmed_mean" {
        let mut sorted_scores = per_hyp.clone();
        sorted_scores.sort_by(|a, b| a.total_cmp(b));
        let keep = ((sorted_scores.len() as f64) * 0.75).floor() as usize;
        let keep = keep.max(1);
        sorted_scores.iter().take(keep).sum::<f64>() / keep as f64
    } else {
        per_hyp.iter().sum::<f64>() / per_hyp.len() as f64
    };
    (agg, per_hyp)
}

/// Aggregate chunk-level scores with confidence-weighted outer reduction.
///
/// `flat_scores` and `flat_confidences` are row-major with shape
/// `(n_prem, n_hyp)`, indexed by `p * n_hyp + h`.
///
/// Mirrors `NLIScorer.score_chunked_confidence_weighted()`.
pub fn aggregate_chunk_scores_confidence_weighted(
    flat_scores: &[f64],
    flat_confidences: &[f64],
    n_prem: usize,
    n_hyp: usize,
    inner_agg: &str,
) -> (f64, Vec<f64>) {
    if n_prem == 0 || n_hyp == 0 || flat_scores.is_empty() || flat_confidences.is_empty() {
        return (0.5, vec![0.5]);
    }

    let mut per_hyp: Vec<f64> = Vec::with_capacity(n_hyp);
    let mut per_hyp_conf: Vec<f64> = Vec::with_capacity(n_hyp);

    for h_idx in 0..n_hyp {
        let mut scores_h: Vec<f64> = Vec::with_capacity(n_prem);
        let mut confs_h: Vec<f64> = Vec::with_capacity(n_prem);
        for p in 0..n_prem {
            let idx = p * n_hyp + h_idx;
            if idx < flat_scores.len() && idx < flat_confidences.len() {
                scores_h.push(flat_scores[idx]);
                confs_h.push(flat_confidences[idx]);
            }
        }

        if scores_h.is_empty() || confs_h.is_empty() {
            per_hyp.push(0.5);
            per_hyp_conf.push(0.0);
            continue;
        }

        let div = if inner_agg == "min" {
            scores_h.iter().copied().fold(f64::INFINITY, f64::min)
        } else if inner_agg == "mean" {
            scores_h.iter().sum::<f64>() / scores_h.len() as f64
        } else {
            scores_h.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        };
        let avg_conf = confs_h.iter().sum::<f64>() / confs_h.len() as f64;
        per_hyp.push(div);
        per_hyp_conf.push(avg_conf);
    }

    if per_hyp.is_empty() {
        return (0.5, vec![0.5]);
    }

    let total_weight: f64 = per_hyp_conf.iter().sum();
    let agg = if total_weight > 1e-9 {
        per_hyp
            .iter()
            .zip(per_hyp_conf.iter())
            .map(|(d, c)| d * c)
            .sum::<f64>()
            / total_weight
    } else {
        per_hyp.iter().sum::<f64>() / per_hyp.len() as f64
    };

    (agg, per_hyp)
}

/// Compute claim coverage from per-claim divergences and a support threshold.
///
/// Returns `(coverage, supported_count)` where a claim is considered supported
/// when `divergence < support_threshold`.
pub fn coverage_from_divergences(divergences: &[f64], support_threshold: f64) -> (f64, usize) {
    if divergences.is_empty() {
        return (0.0, 0);
    }
    let supported = divergences
        .iter()
        .filter(|d| **d < support_threshold)
        .count();
    let coverage = supported as f64 / divergences.len() as f64;
    (coverage, supported)
}

/// Reduce a flat claim×source divergence matrix into per-claim best attributions.
///
/// The input is row-major by claim:
/// `flat_divs[claim_idx * n_src + src_idx]`.
///
/// Returns `(per_claim_best_divergence, per_claim_best_source_index)`.
pub fn reduce_claim_attribution(
    flat_divs: &[f64],
    n_claims: usize,
    n_src: usize,
) -> (Vec<f64>, Vec<usize>) {
    let mut per_claim_divs = Vec::with_capacity(n_claims);
    let mut best_source_indices = Vec::with_capacity(n_claims);

    for claim_idx in 0..n_claims {
        let row_start = claim_idx * n_src;
        let row_end = row_start + n_src;
        let row = &flat_divs[row_start..row_end];
        let mut best_idx = 0usize;
        let mut best_div = row[0];
        for (src_idx, div) in row.iter().enumerate().skip(1) {
            if *div < best_div {
                best_div = *div;
                best_idx = src_idx;
            }
        }
        per_claim_divs.push(best_div);
        best_source_indices.push(best_idx);
    }

    (per_claim_divs, best_source_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let result = softmax(&logits, 3);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_softmax_multi_row() {
        let logits = vec![0.0, 0.0, 1.0, 1.0];
        let result = softmax(&logits, 2);
        assert_eq!(result.len(), 4);
        assert!((result[0] - result[1]).abs() < 1e-9); // row 1: equal
        assert!((result[2] - result[3]).abs() < 1e-9); // row 2: equal
    }

    #[test]
    fn test_probs_to_divergence_2class() {
        let probs = vec![0.3, 0.7]; // P(not_supported)=0.3, P(supported)=0.7
        let divs = probs_to_divergence(&probs, 2, 2, 1);
        assert!((divs[0] - 0.3).abs() < 1e-9); // 1 - 0.7
    }

    #[test]
    fn test_probs_to_divergence_3class() {
        let probs = vec![0.2, 0.3, 0.5]; // entail=0.2, neutral=0.3, contra=0.5
        let divs = probs_to_divergence(&probs, 3, 2, 1);
        // 0.5 + 0.5 * 0.3 = 0.65
        assert!((divs[0] - 0.65).abs() < 1e-9);
    }

    #[test]
    fn test_probs_to_confidence_uniform() {
        // Uniform distribution → max entropy → confidence ≈ 0
        let probs = vec![0.5, 0.5];
        let confs = probs_to_confidence(&probs, 2);
        assert!(confs[0] < 0.01);
    }

    #[test]
    fn test_probs_to_confidence_certain() {
        // Near-certain → confidence ≈ 1
        let probs = vec![0.001, 0.999];
        let confs = probs_to_confidence(&probs, 2);
        assert!(confs[0] > 0.95);
    }

    #[test]
    fn test_aggregate_chunk_scores_inner_min_outer_max() {
        // n_prem=2, n_hyp=3 (row-major)
        let flat = vec![
            0.2, 0.8, 0.4, // prem 0
            0.6, 0.3, 0.9, // prem 1
        ];
        let (agg, per_hyp) = aggregate_chunk_scores(&flat, 2, 3, "min", "max");
        assert_eq!(per_hyp.len(), 3);
        assert!((per_hyp[0] - 0.2).abs() < 1e-12);
        assert!((per_hyp[1] - 0.3).abs() < 1e-12);
        assert!((per_hyp[2] - 0.4).abs() < 1e-12);
        assert!((agg - 0.4).abs() < 1e-12);
    }

    #[test]
    fn test_aggregate_chunk_scores_trimmed_mean() {
        let flat = vec![
            0.1, 0.9, 0.2, 0.8, // prem 0
            0.2, 0.8, 0.3, 0.7, // prem 1
        ];
        let (agg_mean, _per_hyp) = aggregate_chunk_scores(&flat, 2, 4, "min", "mean");
        let (agg_trimmed, _per_hyp_t) = aggregate_chunk_scores(&flat, 2, 4, "min", "trimmed_mean");
        assert!(agg_trimmed <= agg_mean);
    }

    #[test]
    fn test_aggregate_chunk_scores_confidence_weighted() {
        // n_prem=2, n_hyp=2
        let flat_scores = vec![0.2, 0.8, 0.4, 0.6];
        let flat_conf = vec![0.9, 0.1, 0.7, 0.3];
        let (agg, per_hyp) =
            aggregate_chunk_scores_confidence_weighted(&flat_scores, &flat_conf, 2, 2, "max");
        assert_eq!(per_hyp.len(), 2);
        assert!((per_hyp[0] - 0.4).abs() < 1e-12);
        assert!((per_hyp[1] - 0.8).abs() < 1e-12);
        assert!((0.4..=0.8).contains(&agg));
    }

    #[test]
    fn test_coverage_from_divergences() {
        let divs = vec![0.1, 0.3, 0.8, 0.2];
        let (coverage, supported) = coverage_from_divergences(&divs, 0.5);
        assert_eq!(supported, 3);
        assert!((coverage - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_reduce_claim_attribution() {
        // 3 claims × 4 source sentences (row-major by claim)
        let flat = vec![
            0.4, 0.2, 0.7, 0.3, // claim 0 -> best src 1 (0.2)
            0.8, 0.6, 0.5, 0.9, // claim 1 -> best src 2 (0.5)
            0.1, 0.2, 0.3, 0.4, // claim 2 -> best src 0 (0.1)
        ];
        let (divs, idxs) = reduce_claim_attribution(&flat, 3, 4);
        assert_eq!(divs, vec![0.2, 0.5, 0.1]);
        assert_eq!(idxs, vec![1, 2, 0]);
    }
}
