// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — compute::temporal
//! Temporal-claim staleness scoring.
//!
//! Mirrors `score_temporal_freshness()` from `temporal_freshness.py`.

use once_cell::sync::Lazy;
use regex::Regex;

// Mirrors Python _POSITION_PATTERN from temporal_freshness.py
static POSITION_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)(?:the\s+)?(?:CEO|CTO|CFO|COO|president|prime\s+minister|chairman|director|head|leader|secretary|minister|governor|mayor)\s+(?:of\s+)?(?:\S+(?:\s+\S+){0,10})\s+(?:is|was)\b"
    ).unwrap()
});

// Mirrors Python _STAT_PATTERN from temporal_freshness.py
static STATISTIC_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)(?:population|GDP|revenue|market\s+cap|stock\s+price|unemployment|inflation|interest\s+rate|exchange\s+rate|growth\s+rate)(?:\s+\w+){0,5}\s+[\d,.]+\s*(?:million|billion|trillion|%|percent)?"
    ).unwrap()
});

// Mirrors Python _CURRENT_PATTERN from temporal_freshness.py
static CURRENT_REF_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:currently|as of|right now|at present|today|this year|in \d{4})").unwrap()
});

// Mirrors Python _RECORD_PATTERN from temporal_freshness.py
static RECORD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)(?:world\s+record|fastest|tallest|largest|smallest|highest|lowest|most\s+\w+|best\s+selling|top\s+\w+|#1|number\s+one)"
    ).unwrap()
});

/// A temporal claim with staleness risk.
#[derive(Debug, Clone)]
pub struct TemporalClaim {
    pub text: String,
    pub claim_type: String,
    pub staleness_risk: f64,
}

/// Score temporal freshness of claims in text.
///
/// Returns (claims, overall_staleness_risk, has_temporal_claims).
/// Mirrors `score_temporal_freshness()` from `temporal_freshness.py`.
pub fn score_temporal_freshness(text: &str) -> (Vec<TemporalClaim>, f64, bool) {
    let mut claims = Vec::new();

    // age_factor = 0.5 (unknown source = moderate risk, same as Python)
    let age_factor: f64 = 0.5;

    for m in POSITION_RE.find_iter(text) {
        let risk = (0.6 + 0.4 * age_factor).min(1.0);
        claims.push(TemporalClaim {
            text: m.as_str().trim().to_string(),
            claim_type: "position".into(),
            staleness_risk: risk,
        });
    }

    for m in STATISTIC_RE.find_iter(text) {
        let risk = (0.4 + 0.4 * age_factor).min(1.0);
        claims.push(TemporalClaim {
            text: m.as_str().trim().to_string(),
            claim_type: "statistic".into(),
            staleness_risk: risk,
        });
    }

    for m in CURRENT_REF_RE.find_iter(text) {
        // Context extraction: 30 chars before, 50 chars after
        let start = m.start().saturating_sub(30);
        let end = (m.end() + 50).min(text.len());
        let ctx = text[start..end].trim().to_string();
        let risk = (0.5 + 0.5 * age_factor).min(1.0);
        claims.push(TemporalClaim {
            text: ctx,
            claim_type: "current_reference".into(),
            staleness_risk: risk,
        });
    }

    for m in RECORD_RE.find_iter(text) {
        // Context extraction: 20 chars before, 40 chars after
        let start = m.start().saturating_sub(20);
        let end = (m.end() + 40).min(text.len());
        let ctx = text[start..end].trim().to_string();
        let risk = (0.3 + 0.3 * age_factor).min(1.0);
        claims.push(TemporalClaim {
            text: ctx,
            claim_type: "record".into(),
            staleness_risk: risk,
        });
    }

    let has_temporal = !claims.is_empty();
    // Python uses max(), not average
    let overall = claims
        .iter()
        .map(|c| c.staleness_risk)
        .fold(0.0_f64, f64::max);

    (claims, overall, has_temporal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_no_claims() {
        let (claims, risk, has) = score_temporal_freshness("The sky is blue.");
        assert!(claims.is_empty());
        assert_eq!(risk, 0.0);
        assert!(!has);
    }

    #[test]
    fn test_temporal_position() {
        let (claims, risk, has) =
            score_temporal_freshness("The current president of France is Macron.");
        assert!(has);
        assert!(risk > 0.0);
        assert!(claims.iter().any(|c| c.claim_type == "position"));
    }

    #[test]
    fn test_temporal_statistic() {
        let (claims, _, has) = score_temporal_freshness("GDP of Germany was 4.2 trillion.");
        assert!(has);
        assert!(claims.iter().any(|c| c.claim_type == "statistic"));
    }
}
