// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — compute::numeric
//! Numeric consistency verification.
//!
//! Mirrors `verify_numeric()` from `core/verification/numeric_verifier.py`.

use once_cell::sync::Lazy;
use regex::Regex;

static PERCENT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)(?:grew|increased|decreased|dropped|rose|fell|declined|changed|gained|lost)\s+(?:by\s+)?(\d{1,10}(?:\.\d{1,10})?)\s*%.{0,80}?\b(?:from|of)\s+\$?([\d,]{1,20}(?:\.\d{1,10})?)\s*(?:million|billion|thousand|[MBKmk])?.{0,80}?\bto\s+\$?([\d,]{1,20}(?:\.\d{1,10})?)\s*(?:million|billion|thousand|[MBKmk])?"
    ).unwrap()
});

static DATE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b((?:1[0-9]|20)\d{2})\b").unwrap());

static PROB_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(-?\d{1,10}(?:\.\d{1,10})?)\s*(?:%|percent)\s+(?:probability|chance|likelihood|confidence)").unwrap()
});

static BORN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)born\s+(?:in\s+)?(\d{4})").unwrap());
static DIED_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)died\s+(?:in\s+)?(\d{4})").unwrap());
static FOUNDED_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)founded\s+(?:in\s+)?(\d{4})").unwrap());
static TOTAL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)total\s+(?:of\s+)?(\d+(?:,\d+)*(?:\.\d+)?)").unwrap());

static EARTH_POP_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)earth.*?population.*?(\d+(?:\.\d+)?)\s*(billion|million)").unwrap()
});

static NUMBER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|trillion|thousand|percent|%|km|mi|kg|lb|m|ft)?\b").unwrap()
});

static SPEED_LIGHT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)speed\s+of\s+light.*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(km/s|m/s)").unwrap()
});

fn parse_number(s: &str) -> f64 {
    s.replace(',', "").parse::<f64>().unwrap_or(0.0)
}

/// A numeric issue found during verification.
#[derive(Debug, Clone)]
pub struct NumericIssue {
    pub issue_type: String,
    pub description: String,
    pub severity: String,
    pub context: String,
}

/// Verify numeric consistency within text.
///
/// Returns (claims_found, issues, valid).
/// Mirrors `verify_numeric()` from `core/verification/numeric_verifier.py`.
pub fn verify_numeric(text: &str, current_year: i32) -> (usize, Vec<NumericIssue>, bool) {
    let mut issues = Vec::new();
    let mut claims_found = 0usize;

    // 1. Percentage arithmetic
    for cap in PERCENT_RE.captures_iter(text) {
        claims_found += 1;
        let pct = parse_number(&cap[1]);
        let val_from = parse_number(&cap[2]);
        let val_to = parse_number(&cap[3]);
        if val_from > 0.0 {
            let actual_pct = ((val_to - val_from).abs() / val_from) * 100.0;
            if (actual_pct - pct).abs() > 1.0 {
                issues.push(NumericIssue {
                    issue_type: "arithmetic".into(),
                    description: format!(
                        "Claimed {pct}% change from {val_from} to {val_to}, but actual change is {actual_pct:.1}%"
                    ),
                    severity: "error".into(),
                    context: cap[0].to_string(),
                });
            }
        }
    }

    // 2. Date logic
    let dates: Vec<i32> = DATE_RE
        .captures_iter(text)
        .filter_map(|c| c[1].parse::<i32>().ok())
        .collect();
    claims_found += dates.len();

    for &d in &dates {
        if d > current_year + 5 {
            issues.push(NumericIssue {
                issue_type: "date_logic".into(),
                description: format!("Year {d} is in the far future (current: {current_year})"),
                severity: "warning".into(),
                context: d.to_string(),
            });
        }
    }

    // Birth/death ordering
    let born: Vec<i32> = BORN_RE
        .captures_iter(text)
        .filter_map(|c| c[1].parse().ok())
        .collect();
    let died: Vec<i32> = DIED_RE
        .captures_iter(text)
        .filter_map(|c| c[1].parse().ok())
        .collect();
    for &b in &born {
        for &d in &died {
            if d < b {
                issues.push(NumericIssue {
                    issue_type: "date_logic".into(),
                    description: format!("Death year {d} is before birth year {b}"),
                    severity: "error".into(),
                    context: format!("born {b}, died {d}"),
                });
            }
        }
    }

    // Founded in future
    for cap in FOUNDED_RE.captures_iter(text) {
        if let Ok(f) = cap[1].parse::<i32>() {
            if f > current_year {
                issues.push(NumericIssue {
                    issue_type: "date_logic".into(),
                    description: format!("Founded in {f} is in the future"),
                    severity: "error".into(),
                    context: format!("founded {f}"),
                });
            }
        }
    }

    // 3. Probability bounds
    for cap in PROB_RE.captures_iter(text) {
        claims_found += 1;
        let prob = parse_number(&cap[1]);
        if prob > 100.0 {
            issues.push(NumericIssue {
                issue_type: "probability".into(),
                description: format!("Probability {prob}% exceeds 100%"),
                severity: "error".into(),
                context: cap[0].to_string(),
            });
        } else if prob < 0.0 {
            issues.push(NumericIssue {
                issue_type: "probability".into(),
                description: format!("Negative probability {prob}%"),
                severity: "error".into(),
                context: cap[0].to_string(),
            });
        }
    }

    // 4. Magnitude checks
    if let Some(cap) = EARTH_POP_RE.captures(text) {
        let mut val = parse_number(&cap[1]);
        if cap[2].to_lowercase() == "million" {
            val /= 1000.0;
        }
        if !(6.0..=12.0).contains(&val) {
            issues.push(NumericIssue {
                issue_type: "magnitude".into(),
                description: format!(
                    "earth_population: {val} {} outside expected range [6-12] billion",
                    &cap[2]
                ),
                severity: "warning".into(),
                context: cap[0].to_string(),
            });
        }
    }

    if let Some(cap) = SPEED_LIGHT_RE.captures(text) {
        let val = parse_number(&cap[1]);
        if !(200_000.0..=400_000.0).contains(&val) {
            issues.push(NumericIssue {
                issue_type: "magnitude".into(),
                description: format!(
                    "speed_of_light_km: {val} {} outside expected range [200000-400000] km/s",
                    &cap[2]
                ),
                severity: "warning".into(),
                context: cap[0].to_string(),
            });
        }
    }

    // 5. Internal consistency
    let totals: Vec<f64> = TOTAL_RE
        .captures_iter(text)
        .map(|c| parse_number(&c[1]))
        .collect();
    if totals.len() >= 2 {
        for i in 1..totals.len() {
            if (totals[i] - totals[0]).abs() > 0.01 * totals[0].max(1.0) {
                issues.push(NumericIssue {
                    issue_type: "internal".into(),
                    description: format!("Inconsistent totals: {} vs {}", totals[0], totals[i]),
                    severity: "error".into(),
                    context: format!("total {} ... total {}", totals[0], totals[i]),
                });
            }
        }
    }

    // Count raw numbers
    claims_found += NUMBER_RE.find_iter(text).count();

    let valid = issues.iter().all(|i| i.severity != "error");
    (claims_found, issues, valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_clean() {
        let (_, issues, valid) = verify_numeric("The population is 8 billion.", 2026);
        assert!(valid);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_verify_bad_percentage() {
        let text = "Revenue grew 50% from 100 to 200";
        let (claims, issues, _) = verify_numeric(text, 2026);
        assert!(claims > 0);
        // 50% of 100 = 50, so 100→200 is actually 100% — should flag
        assert!(!issues.is_empty());
        assert_eq!(issues[0].issue_type, "arithmetic");
    }

    #[test]
    fn test_verify_death_before_birth() {
        let text = "Born in 1990, died in 1980";
        let (_, issues, valid) = verify_numeric(text, 2026);
        assert!(!valid);
        assert!(issues.iter().any(|i| i.issue_type == "date_logic"));
    }

    #[test]
    fn test_verify_probability_bounds() {
        let text = "There is a 150% probability of success";
        let (_, issues, valid) = verify_numeric(text, 2026);
        assert!(!valid);
        assert!(issues.iter().any(|i| i.issue_type == "probability"));
    }

    #[test]
    fn test_verify_inconsistent_totals() {
        let text = "The total of 500 items. Later, the total of 600 items.";
        let (_, issues, valid) = verify_numeric(text, 2026);
        assert!(!valid);
        assert!(issues.iter().any(|i| i.issue_type == "internal"));
    }
}
