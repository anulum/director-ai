#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Verified Scorer Benchmark

"""Measure VerifiedScorer accuracy on synthetic customer-KB scenarios.

Tests the REAL use case: customer has product docs, chatbot generates
answers, some are faithful and some hallucinate. The scorer must
distinguish them with high accuracy and confidence.

Each test case has:
- source: a paragraph from a "product KB"
- correct: a faithful response
- hallucinated: a response with subtle errors (wrong numbers, entities, negations)

Usage::

    python -m benchmarks.verified_scorer_bench
    python -m benchmarks.verified_scorer_bench --with-nli
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("VerifiedBench")


@dataclass
class TestCase:
    category: str
    source: str
    correct: str
    hallucinated: str
    error_type: str


# Synthetic customer-KB test cases covering common hallucination patterns
TEST_CASES = [
    # ── Number substitution ──
    TestCase(
        "product_spec",
        "The ProMax 3000 supports up to 500 concurrent users and provides 99.9% uptime SLA.",
        "The ProMax 3000 supports up to 500 concurrent users with a 99.9% uptime guarantee.",
        "The ProMax 3000 supports up to 5000 concurrent users with a 99.99% uptime guarantee.",
        "number_substitution",
    ),
    TestCase(
        "pricing",
        "Our Basic plan costs $29 per month. The Professional plan is $79 per month. Enterprise pricing starts at $199 per month.",
        "The Basic plan is $29/month, Professional is $79/month, and Enterprise starts at $199/month.",
        "The Basic plan is $39/month, Professional is $99/month, and Enterprise starts at $299/month.",
        "number_substitution",
    ),
    TestCase(
        "policy",
        "Customers can return products within 30 days of purchase for a full refund. After 30 days, only store credit is available.",
        "You can return items within 30 days for a full refund. After that period, store credit is provided.",
        "You can return items within 60 days for a full refund. After that period, store credit is provided.",
        "number_substitution",
    ),
    # ── Entity substitution ──
    TestCase(
        "product_spec",
        "The DataVault encryption module uses AES-256 encryption and is certified by NIST.",
        "DataVault uses AES-256 encryption with NIST certification.",
        "DataVault uses AES-128 encryption with ISO certification.",
        "entity_substitution",
    ),
    TestCase(
        "support",
        "Technical support is available Monday through Friday, 9 AM to 6 PM Eastern Time. Contact support@acme.com for assistance.",
        "You can reach technical support Monday-Friday, 9 AM to 6 PM ET at support@acme.com.",
        "You can reach technical support Monday-Friday, 9 AM to 6 PM PT at help@acme.com.",
        "entity_substitution",
    ),
    TestCase(
        "compliance",
        "Our platform is SOC 2 Type II compliant and HIPAA certified. Data is stored in AWS us-east-1 region.",
        "The platform has SOC 2 Type II compliance and HIPAA certification, with data stored in AWS us-east-1.",
        "The platform has SOC 2 Type I compliance and GDPR certification, with data stored in AWS eu-west-1.",
        "entity_substitution",
    ),
    # ── Negation flip ──
    TestCase(
        "feature",
        "The free tier does not include API access. API access requires a Professional or Enterprise subscription.",
        "API access is not available on the free tier. You need Professional or Enterprise for API access.",
        "The free tier includes API access. No additional subscription is needed.",
        "negation_flip",
    ),
    TestCase(
        "policy",
        "We do not offer phone support for Basic plan customers. Phone support is available for Professional and Enterprise plans.",
        "Phone support is not available for Basic plan customers. It is included in Professional and Enterprise plans.",
        "We offer phone support for all plan levels including Basic.",
        "negation_flip",
    ),
    TestCase(
        "limitation",
        "The system cannot process files larger than 100 MB. Files must be under 100 MB to be uploaded.",
        "There is a 100 MB file size limit. Files larger than 100 MB cannot be uploaded.",
        "The system can process files of any size. There are no file size limitations.",
        "negation_flip",
    ),
    # ── Fabrication (added facts not in source) ──
    TestCase(
        "product_spec",
        "The Analytics Dashboard provides real-time metrics including response time, error rate, and throughput.",
        "The Analytics Dashboard shows real-time metrics: response time, error rate, and throughput.",
        "The Analytics Dashboard shows real-time metrics including response time, error rate, throughput, and AI-powered anomaly detection.",
        "fabrication",
    ),
    TestCase(
        "integration",
        "We integrate with Slack and Microsoft Teams for notifications. Email notifications are also supported.",
        "Notifications are supported via Slack, Microsoft Teams, and email.",
        "Notifications are supported via Slack, Microsoft Teams, email, Discord, and WhatsApp.",
        "fabrication",
    ),
    TestCase(
        "security",
        "All data is encrypted at rest using AES-256. Data in transit is protected by TLS 1.3.",
        "Data is encrypted at rest with AES-256 and in transit with TLS 1.3.",
        "Data is encrypted at rest with AES-256, in transit with TLS 1.3, and features quantum-resistant encryption for future-proofing.",
        "fabrication",
    ),
    # ── Temporal/version errors ──
    TestCase(
        "changelog",
        "Version 3.2 was released in January 2025. It introduced the new dashboard and fixed 47 bugs.",
        "Version 3.2 came out in January 2025 with a new dashboard and 47 bug fixes.",
        "Version 3.2 was released in March 2025 with a new dashboard and 23 bug fixes.",
        "temporal_error",
    ),
    TestCase(
        "deprecation",
        "The v1 API will be deprecated on June 30, 2026. All clients must migrate to v2 by that date.",
        "The v1 API deprecation date is June 30, 2026. Migration to v2 is required by then.",
        "The v1 API was deprecated on December 31, 2025. Migration to v2 is required immediately.",
        "temporal_error",
    ),
    # ── Correct but rephrased (should NOT be flagged) ──
    TestCase(
        "product_spec",
        "The platform processes up to 10,000 requests per second with sub-millisecond latency.",
        "Our platform handles 10K RPS with latency under one millisecond.",
        "Our platform handles 100K RPS with latency under ten milliseconds.",
        "number_substitution",
    ),
    TestCase(
        "sla",
        "We guarantee 99.95% availability measured on a monthly basis. Downtime credits are issued automatically.",
        "Monthly availability is guaranteed at 99.95% with automatic downtime credits.",
        "Monthly availability is guaranteed at 99.5% with manual downtime credit requests required.",
        "number_and_negation",
    ),
    TestCase(
        "compliance",
        "Personal data is retained for 90 days after account deletion. After 90 days, all data is permanently purged.",
        "After you delete your account, personal data is kept for 90 days and then permanently removed.",
        "After you delete your account, personal data is kept for 365 days and can be recovered at any time.",
        "number_substitution",
    ),
    TestCase(
        "limits",
        "Free accounts are limited to 3 projects and 1 GB of storage. Paid accounts get unlimited projects and 100 GB storage.",
        "Free accounts have a limit of 3 projects with 1 GB storage. Paid plans offer unlimited projects and 100 GB.",
        "Free accounts have a limit of 10 projects with 5 GB storage. Paid plans offer unlimited projects and 500 GB.",
        "number_substitution",
    ),
    TestCase(
        "feature",
        "Two-factor authentication is required for all admin accounts. Regular users can optionally enable it.",
        "Admin accounts must use two-factor authentication. It is optional for regular users.",
        "Two-factor authentication is optional for all account types including admins.",
        "negation_flip",
    ),
    TestCase(
        "api",
        "Rate limiting is set at 100 requests per minute for the standard API. The batch API allows up to 1000 requests per minute.",
        "The standard API has a rate limit of 100 req/min. The batch API allows 1000 req/min.",
        "The standard API has a rate limit of 1000 req/min. The batch API allows 10000 req/min.",
        "number_substitution",
    ),
]


def run_benchmark(use_nli: bool = False):
    from director_ai.core.verified_scorer import VerifiedScorer

    nli = None
    if use_nli:
        from director_ai.core.nli import NLIScorer

        nli = NLIScorer(use_model=True)
        logger.info("NLI model loaded")

    vs = VerifiedScorer(nli_scorer=nli)

    tp = fp = tn = fn = 0
    results = []

    for tc in TEST_CASES:
        # Test correct response — should be APPROVED
        t0 = time.monotonic()
        r_correct = vs.verify(tc.correct, tc.source)
        lat_correct = time.monotonic() - t0

        if r_correct.approved:
            tn += 1  # correctly approved
        else:
            fp += 1  # false positive (flagged a correct response)

        # Test hallucinated response — should be REJECTED
        t0 = time.monotonic()
        r_halluc = vs.verify(tc.hallucinated, tc.source)
        lat_halluc = time.monotonic() - t0

        if not r_halluc.approved:
            tp += 1  # correctly caught
        else:
            fn += 1  # missed hallucination

        results.append(
            {
                "category": tc.category,
                "error_type": tc.error_type,
                "correct_approved": r_correct.approved,
                "correct_confidence": r_correct.confidence,
                "correct_score": round(r_correct.overall_score, 4),
                "halluc_rejected": not r_halluc.approved,
                "halluc_confidence": r_halluc.confidence,
                "halluc_score": round(r_halluc.overall_score, 4),
                "halluc_contradictions": r_halluc.contradicted_count,
                "latency_correct_ms": round(lat_correct * 1000, 1),
                "latency_halluc_ms": round(lat_halluc * 1000, 1),
            }
        )

    total = len(TEST_CASES)
    catch_rate = tp / (tp + fn) if (tp + fn) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    accuracy = (tp + tn) / (total * 2)
    ba = (catch_rate + (1 - fpr)) / 2

    # Per error type
    by_type: dict[str, dict] = {}
    for r in results:
        et = r["error_type"]
        if et not in by_type:
            by_type[et] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        if r["correct_approved"]:
            by_type[et]["tn"] += 1
        else:
            by_type[et]["fp"] += 1
        if r["halluc_rejected"]:
            by_type[et]["tp"] += 1
        else:
            by_type[et]["fn"] += 1

    summary = {
        "mode": "nli" if use_nli else "heuristic",
        "total_cases": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "catch_rate": round(catch_rate * 100, 1),
        "false_positive_rate": round(fpr * 100, 1),
        "balanced_accuracy": round(ba * 100, 1),
        "accuracy": round(accuracy * 100, 1),
        "per_error_type": {},
    }
    for et, counts in sorted(by_type.items()):
        cr = (
            counts["tp"] / (counts["tp"] + counts["fn"])
            if (counts["tp"] + counts["fn"])
            else 0
        )
        fr = (
            counts["fp"] / (counts["fp"] + counts["tn"])
            if (counts["fp"] + counts["tn"])
            else 0
        )
        summary["per_error_type"][et] = {
            "catch_rate": round(cr * 100, 1),
            "fpr": round(fr * 100, 1),
            "n": counts["tp"] + counts["fn"],
        }

    results_dir = Path("gpu_results/verified_bench")
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "nli" if use_nli else "heuristic"
    Path(results_dir / f"verified_{suffix}_results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    Path(results_dir / f"verified_{suffix}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("\n" + "=" * 60)
    print(f"  VerifiedScorer Benchmark ({suffix})")
    print("=" * 60)
    print(f"  Cases: {total} (each tested correct + hallucinated)")
    print(f"  Catch rate: {summary['catch_rate']}% ({tp}/{tp + fn})")
    print(f"  False positive rate: {summary['false_positive_rate']}% ({fp}/{fp + tn})")
    print(f"  Balanced accuracy: {summary['balanced_accuracy']}%")
    print(f"  Accuracy: {summary['accuracy']}%")
    print("=" * 60)
    print("\nPer error type:")
    for et, info in sorted(summary["per_error_type"].items()):
        print(
            f"  {et:25s}: catch={info['catch_rate']}%, FPR={info['fpr']}% (n={info['n']})"
        )

    print("\nDetailed results:")
    for r in results:
        ok = "OK" if r["correct_approved"] else "FP"
        catch = "CAUGHT" if r["halluc_rejected"] else "MISSED"
        print(
            f"  [{ok}/{catch}] {r['category']:15s} {r['error_type']:20s} "
            f"correct={r['correct_score']:.2f}/{r['correct_confidence']} "
            f"halluc={r['halluc_score']:.2f}/{r['halluc_confidence']}"
        )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-nli", action="store_true")
    args = parser.parse_args()
    run_benchmark(use_nli=args.with_nli)


if __name__ == "__main__":
    main()
