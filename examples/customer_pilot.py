# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Customer Pilot Demo

"""Customer pilot: load product docs, verify chatbot answers.

Demonstrates the full workflow a customer would use:
1. Load product documentation into the knowledge base
2. Simulate chatbot answers (some correct, some hallucinated)
3. Run VerifiedScorer on each answer
4. Show per-claim verdicts with source citations

Usage::

    python examples/customer_pilot.py
"""

from __future__ import annotations

# Sample product documentation (simulates a real customer's KB)
PRODUCT_DOCS = {
    "pricing": (
        "DataFlow Pro Pricing\n\n"
        "Starter Plan: $29/month, up to 10 users, 5 GB storage, email support.\n"
        "Business Plan: $79/month, up to 50 users, 50 GB storage, priority support.\n"
        "Enterprise Plan: $199/month, unlimited users, 500 GB storage, "
        "dedicated account manager, SLA guarantee.\n"
        "All plans include a 14-day free trial. Annual billing saves 20%."
    ),
    "features": (
        "DataFlow Pro Features\n\n"
        "Real-time data synchronization across all connected devices.\n"
        "Supports PostgreSQL, MySQL, and MongoDB databases.\n"
        "REST API with rate limiting at 100 requests per minute.\n"
        "OAuth 2.0 and SAML authentication.\n"
        "Automated daily backups with 30-day retention.\n"
        "SOC 2 Type II certified. GDPR compliant."
    ),
    "support": (
        "DataFlow Pro Support Policy\n\n"
        "Email support available Monday through Friday, 9 AM to 6 PM Eastern.\n"
        "Phone support is available only for Business and Enterprise plans.\n"
        "Starter plan customers do not have access to phone support.\n"
        "Critical issues are responded to within 4 hours for Enterprise customers.\n"
        "Bug reports can be submitted at support.dataflow.io."
    ),
    "security": (
        "DataFlow Pro Security\n\n"
        "All data encrypted at rest using AES-256.\n"
        "Data in transit protected by TLS 1.3.\n"
        "Two-factor authentication required for all admin accounts.\n"
        "Regular users can optionally enable two-factor authentication.\n"
        "The system cannot process files larger than 100 MB.\n"
        "Personal data retained for 90 days after account deletion."
    ),
}

# Simulated chatbot answers — mix of correct and hallucinated
CHATBOT_ANSWERS = [
    {
        "question": "How much does the Business plan cost?",
        "answer": "The Business Plan costs $79 per month and includes up to 50 users with 50 GB storage.",
        "expected": "correct",
    },
    {
        "question": "What databases are supported?",
        "answer": "DataFlow Pro supports PostgreSQL, MySQL, MongoDB, Redis, and Cassandra databases.",
        "expected": "hallucinated",
        "error": "Fabricated Redis and Cassandra — not in the docs",
    },
    {
        "question": "Is phone support available for Starter plan?",
        "answer": "Yes, phone support is available for all plans including Starter.",
        "expected": "hallucinated",
        "error": "Negation flip — Starter does NOT have phone support",
    },
    {
        "question": "What is the file size limit?",
        "answer": "The system can process files up to 100 MB in size.",
        "expected": "correct",
    },
    {
        "question": "How long is data retained after deletion?",
        "answer": "Personal data is retained for 365 days after account deletion.",
        "expected": "hallucinated",
        "error": "Number substitution — 90 days, not 365",
    },
    {
        "question": "What encryption is used?",
        "answer": "All data is encrypted at rest using AES-256 and in transit with TLS 1.3.",
        "expected": "correct",
    },
    {
        "question": "Is there a free trial?",
        "answer": "Yes, all plans include a 30-day free trial with full features.",
        "expected": "hallucinated",
        "error": "Number substitution — 14-day trial, not 30-day",
    },
    {
        "question": "What compliance certifications do you have?",
        "answer": "DataFlow Pro is SOC 2 Type II certified, GDPR compliant, and HIPAA certified.",
        "expected": "hallucinated",
        "error": "Fabrication — HIPAA not mentioned in docs",
    },
    {
        "question": "What is the API rate limit?",
        "answer": "The REST API has a rate limit of 100 requests per minute.",
        "expected": "correct",
    },
    {
        "question": "What authentication methods are supported?",
        "answer": "DataFlow Pro supports OAuth 2.0 and SAML authentication.",
        "expected": "correct",
    },
]


def main():
    from director_ai.core.verified_scorer import VerifiedScorer

    # Build the knowledge base from product docs
    full_source = "\n\n".join(PRODUCT_DOCS.values())

    vs = VerifiedScorer()

    print("=" * 70)
    print("  DataFlow Pro — Customer Pilot: Chatbot Answer Verification")
    print("=" * 70)
    print(f"  Knowledge base: {len(PRODUCT_DOCS)} documents, {len(full_source)} chars")
    print(f"  Chatbot answers to verify: {len(CHATBOT_ANSWERS)}")
    print()

    tp = fp = tn = fn = 0

    for i, qa in enumerate(CHATBOT_ANSWERS):
        result = vs.verify(qa["answer"], full_source)

        is_hallucinated = qa["expected"] == "hallucinated"
        detected = not result.approved

        if is_hallucinated and detected:
            tp += 1
            status = "\u2705 CAUGHT"
        elif is_hallucinated and not detected:
            fn += 1
            status = "\u274c MISSED"
        elif not is_hallucinated and detected:
            fp += 1
            status = "\u26a0\ufe0f FALSE POSITIVE"
        else:
            tn += 1
            status = "\u2705 CORRECT"

        print(f"  Q{i + 1}: {qa['question']}")
        print(f"  A: {qa['answer'][:80]}...")
        print(
            f"  Status: {status} (expected={qa['expected']}, score={result.overall_score:.2f}, conf={result.confidence})"
        )

        if result.claims:
            for c in result.claims:
                marker = {
                    "supported": "\u2713",
                    "contradicted": "\u2717",
                    "fabricated": "\u26a0",
                    "unverifiable": "?",
                }.get(c.verdict, "?")
                print(
                    f'    {marker} [{c.verdict}] "{c.claim[:60]}" (trace={c.traceability:.2f})'
                )
                if c.verdict in ("contradicted", "fabricated"):
                    print(f'      Source: "{c.matched_source[:60]}"')

        if is_hallucinated and not detected:
            print(f"    Expected error: {qa.get('error', 'unknown')}")
        print()

    catch_rate = tp / (tp + fn) if (tp + fn) else 0
    fpr_rate = fp / (fp + tn) if (fp + tn) else 0
    ba = (catch_rate + (1 - fpr_rate)) / 2

    print("=" * 70)
    print(f"  Results: {tp} caught, {fn} missed, {fp} false positives, {tn} correct")
    print(f"  Catch rate: {catch_rate:.0%} ({tp}/{tp + fn})")
    print(f"  False positive rate: {fpr_rate:.0%} ({fp}/{fp + tn})")
    print(f"  Balanced accuracy: {ba:.0%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
