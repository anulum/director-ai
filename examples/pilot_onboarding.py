# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Pilot Onboarding Script

"""End-to-end onboarding for new pilot customers.

Walks through every step a customer needs to deploy Director-AI
as a hallucination guardrail for their chatbot:

1. Load product documentation (plain text or files)
2. Build a vector-backed knowledge base
3. Configure scoring thresholds
4. Verify sample chatbot answers with per-claim verdicts
5. Show integration patterns (API server, streaming, VoiceGuard)

Usage::

    python examples/pilot_onboarding.py
    python examples/pilot_onboarding.py --profile medical
    python examples/pilot_onboarding.py --use-nli
"""

from __future__ import annotations

import sys

# ── Sample product documentation ────────────────────────────────────
# Replace with your own docs, or point --docs-dir at a folder of .txt/.md files.

ACME_DOCS = [
    (
        "Acme CloudSync Pricing\n\n"
        "Free Tier: 5 users, 1 GB storage, community support.\n"
        "Team Plan: $19/user/month, 25 users max, 100 GB, email support.\n"
        "Business Plan: $49/user/month, unlimited users, 1 TB, "
        "priority support, SSO, audit logs.\n"
        "Enterprise: custom pricing, dedicated infrastructure, SLA.\n"
        "All paid plans include a 14-day free trial. Annual billing saves 15%."
    ),
    (
        "Acme CloudSync Features\n\n"
        "Real-time file synchronisation across desktop, mobile, and web.\n"
        "End-to-end encryption (AES-256 at rest, TLS 1.3 in transit).\n"
        "Version history: 90 days on Team, unlimited on Business/Enterprise.\n"
        "Integrations: Slack, Microsoft Teams, Google Workspace, Jira.\n"
        "API rate limit: 500 requests per minute per user.\n"
        "Maximum file size: 5 GB per file."
    ),
    (
        "Acme CloudSync Support Policy\n\n"
        "Community forum available to all tiers.\n"
        "Email support: Team and above, response within 24 hours.\n"
        "Priority support: Business and above, response within 4 hours.\n"
        "Phone support: Enterprise only.\n"
        "Uptime SLA: 99.9% for Business, 99.99% for Enterprise.\n"
        "Maintenance windows: Sundays 02:00–04:00 UTC, announced 72 hours in advance."
    ),
    (
        "Acme CloudSync Security & Compliance\n\n"
        "SOC 2 Type II certified. ISO 27001 certified.\n"
        "GDPR compliant. Data residency options: US, EU, APAC.\n"
        "Two-factor authentication mandatory for all admin accounts.\n"
        "SSO via SAML 2.0 and OpenID Connect (Business and above).\n"
        "Data retained for 30 days after account deletion.\n"
        "Penetration testing conducted quarterly by third-party auditors."
    ),
]

# Simulated chatbot answers — mix of correct and hallucinated
CHATBOT_QA = [
    {
        "question": "How much does the Team plan cost?",
        "answer": "The Team Plan costs $19 per user per month and supports up to 25 users.",
        "label": "correct",
    },
    {
        "question": "What integrations are available?",
        "answer": (
            "Acme CloudSync integrates with Slack, Microsoft Teams, "
            "Google Workspace, Jira, Salesforce, and Notion."
        ),
        "label": "hallucinated",
        "note": "Salesforce and Notion are fabricated — not in the docs",
    },
    {
        "question": "Is phone support available on the Team plan?",
        "answer": "Yes, phone support is available for all paid plans including Team.",
        "label": "hallucinated",
        "note": "Phone support is Enterprise only",
    },
    {
        "question": "What is the uptime SLA for Business?",
        "answer": "Business plan has a 99.9% uptime SLA.",
        "label": "correct",
    },
    {
        "question": "How long is data retained after deletion?",
        "answer": "Data is retained for 90 days after account deletion.",
        "label": "hallucinated",
        "note": "30 days, not 90 — confused with version history retention",
    },
    {
        "question": "What encryption does CloudSync use?",
        "answer": "AES-256 at rest and TLS 1.3 in transit.",
        "label": "correct",
    },
    {
        "question": "What compliance certifications do you have?",
        "answer": (
            "Acme CloudSync is SOC 2 Type II, ISO 27001, HIPAA, and FedRAMP certified."
        ),
        "label": "hallucinated",
        "note": "HIPAA and FedRAMP are fabricated",
    },
    {
        "question": "What is the maximum file size?",
        "answer": "The maximum file size is 5 GB per file.",
        "label": "correct",
    },
    {
        "question": "What is the API rate limit?",
        "answer": "The API rate limit is 500 requests per minute per user.",
        "label": "correct",
    },
    {
        "question": "Is there a free trial?",
        "answer": "Yes, all paid plans include a 30-day free trial.",
        "label": "hallucinated",
        "note": "14-day trial, not 30-day",
    },
]


def parse_args(argv: list[str]) -> dict:
    profile = "fast"
    use_nli = False
    i = 0
    while i < len(argv):
        if argv[i] == "--profile" and i + 1 < len(argv):
            profile = argv[i + 1]
            i += 2
        elif argv[i] == "--use-nli":
            use_nli = True
            i += 1
        else:
            i += 1
    return {"profile": profile, "use_nli": use_nli}


def main(argv: list[str] | None = None):
    args = parse_args(argv or sys.argv[1:])

    from director_ai.core import CoherenceScorer, GroundTruthStore
    from director_ai.core.config import DirectorConfig
    from director_ai.core.verified_scorer import VerifiedScorer

    # ── Step 1: Build knowledge base ────────────────────────────────
    print("=" * 72)
    print("  Director-AI — Pilot Onboarding")
    print("=" * 72)
    print()
    print("Step 1: Loading product documentation into knowledge base...")

    store = GroundTruthStore()
    for i, doc in enumerate(ACME_DOCS):
        title = doc.split("\n")[0]
        store.add(f"doc_{i}", doc)
        print(f"  Loaded: {title}")

    total_chars = sum(len(d) for d in ACME_DOCS)
    print(f"  -> {len(ACME_DOCS)} documents, {total_chars:,} characters")
    print()

    # ── Step 2: Configure scorer ────────────────────────────────────
    print("Step 2: Configuring scorer...")

    cfg = DirectorConfig.from_profile(args["profile"])
    scorer = CoherenceScorer(
        threshold=cfg.coherence_threshold,
        hard_limit=cfg.hard_limit,
        ground_truth_store=store,
        use_nli=args["use_nli"],
    )

    mode = "NLI + heuristic" if args["use_nli"] else "heuristic-only"
    print(f"  Profile: {args['profile']}")
    print(f"  Mode: {mode}")
    print(f"  Threshold: {cfg.coherence_threshold}")
    print(f"  Hard limit: {cfg.hard_limit}")
    print()

    # ── Step 3: Verify chatbot answers ──────────────────────────────
    print("Step 3: Verifying chatbot answers...")
    print()

    tp = fp = tn = fn = 0
    full_source = "\n\n".join(ACME_DOCS)
    vs = VerifiedScorer()

    for idx, qa in enumerate(CHATBOT_QA):
        _, score = scorer.review(qa["question"], qa["answer"])
        vr = vs.verify(qa["answer"], full_source)

        is_hallucinated = qa["label"] == "hallucinated"
        detected = not vr.approved

        if is_hallucinated and detected:
            tp += 1
            tag = "CAUGHT"
        elif is_hallucinated and not detected:
            fn += 1
            tag = "MISSED"
        elif not is_hallucinated and detected:
            fp += 1
            tag = "FALSE POS"
        else:
            tn += 1
            tag = "OK"

        print(f"  Q{idx + 1}: {qa['question']}")
        print(f"  A: {qa['answer'][:90]}")
        print(
            f"  [{tag}] coherence={score.score:.3f} "
            f"verified={vr.overall_score:.3f} "
            f"claims={len(vr.claims)}"
        )
        if vr.claims:
            for c in vr.claims:
                markers = {
                    "supported": "+",
                    "contradicted": "X",
                    "fabricated": "!",
                    "unverifiable": "?",
                }
                m = markers.get(c.verdict, "?")
                print(f'    [{m}] {c.verdict}: "{c.claim[:70]}"')
        if qa.get("note"):
            print(f"    Note: {qa['note']}")
        print()

    # ── Step 4: Results summary ─────────────────────────────────────
    catch_rate = tp / (tp + fn) if (tp + fn) else 0
    false_pos_rate = fp / (fp + tn) if (fp + tn) else 0
    ba = (catch_rate + (1 - false_pos_rate)) / 2

    print("-" * 72)
    print("  Results Summary")
    print("-" * 72)
    print(f"  Caught: {tp}  Missed: {fn}  False positives: {fp}  Correct: {tn}")
    print(f"  Catch rate:       {catch_rate:.0%} ({tp}/{tp + fn})")
    print(f"  False positive:   {false_pos_rate:.0%} ({fp}/{fp + tn})")
    print(f"  Balanced accuracy: {ba:.0%}")
    print()

    # ── Step 5: Next steps ──────────────────────────────────────────
    print("-" * 72)
    print("  Next Steps")
    print("-" * 72)
    print(
        "  1. Replace ACME_DOCS with your product documentation\n"
        "  2. Run with --use-nli for NLI-based scoring (pip install director-ai[nli])\n"
        "  3. Ingest files from disk:\n"
        "       director-ai ingest ./docs/ --persist ./kb_data\n"
        "  4. Start the API server:\n"
        "       director-ai serve --port 8080\n"
        "  5. Add streaming guardrail:\n"
        "       from director_ai import VoiceGuard\n"
        "  6. See full guide: https://anulum.github.io/director-ai/guide/kb-ingestion/"
    )
    print()


if __name__ == "__main__":
    main()
