#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Verification gems demo — standalone modules, no NLI model needed.

Run:
    python examples/verification_gems_demo.py
"""

from director_ai.agentic.loop_monitor import LoopMonitor
from director_ai.compliance.feedback_loop_detector import FeedbackLoopDetector
from director_ai.core.calibration.conformal import ConformalPredictor
from director_ai.core.scoring.consensus import ConsensusScorer, ModelResponse
from director_ai.core.scoring.temporal_freshness import score_temporal_freshness
from director_ai.core.verification.numeric_verifier import verify_numeric
from director_ai.core.verification.reasoning_verifier import verify_reasoning_chain


def demo_numeric():
    print("=== Numeric Verification ===")
    result = verify_numeric(
        "Revenue grew 50% from $100 to $120. "
        "She was born in 1990 and died in 1980. "
        "There is a 150% probability of success."
    )
    print(f"  Valid: {result.valid}")
    print(f"  Claims found: {result.claims_found}")
    print(f"  Errors: {result.error_count}, Warnings: {result.warning_count}")
    for issue in result.issues:
        print(f"    [{issue.severity}] {issue.issue_type}: {issue.description}")
    print()


def demo_reasoning():
    print("=== Reasoning Chain Verification ===")
    result = verify_reasoning_chain(
        "Step 1: All mammals are warm-blooded. "
        "Step 2: Dogs are mammals. "
        "Step 3: Therefore, dogs are warm-blooded."
    )
    print(f"  Chain valid: {result.chain_valid}")
    print(f"  Steps: {result.steps_found}, Issues: {result.issues_found}")
    for v in result.verdicts:
        print(f"    Step {v.step_index}: {v.verdict} ({v.confidence:.2f})")
    print()


def demo_temporal():
    print("=== Temporal Freshness ===")
    result = score_temporal_freshness("The CEO of Apple is Tim Cook.")
    print(f"  Has temporal claims: {result.has_temporal_claims}")
    print(f"  Overall staleness risk: {result.overall_staleness_risk:.2f}")
    for c in result.claims:
        print(f"    [{c.claim_type}] {c.text} (risk: {c.staleness_risk:.2f})")
    print()


def demo_consensus():
    print("=== Cross-Model Consensus ===")
    scorer = ConsensusScorer(models=["gpt-4o", "claude", "gemini"])
    result = scorer.score_responses(
        [
            ModelResponse(model="gpt-4o", response="Paris is the capital of France"),
            ModelResponse(model="claude", response="Paris is the capital of France"),
            ModelResponse(model="gemini", response="Berlin is the capital of Germany"),
        ]
    )
    print(f"  Agreement score: {result.agreement_score:.2f}")
    print(f"  Has consensus: {result.has_consensus}")
    print("  Pairs:")
    for p in result.pairs:
        print(
            f"    {p.model_a} vs {p.model_b}: divergence={p.divergence:.2f}, agreed={p.agreed}"
        )
    print()


def demo_conformal():
    print("=== Conformal Prediction ===")
    predictor = ConformalPredictor(coverage=0.95)
    scores = [0.9, 0.85, 0.1, 0.15, 0.88, 0.12] * 6
    labels = [False, False, True, True, False, True] * 6
    predictor.calibrate(scores, labels)

    interval = predictor.predict(score=0.7)
    print(f"  P(hallucination) point estimate: {interval.point_estimate:.2f}")
    print(f"  95% interval: [{interval.lower:.2f}, {interval.upper:.2f}]")
    print(f"  Calibration size: {interval.calibration_size}")
    print(f"  Reliable: {interval.is_reliable}")
    print()


def demo_feedback_loop():
    print("=== Feedback Loop Detection ===")
    detector = FeedbackLoopDetector(similarity_threshold=0.5)
    output = "Machine learning enables systems to learn from data patterns."
    detector.record_output(output, 1.0)

    alert = detector.check_input(output)
    if alert:
        print(
            f"  Loop detected: similarity={alert.similarity:.2f}, severity={alert.severity}"
        )
    else:
        print("  No feedback loop detected")

    alert2 = detector.check_input("What is the weather today?")
    print(f"  Different input: loop_detected={alert2 is not None}")
    print()


def demo_agentic():
    print("=== Agentic Loop Monitor ===")
    monitor = LoopMonitor(goal="Find quarterly revenue data", max_steps=10)

    for _i in range(4):
        verdict = monitor.check_step(
            action="search_database",
            args="revenue Q3 2025",
            tokens=500,
        )
        print(
            f"  Step {verdict.step_number}: halt={verdict.should_halt}, "
            f"warn={verdict.should_warn}, drift={verdict.goal_drift_score:.2f}"
        )
        if verdict.reasons:
            for r in verdict.reasons:
                print(f"    -> {r}")
        if verdict.should_halt:
            break

    status = monitor.status()
    print(f"  Total steps: {status.total_steps}")
    print(f"  Circular detections: {status.circular_detections}")
    print()


if __name__ == "__main__":
    demo_numeric()
    demo_reasoning()
    demo_temporal()
    demo_consensus()
    demo_conformal()
    demo_feedback_loop()
    demo_agentic()
    print("All demos completed.")
