# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""director-ai verification-gem CLI commands (numeric, reasoning, temporal, consensus, adversarial)."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def _cmd_verify_numeric(args: list[str]) -> None:
    """Check numeric consistency in text."""
    if not args:
        print("Usage: director-ai verify-numeric <text>")
        sys.exit(1)

    from director_ai.core.verification.numeric_verifier import verify_numeric

    result = verify_numeric(" ".join(args))
    print(f"Valid:    {result.valid}")
    print(f"Claims:  {result.claims_found}")
    print(f"Errors:  {result.error_count}")
    print(f"Warnings:{result.warning_count}")
    for issue in result.issues:
        print(f"  [{issue.severity}] {issue.issue_type}: {issue.description}")


def _cmd_verify_reasoning(args: list[str]) -> None:
    """Verify logical structure of a reasoning chain."""
    if not args:
        print("Usage: director-ai verify-reasoning <text>")
        sys.exit(1)

    from director_ai.core.verification.reasoning_verifier import verify_reasoning_chain

    result = verify_reasoning_chain(" ".join(args))
    print(f"Chain valid: {result.chain_valid}")
    print(f"Steps:       {result.steps_found}")
    print(f"Issues:      {result.issues_found}")
    for v in result.verdicts:
        print(f"  Step {v.step_index}: {v.verdict} ({v.confidence:.2f}) {v.reason}")


def _cmd_temporal_freshness(args: list[str]) -> None:
    """Score temporal freshness of claims."""
    if not args:
        print("Usage: director-ai temporal-freshness <text>")
        sys.exit(1)

    from director_ai.core.scoring.temporal_freshness import score_temporal_freshness

    result = score_temporal_freshness(" ".join(args))
    print(f"Has temporal claims: {result.has_temporal_claims}")
    print(f"Staleness risk:      {result.overall_staleness_risk:.2f}")
    print(f"Stale claims:        {len(result.stale_claims)}")
    for c in result.claims:
        print(f"  [{c.claim_type}] {c.text} (risk: {c.staleness_risk:.2f})")


def _cmd_check_step(args: list[str]) -> None:
    """Check an agentic step for safety issues."""
    if len(args) < 2:
        print("Usage: director-ai check-step <goal> <action> [args]")
        sys.exit(1)

    from director_ai.agentic.loop_monitor import LoopMonitor

    goal = args[0]
    action = args[1]
    action_args = args[2] if len(args) > 2 else ""

    monitor = LoopMonitor(goal=goal)
    verdict = monitor.check_step(action=action, args=action_args)
    print(f"Step:    {verdict.step_number}")
    print(f"Halt:    {verdict.should_halt}")
    print(f"Warn:    {verdict.should_warn}")
    print(f"Drift:   {verdict.goal_drift_score:.2f}")
    print(f"Budget:  {verdict.budget_remaining_pct:.0%}")
    if verdict.reasons:
        for r in verdict.reasons:
            print(f"  -> {r}")


def _cmd_consensus(args: list[str]) -> None:
    """Score factual agreement across multiple model responses."""
    if len(args) < 2:
        print(
            "Usage: director-ai consensus <model1:response1> <model2:response2> ...\n"
            "\n"
            "Each argument is model_name:response_text (colon-separated).\n"
            "Example: director-ai consensus 'gpt:Paris is the capital' 'claude:Paris is the capital'"
        )
        sys.exit(1)

    from director_ai.core.scoring.consensus import ConsensusScorer, ModelResponse

    responses = []
    for arg in args:
        if ":" not in arg:
            print(f"Invalid format: {arg!r} — expected model:response")
            sys.exit(1)
        model, _, response = arg.partition(":")
        responses.append(ModelResponse(model=model.strip(), response=response.strip()))

    scorer = ConsensusScorer(models=[r.model for r in responses])
    result = scorer.score_responses(responses)
    print(f"Models:    {result.num_models}")
    print(f"Agreement: {result.agreement_score:.2f}")
    print(f"Consensus: {result.has_consensus}")
    print(f"Lowest:    {result.lowest_pair_agreement:.2f}")
    for p in result.pairs:
        status = "agree" if p.agreed else "DISAGREE"
        print(f"  {p.model_a} vs {p.model_b}: {status} (divergence={p.divergence:.2f})")


def _cmd_adversarial_test(args: list[str]) -> None:
    """Run adversarial robustness test against the guardrail."""
    from director_ai.core.config import DirectorConfig

    cfg = DirectorConfig.from_env()
    scorer = cfg.build_scorer()

    from director_ai.testing.adversarial_suite import AdversarialTester

    prompt = args[0] if args else "Tell me about this topic."

    def review_fn(p: str, r: str) -> tuple[bool, float]:
        approved, score = scorer.review(p, r)
        return approved, score.score

    tester = AdversarialTester(review_fn=review_fn, prompt=prompt)
    report = tester.run()
    print(f"Patterns:   {report.total_patterns}")
    print(f"Detected:   {report.detected}")
    print(f"Bypassed:   {report.bypassed}")
    print(f"Rate:       {report.detection_rate:.0%}")
    print(f"Robust:     {report.is_robust}")
    if report.vulnerable_categories:
        print(f"Vulnerable: {', '.join(report.vulnerable_categories)}")
