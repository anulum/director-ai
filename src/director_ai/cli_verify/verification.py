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
    if _is_help_request(args):
        _print_verify_numeric_help()
        return

    if not args:
        _print_verify_numeric_help()
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
    if _is_help_request(args):
        _print_verify_reasoning_help()
        return

    if not args:
        _print_verify_reasoning_help()
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
    if _is_help_request(args):
        _print_temporal_freshness_help()
        return

    if not args:
        _print_temporal_freshness_help()
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
    if _is_help_request(args):
        _print_check_step_help()
        return

    if len(args) < 2:
        _print_check_step_help()
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
    if _is_help_request(args):
        _print_consensus_help()
        return

    if len(args) < 2:
        _print_consensus_help()
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
    if _is_help_request(args):
        _print_adversarial_test_help()
        return

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


def _is_help_request(args: list[str]) -> bool:
    """Return whether a command received an explicit help token."""
    return bool(args) and args[0] in ("-h", "--help", "help")


def _print_verify_numeric_help() -> None:
    """Print verify-numeric usage without importing numeric verification code."""
    print(
        "Usage: director-ai verify-numeric <text>\n"
        "\n"
        "Check numeric consistency in text and report claims, errors, and warnings.\n"
    )


def _print_verify_reasoning_help() -> None:
    """Print verify-reasoning usage without importing reasoning verification code."""
    print(
        "Usage: director-ai verify-reasoning <text>\n"
        "\n"
        "Verify the logical structure of a reasoning chain.\n"
    )


def _print_temporal_freshness_help() -> None:
    """Print temporal-freshness usage without scoring temporal claims."""
    print(
        "Usage: director-ai temporal-freshness <text>\n"
        "\n"
        "Score temporal freshness and staleness risk for claims in text.\n"
    )


def _print_check_step_help() -> None:
    """Print check-step usage without constructing the loop monitor."""
    print(
        "Usage: director-ai check-step <goal> <action> [args]\n"
        "\n"
        "Check one agentic step for halt, warning, drift, and budget signals.\n"
    )


def _print_consensus_help() -> None:
    """Print consensus usage without constructing the consensus scorer."""
    print(
        "Usage: director-ai consensus <model:response> <model:response> ...\n"
        "\n"
        "Each argument is model_name:response_text (colon-separated).\n"
        "Example: director-ai consensus 'gpt:Paris is the capital' "
        "'claude:Paris is the capital'\n"
    )


def _print_adversarial_test_help() -> None:
    """Print adversarial-test usage without building a scorer or running probes."""
    print(
        "Usage: director-ai adversarial-test [prompt]\n"
        "\n"
        "Run the adversarial prompt suite against the configured guardrail.\n"
    )
