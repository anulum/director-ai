# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — standalone verification and analysis routes

"""Standalone verification and analysis routes.

Covers numeric, reasoning, temporal-freshness, consensus, adversarial
robustness, conformal prediction, feedback-loop, and agentic step checks. Split
out of the ``create_app`` factory. Each handler constructs its own
verifier/detector from the request body; only the adversarial route reads the
scorer from ``request.app.state``. ``create_verification_router`` needs no
construction-time dependencies.
"""

from __future__ import annotations

try:
    from fastapi import APIRouter, HTTPException, Request

    from .._server_models import (
        AdversarialPatternResponse,
        AdversarialResponse,
        AgenticStepRequest,
        AgenticStepResponse,
        ConformalRequest,
        ConformalResponse,
        ConsensusRequest,
        ConsensusResponse,
        ConsensusResponseItem,
        FeedbackLoopCheckRequest,
        FeedbackLoopResponse,
        FreshnessClaimResponse,
        FreshnessResponse,
        FreshnessStatusResponse,
        NumericIssueResponse,
        NumericVerifyResponse,
        PairwiseAgreementResponse,
        ReasoningVerdictResponse,
        ReasoningVerifyResponse,
        ReviewRequest,
        TextRequest,
    )

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - server extras absent
    _FASTAPI_AVAILABLE = False


def create_verification_router() -> APIRouter:
    """Build the standalone verification route group."""
    if not _FASTAPI_AVAILABLE:  # pragma: no cover - guarded by create_app
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install director-ai[server]",
        )

    router = APIRouter()

    @router.post("/v1/verify/numeric", response_model=NumericVerifyResponse)
    async def verify_numeric_endpoint(req: TextRequest) -> NumericVerifyResponse:
        """Verify numeric consistency in text.

        Checks percentage arithmetic, date logic, probability bounds,
        order-of-magnitude sanity, and internal number consistency.
        """
        from ..core.verification.numeric_verifier import verify_numeric

        result = verify_numeric(req.text)
        return NumericVerifyResponse(
            claims_found=result.claims_found,
            issues=[
                NumericIssueResponse(
                    issue_type=i.issue_type,
                    description=i.description,
                    severity=i.severity,
                    context=i.context,
                )
                for i in result.issues
            ],
            valid=result.valid,
            error_count=result.error_count,
            warning_count=result.warning_count,
        )

    @router.post("/v1/verify/reasoning", response_model=ReasoningVerifyResponse)
    async def verify_reasoning_endpoint(req: TextRequest) -> ReasoningVerifyResponse:
        """Verify logical structure of a reasoning chain.

        Extracts reasoning steps and checks each follows from its
        premises. Detects non-sequiturs, circular reasoning, and
        unsupported leaps.
        """
        from ..core.verification.reasoning_verifier import verify_reasoning_chain

        result = verify_reasoning_chain(req.text)
        return ReasoningVerifyResponse(
            steps_found=result.steps_found,
            verdicts=[
                ReasoningVerdictResponse(
                    step_index=v.step_index,
                    step_text=v.step_text,
                    verdict=v.verdict,
                    confidence=v.confidence,
                    reason=v.reason,
                    premise_text=v.premise_text,
                )
                for v in result.verdicts
            ],
            chain_valid=result.chain_valid,
            issues_found=result.issues_found,
        )

    @router.post("/v1/temporal-freshness", response_model=FreshnessResponse)
    async def temporal_freshness_endpoint(req: TextRequest) -> FreshnessResponse:
        """Score temporal freshness of claims in text.

        Detects date-sensitive entities (positions, prices, statistics)
        and assesses staleness risk based on entity type.
        """
        from ..core.scoring.temporal_freshness import score_temporal_freshness

        result = score_temporal_freshness(req.text)
        return FreshnessResponse(
            claims=[
                FreshnessClaimResponse(
                    text=c.text,
                    claim_type=c.claim_type,
                    staleness_risk=c.staleness_risk,
                    reason=c.reason,
                    source_id=c.source_id,
                    external_status=c.external_status,
                )
                for c in result.claims
            ],
            citation_statuses=[
                FreshnessStatusResponse(
                    source_id=v.source_id,
                    status=v.status,
                    risk=v.risk,
                    reason=v.reason,
                    status_source=v.status_source,
                )
                for v in result.citation_status_verdicts
            ],
            overall_staleness_risk=result.overall_staleness_risk,
            external_status_risk=result.external_status_risk,
            has_temporal_claims=result.has_temporal_claims,
            stale_claim_count=len(result.stale_claims),
            risky_status_count=len(result.risky_statuses),
        )

    @router.post("/v1/consensus", response_model=ConsensusResponse)
    async def consensus_endpoint(req: ConsensusRequest) -> ConsensusResponse:
        """Score factual agreement across pre-generated model responses.

        Accepts responses from multiple models and computes pairwise
        agreement using Jaccard word overlap.
        """
        from ..core.scoring.consensus import ConsensusScorer, ModelResponse

        scorer = ConsensusScorer(
            models=[r.model for r in req.responses],
            generate_fn=None,
        )
        model_responses = [
            ModelResponse(model=r.model, response=r.response) for r in req.responses
        ]
        result = scorer.score_responses(model_responses)
        return ConsensusResponse(
            responses=[
                ConsensusResponseItem(model=r.model, response=r.response)
                for r in result.responses
            ],
            pairs=[
                PairwiseAgreementResponse(
                    model_a=p.model_a,
                    model_b=p.model_b,
                    divergence=p.divergence,
                    agreed=p.agreed,
                )
                for p in result.pairs
            ],
            agreement_score=result.agreement_score,
            lowest_pair_agreement=result.lowest_pair_agreement,
            has_consensus=result.has_consensus,
            num_models=result.num_models,
        )

    @router.post("/v1/adversarial/test", response_model=AdversarialResponse)
    async def adversarial_test_endpoint(
        req: ReviewRequest, request: Request
    ) -> AdversarialResponse:
        """Run adversarial robustness tests against the guardrail.

        Uses the prompt+response as a baseline, then tests adversarial
        transformations of the response against the scorer.
        """
        from ..testing.adversarial_suite import AdversarialTester

        app_scorer = request.app.state._state.get("scorer")
        if app_scorer is None:
            raise HTTPException(503, "Scorer not initialised")

        def review_fn(prompt: str, response: str) -> tuple[bool, float]:
            """Adapt the configured scorer to the adversarial tester API."""
            approved, score = app_scorer.review(prompt, response)
            return approved, score.score

        tester = AdversarialTester(
            review_fn=review_fn,
            prompt=req.prompt,
        )
        report = tester.run()
        return AdversarialResponse(
            total_patterns=report.total_patterns,
            detected=report.detected,
            bypassed=report.bypassed,
            detection_rate=report.detection_rate,
            is_robust=report.is_robust,
            vulnerable_categories=report.vulnerable_categories,
            results=[
                AdversarialPatternResponse(
                    name=r.pattern.name,
                    category=r.pattern.category,
                    transform=r.pattern.transform,
                    detected=r.detected,
                    score=r.score,
                    original_score=r.original_score,
                )
                for r in report.results
            ],
        )

    @router.post("/v1/conformal/predict", response_model=ConformalResponse)
    async def conformal_predict_endpoint(req: ConformalRequest) -> ConformalResponse:
        """Compute conformal prediction interval for hallucination probability.

        Optionally calibrate from provided historical data first.
        """
        from ..core.calibration.conformal import ConformalPredictor

        predictor = ConformalPredictor(coverage=req.coverage)
        if req.calibration_scores and req.calibration_labels:
            if len(req.calibration_scores) != len(req.calibration_labels):
                raise HTTPException(
                    status_code=422,
                    detail="calibration_scores and calibration_labels must have same length",
                )
            predictor.calibrate(req.calibration_scores, req.calibration_labels)
        interval = predictor.predict(req.score)
        return ConformalResponse(
            point_estimate=interval.point_estimate,
            lower=interval.lower,
            upper=interval.upper,
            coverage=interval.coverage,
            calibration_size=interval.calibration_size,
            is_reliable=interval.is_reliable,
        )

    @router.post("/v1/compliance/feedback-loops", response_model=FeedbackLoopResponse)
    async def feedback_loop_endpoint(
        req: FeedbackLoopCheckRequest,
    ) -> FeedbackLoopResponse:
        """Check if input text matches any previous AI output (feedback loop).

        Pass previous_outputs to seed the detector buffer, then checks
        the input_text against them.
        """
        from ..compliance.feedback_loop_detector import FeedbackLoopDetector

        detector = FeedbackLoopDetector(
            similarity_threshold=req.similarity_threshold,
        )
        for i, output in enumerate(req.previous_outputs):
            detector.record_output(output, float(i))

        alert = detector.check_input(req.input_text)
        if alert is None:
            return FeedbackLoopResponse(
                loop_detected=False,
                similarity=0.0,
            )
        return FeedbackLoopResponse(
            loop_detected=True,
            similarity=alert.similarity,
            severity=alert.severity,
            matched_output=alert.matched_output,
        )

    @router.post("/v1/agentic/check-step", response_model=AgenticStepResponse)
    async def agentic_check_step_endpoint(
        req: AgenticStepRequest,
    ) -> AgenticStepResponse:
        """Evaluate a single agentic step for safety issues.

        Replays step_history to build monitor state, then evaluates
        the current step.
        """
        from ..agentic.loop_monitor import LoopMonitor

        monitor = LoopMonitor(
            goal=req.goal,
            max_steps=req.max_steps,
        )
        for prev in req.step_history:
            monitor.check_step(
                action=prev.get("action", ""),
                args=prev.get("args", ""),
            )
        verdict = monitor.check_step(
            action=req.action,
            args=req.args,
            result=req.result,
            tokens=req.tokens,
        )
        return AgenticStepResponse(
            step_number=verdict.step_number,
            should_halt=verdict.should_halt,
            should_warn=verdict.should_warn,
            reasons=verdict.reasons,
            goal_drift_score=verdict.goal_drift_score,
            budget_remaining_pct=verdict.budget_remaining_pct,
        )

    return router
