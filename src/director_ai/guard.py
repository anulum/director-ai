# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Production Guard (Batteries-Included)
"""High-level production guard combining calibrated scoring, feedback, and tool verification.

Bundles three capabilities that individually exist in the codebase into
a single entry point for production deployments:

1. **Calibrated scoring** — CoherenceScorer + OnlineCalibrator with
   confidence intervals from ConformalPredictor.
2. **Human feedback loop** — FeedbackStore records corrections;
   calibrator absorbs them to update thresholds.
3. **Agent tool-call guardrails** — verify_tool_call checks function
   calls against a manifest before execution.

Usage::

    from director_ai.guard import ProductionGuard

    guard = ProductionGuard.from_profile("medical")
    guard.load_facts({"dosage": "Max 400mg ibuprofen per dose."})

    # Score a response
    result = guard.check("What is the max dose?", "Take up to 800mg.")
    print(result.approved, result.score, result.confidence_interval)

    # Record human correction
    guard.record_feedback(result, correct_label=False)

    # Verify an agent tool call
    tool_result = guard.verify_tool(
        "get_dosage", {"drug": "ibuprofen"}, '{"max_dose": "400mg"}',
        manifest=tool_manifest,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.config import DirectorConfig
from director_ai.core.scoring.verified_scorer import VerifiedScorer
from director_ai.core.types import CoherenceScore, InjectionResult

if TYPE_CHECKING:
    from director_ai.core.safety.injection import InjectionDetector

logger = logging.getLogger("DirectorAI.Guard")


@dataclass
class GuardResult:
    """Result from ProductionGuard.check()."""

    approved: bool
    score: float
    coherence: CoherenceScore
    confidence_interval: tuple[float, float] | None = None
    calibrated_threshold: float | None = None


class ProductionGuard:
    """Batteries-included guardrail for production deployments.

    Wires together CoherenceScorer, OnlineCalibrator, FeedbackStore,
    ConformalPredictor, and VerifiedScorer into a single API.
    """

    def __init__(
        self,
        config: DirectorConfig | None = None,
        store: GroundTruthStore | None = None,
    ) -> None:
        self._config = config or DirectorConfig()
        self._store = store or GroundTruthStore()
        self._scorer = CoherenceScorer(
            threshold=self._config.coherence_threshold,
            ground_truth_store=self._store,
            use_nli=self._config.use_nli,
        )
        self._verified = VerifiedScorer()
        self._calibrator = None
        self._conformal = None
        self._feedback = None
        self._injection_detector: InjectionDetector | None = None

    @classmethod
    def from_profile(
        cls,
        profile: str = "fast",
        store: GroundTruthStore | None = None,
    ) -> ProductionGuard:
        """Create a guard from a named profile (fast, medical, finance, etc.)."""
        config = DirectorConfig.from_profile(profile)
        return cls(config=config, store=store)

    def load_facts(self, facts: dict[str, str]) -> None:
        """Load key-value facts into the knowledge base."""
        for k, v in facts.items():
            self._store.add(k, v)

    def enable_calibration(self, alpha: float = 0.1) -> None:
        """Enable online calibration with conformal confidence intervals.

        Parameters
        ----------
        alpha : float — significance level for conformal intervals (default 0.1 = 90% CI).
        """
        from director_ai.core.calibration.conformal import ConformalPredictor
        from director_ai.core.calibration.feedback_store import FeedbackStore
        from director_ai.core.calibration.online_calibrator import OnlineCalibrator

        fb = FeedbackStore()
        self._feedback = fb  # type: ignore[assignment]
        self._calibrator = OnlineCalibrator(store=fb)  # type: ignore[assignment]
        self._conformal = ConformalPredictor(coverage=1.0 - alpha)  # type: ignore[assignment]
        logger.info("Calibration enabled (alpha=%.2f)", alpha)

    def check(
        self,
        prompt: str,
        response: str,
        atomic: bool = False,
    ) -> GuardResult:
        """Score a response and return a GuardResult with optional calibration."""
        approved, cs = self._scorer.review(prompt, response)

        ci = None
        cal_threshold = None
        if self._conformal is not None:
            ci = self._conformal.predict_interval(cs.score)
        if self._calibrator is not None:
            cal_threshold = self._calibrator.adjusted_threshold

        return GuardResult(
            approved=approved,
            score=cs.score,
            coherence=cs,
            confidence_interval=ci,
            calibrated_threshold=cal_threshold,
        )

    def check_verified(
        self,
        response: str,
        source: str,
        atomic: bool = True,
    ):
        """Run per-claim verification against source text."""
        return self._verified.verify(response, source, atomic=atomic)

    def record_feedback(
        self,
        result: GuardResult,
        correct_label: bool,
    ) -> None:
        """Record human feedback on a guard result.

        Feeds the correction into the calibrator for threshold adjustment.
        """
        if self._feedback is None or self._calibrator is None:
            logger.warning("Calibration not enabled — call enable_calibration() first")
            return
        self._feedback.add(result.score, correct_label)
        self._calibrator.update(result.score, correct_label)
        if self._conformal is not None:
            self._conformal.add_observation(result.score, correct_label)

    def check_injection(
        self,
        intent: str,
        response: str,
        user_query: str = "",
        system_prompt: str = "",
    ) -> InjectionResult:
        """Detect prompt injection effects in a response via NLI divergence.

        Lazily initialises InjectionDetector on first call using config
        thresholds.  Reuses the scorer's NLI model when available.
        """
        if self._injection_detector is None:
            from director_ai.core.safety.injection import InjectionDetector

            nli = getattr(self._scorer, "_nli", None)
            cfg = self._config
            self._injection_detector = InjectionDetector(
                nli_scorer=nli,
                injection_threshold=cfg.injection_threshold,
                drift_threshold=cfg.injection_drift_threshold,
                injection_claim_threshold=cfg.injection_claim_threshold,
                baseline_divergence=cfg.injection_baseline_divergence,
                stage1_weight=cfg.injection_stage1_weight,
            )
            logger.info(
                "Injection detector initialised (threshold=%.2f)",
                cfg.injection_threshold,
            )

        return self._injection_detector.detect(
            intent=intent,
            response=response,
            user_query=user_query,
            system_prompt=system_prompt,
        )

    def verify_tool(
        self,
        function_name: str,
        arguments: dict,
        claimed_result: str = "",
        manifest: dict | None = None,
        execution_log: list[dict] | None = None,
    ):
        """Verify an agent tool/function call against a manifest."""
        from director_ai.core.verification.tool_call_verifier import verify_tool_call

        return verify_tool_call(
            function_name=function_name,
            arguments=arguments,
            claimed_result=claimed_result,
            manifest=manifest,
            execution_log=execution_log,
        )

    @property
    def scorer(self) -> CoherenceScorer:
        return self._scorer

    @property
    def config(self) -> DirectorConfig:
        return self._config
