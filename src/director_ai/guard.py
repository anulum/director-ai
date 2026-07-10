# SPDX-License-Identifier: Apache-2.0
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
4. **Sector policy controls** — optional deterministic policy checks for
   high-stakes domains such as banking responses.

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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from director_ai._guard_canary import CanaryOperationsMixin
from director_ai._guard_defence import (
    FirewallDecision as FirewallDecision,
)
from director_ai._guard_defence import (
    ResponseDefenceMixin,
)
from director_ai._guard_distributed import DistributedTrustMixin
from director_ai._guard_hardening import RuntimeHardeningMixin
from director_ai._guard_quality import DecisionQualityMixin
from director_ai._guard_verification import ExtendedVerificationMixin
from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.config import DirectorConfig
from director_ai.core.license import enforce_capability_tier
from director_ai.core.redactor import PIIRedactor
from director_ai.core.scoring.verified_scorer import VerifiedScorer
from director_ai.core.types import CoherenceScore

if TYPE_CHECKING:
    # Advanced-tier (BUSL-1.1) types used only in annotations — runtime imports
    # are lazy inside the methods that need them, so the Apache core wheel does
    # not require these modules to be installed.
    from director_ai.core.financial_services import BankingPolicyReport
    from director_ai.core.scoring.verified_scorer import VerificationResult

logger = logging.getLogger("DirectorAI.Guard")


@dataclass
class GuardResult:
    """Result from ProductionGuard.check()."""

    approved: bool
    score: float
    coherence: CoherenceScore
    confidence_interval: tuple[float, float] | None = None
    calibrated_threshold: float | None = None
    sector_policy_report: BankingPolicyReport | None = None
    uncertainty_action: str | None = None


class ProductionGuard(
    CanaryOperationsMixin,
    ResponseDefenceMixin,
    DistributedTrustMixin,
    RuntimeHardeningMixin,
    DecisionQualityMixin,
    ExtendedVerificationMixin,
):
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
        # PII redaction parity with the REST server: when redact_pii is set, the
        # SDK path scrubs prompt + response before scoring, so direct guard use
        # does not leak PII that the server would have redacted.
        self._redactor = PIIRedactor(
            enabled=bool(getattr(self._config, "redact_pii", False)),
        )
        # Calibration pieces are lazily installed by
        # :meth:`enable_calibration`; declare them as optionals
        # up-front so the later assignments do not narrow.
        self._calibrator: Any = None
        self._conformal: Any = None
        self._feedback: Any = None
        self._uncertainty_router: Any = None
        self._injection_detector = None
        self._moderation_detectors = None
        self._canary_registry = None
        self._canary_detector = None
        self._preflight = None
        self._risk_threshold = None
        self._labelling_cockpit = None
        self._temporal_consistency = None
        self._self_healing = None
        self._dp_retrieval = None
        self._root_cause = None
        self._output_trust = None
        self._execution_rings = None
        self._output_integrity = None
        self._ml_bom = None
        self._rasp = None
        self._threat_intel = None
        self._forecaster = None
        self._temporal_refresher = None
        self._cross_model = None
        self._economics = None
        self._multimodal = None
        self._span_detector = None

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
        self._feedback = fb
        self._calibrator = OnlineCalibrator(store=fb)
        self._conformal = ConformalPredictor(coverage=1.0 - alpha)
        logger.info("Calibration enabled (alpha=%.2f)", alpha)

    def enable_uncertainty_routing(
        self,
        *,
        allow_upper: float = 0.2,
        reject_lower: float = 0.8,
        escalate_human_width: float = 0.5,
    ) -> None:
        """Route calibrated results by uncertainty.

        Requires :meth:`enable_calibration`; once enabled, :meth:`check`
        populates :attr:`GuardResult.uncertainty_action` from the conformal
        interval (``allow`` / ``reject`` / ``escalate_human`` /
        ``escalate_model``).
        """
        from director_ai.core.routing.uncertainty_router import UncertaintyRouter

        self._uncertainty_router = UncertaintyRouter(
            allow_upper=allow_upper,
            reject_lower=reject_lower,
            escalate_human_width=escalate_human_width,
        )
        logger.info("Uncertainty routing enabled")

    def check(
        self,
        prompt: str,
        response: str,
        atomic: bool = False,
        sector_policy: str | None = None,
        evidence_refs: Iterable[str] = (),
        numeric_evidence_refs: Iterable[str] = (),
        policy_refs: Iterable[str] = (),
        jurisdiction: str = "US",
        product_line: str = "default",
        human_review_acknowledged: bool = False,
    ) -> GuardResult:
        """Score a response and return a GuardResult with optional policy checks."""
        if self._redactor.enabled:
            prompt = self._redactor.redact(prompt)
            response = self._redactor.redact(response)
        sector_report = self._evaluate_sector_policy(
            sector_policy=sector_policy,
            prompt=prompt,
            response=response,
            evidence_refs=evidence_refs,
            numeric_evidence_refs=numeric_evidence_refs,
            policy_refs=policy_refs,
            jurisdiction=jurisdiction,
            product_line=product_line,
            human_review_acknowledged=human_review_acknowledged,
        )
        approved, cs = self._scorer.review(prompt, response)

        ci = None
        cal_threshold = None
        uncertainty_action = None
        if self._conformal is not None:
            ci = self._conformal.predict_interval(cs.score)
            if self._uncertainty_router is not None:
                uncertainty_action = self._uncertainty_router.route(
                    self._conformal.predict(cs.score)
                ).action
        if self._calibrator is not None:
            cal_threshold = self._calibrator.adjusted_threshold

        return GuardResult(
            approved=approved and (sector_report.approved if sector_report else True),
            score=cs.score,
            coherence=cs,
            confidence_interval=ci,
            calibrated_threshold=cal_threshold,
            sector_policy_report=sector_report,
            uncertainty_action=uncertainty_action,
        )

    def _evaluate_sector_policy(
        self,
        *,
        sector_policy: str | None,
        prompt: str,
        response: str,
        evidence_refs: Iterable[str],
        numeric_evidence_refs: Iterable[str],
        policy_refs: Iterable[str],
        jurisdiction: str,
        product_line: str,
        human_review_acknowledged: bool,
    ) -> BankingPolicyReport | None:
        """Run an optional deterministic sector policy before final approval."""
        if sector_policy is None or not sector_policy.strip():
            return None
        normalised = sector_policy.strip().casefold().replace("_", "-")
        if normalised not in {"banking", "financial-services"}:
            raise ValueError(
                "sector_policy must be one of: banking, financial-services"
            )
        enforce_capability_tier("sector_policy")
        try:
            from director_ai.core.financial_services import assess_banking_response
        except ModuleNotFoundError as exc:  # pragma: no cover - advanced tier only
            raise RuntimeError(
                "sector_policy checks require the advanced tier "
                "(director_ai.core.financial_services is not installed)."
            ) from exc
        return assess_banking_response(
            prompt,
            response,
            evidence_refs=evidence_refs,
            numeric_evidence_refs=numeric_evidence_refs,
            policy_refs=policy_refs,
            jurisdiction=jurisdiction,
            product_line=product_line,
            human_review_acknowledged=human_review_acknowledged,
        )

    def check_verified(
        self,
        response: str,
        source: str,
        atomic: bool = True,
    ) -> VerificationResult:
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

    @property
    def scorer(self) -> CoherenceScorer:
        """Return the underlying coherence scorer."""
        return self._scorer

    @property
    def config(self) -> DirectorConfig:
        """Return the guard's resolved configuration."""
        return self._config
