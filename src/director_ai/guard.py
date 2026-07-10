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
    from director_ai.core.multimodal_guard import (
        MultimodalCheckRequest,
        MultimodalCheckResult,
        MultimodalVerifierAdapter,
    )
    from director_ai.core.neuro_symbolic import (
        CompliancePolicy,
        NeuroSymbolicComplianceEngine,
    )
    from director_ai.core.scoring.span_detector import (
        HallucinationSpanDetector,
        SpanDetection,
    )
    from director_ai.core.scoring.temporal_refresh import (
        RefreshReport,
        TemporalRefresher,
    )
    from director_ai.core.scoring.verified_scorer import VerificationResult
    from director_ai.core.temporal_consistency import TemporalConsistencyGraph
    from director_ai.core.temporal_logic import TrajectorySafetyMonitor

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
        self._temporal_consistency: TemporalConsistencyGraph | None = None
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
        self._temporal_refresher: TemporalRefresher | None = None
        self._cross_model = None
        self._economics = None
        self._multimodal: MultimodalVerifierAdapter | None = None
        self._span_detector: HallucinationSpanDetector | None = None

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
    def temporal_consistency(self) -> TemporalConsistencyGraph:
        """Cross-session structured-claim temporal consistency graph.

        Persists across calls on this guard so claims recorded in one session are
        checked against earlier sessions — the formal answer to "the system said
        X yesterday and ¬X today". Record
        :class:`~director_ai.core.temporal_consistency.TemporalClaim` objects and
        read contradictions back. For single-valued predicates (one diagnosis per
        patient) construct
        :class:`~director_ai.core.temporal_consistency.TemporalConsistencyGraph`
        directly with ``functional_predicates``.
        """
        if self._temporal_consistency is None:
            from director_ai.core.temporal_consistency import (
                TemporalConsistencyGraph,
            )

            self._temporal_consistency = TemporalConsistencyGraph()
        return self._temporal_consistency

    @property
    def temporal_refresher(self) -> TemporalRefresher:
        """Live web-search refresher for stale temporal claims.

        Where :attr:`temporal_consistency` checks claims against this system's own
        history, the refresher checks them against the live web: it scores a
        response for staleness, then for each flagged claim (a named office-holder,
        a statistic, a record) queries a search provider and reports whether
        current sources still support it. Persists across calls. By default it runs
        the dependency-free lexical triage; inject an NLI
        :class:`~director_ai.core.scoring.contradiction.ContradictionScorer` via
        ``TemporalRefresher(nli=...)`` for adjudicated ``contradicted`` verdicts.
        """
        if self._temporal_refresher is None:
            from director_ai.core.scoring.temporal_refresh import TemporalRefresher

            self._temporal_refresher = TemporalRefresher()
        return self._temporal_refresher

    def refresh_temporal(
        self,
        text: str,
        *,
        source_timestamp: float | None = None,
        max_age_days: float = 180,
        domain: str = "",
    ) -> RefreshReport:
        """Score *text* for staleness and live-refresh its stale claims.

        Convenience wrapper over :attr:`temporal_refresher` that returns a
        :class:`~director_ai.core.scoring.temporal_refresh.RefreshReport`.
        """
        return self.temporal_refresher.refresh_response(
            text,
            source_timestamp=source_timestamp,
            max_age_days=max_age_days,
            domain=domain,
        )

    @property
    def multimodal_adapter(self) -> MultimodalVerifierAdapter:
        """In-process multimodal verifier (created on first use).

        Builds the dependency-free hash-bag
        :class:`~director_ai.core.multimodal_guard.MultimodalVerifierAdapter` from
        the guard's ``multimodal_*`` config. Requires
        ``multimodal_enabled_modalities`` to be set (the guard is opt-in); raises
        otherwise. Pass a torch/CLIP-backed adapter to
        :meth:`check_multimodal` for semantic verification.
        """
        if self._multimodal is None:
            cfg = self._config
            if not cfg.multimodal_enabled_modalities:
                raise RuntimeError(
                    "multimodal guard is disabled; set "
                    "multimodal_enabled_modalities in the config to enable it"
                )
            if cfg.multimodal_backend == "clip":
                from director_ai.core.multimodal_guard import build_clip_adapter

                self._multimodal = build_clip_adapter(
                    enabled_modalities=cfg.multimodal_enabled_modalities,
                    benchmarked_modalities=cfg.multimodal_benchmarked_modalities,
                    model_name=cfg.multimodal_clip_model,
                    pretrained=cfg.multimodal_clip_pretrained,
                    device=cfg.multimodal_clip_device,
                    text_dim=cfg.multimodal_embedding_dim,
                    hallucination_threshold=cfg.multimodal_hallucination_threshold,
                    consistency_threshold=cfg.multimodal_consistency_threshold,
                    temporal_alpha=cfg.multimodal_temporal_alpha,
                    temporal_floor=cfg.multimodal_temporal_floor,
                    grounding_floor=cfg.multimodal_grounding_floor,
                    grounding_allow_threshold=cfg.multimodal_grounding_allow_threshold,
                )
            elif cfg.multimodal_backend == "hashbag":
                from director_ai.core.multimodal_guard import build_hashbag_adapter

                self._multimodal = build_hashbag_adapter(
                    enabled_modalities=cfg.multimodal_enabled_modalities,
                    benchmarked_modalities=cfg.multimodal_benchmarked_modalities,
                    dim=cfg.multimodal_embedding_dim,
                    hallucination_threshold=cfg.multimodal_hallucination_threshold,
                    consistency_threshold=cfg.multimodal_consistency_threshold,
                    temporal_alpha=cfg.multimodal_temporal_alpha,
                    temporal_floor=cfg.multimodal_temporal_floor,
                    grounding_floor=cfg.multimodal_grounding_floor,
                    grounding_allow_threshold=cfg.multimodal_grounding_allow_threshold,
                )
            else:
                raise RuntimeError(
                    f"unknown multimodal_backend {cfg.multimodal_backend!r}; "
                    "expected 'hashbag' or 'clip'"
                )
        return self._multimodal

    @property
    def span_detector(self) -> HallucinationSpanDetector:
        """Token-level hallucinated-span detector (created on first use).

        Loads the ModernBERT token classifier named by ``span_model`` and flags
        the unsupported spans inside a RAG response — the span-level signal the
        response/claim-level scorer cannot isolate. Opt-in: requires
        ``span_detection_enabled`` in the config; raises otherwise.
        """
        if self._span_detector is None:
            cfg = self._config
            if not cfg.span_detection_enabled:
                raise RuntimeError(
                    "span detection is disabled; set span_detection_enabled in "
                    "the config to enable it"
                )
            from director_ai.core.scoring.span_detector import (
                HallucinationSpanDetector,
            )

            self._span_detector = HallucinationSpanDetector.from_pretrained(
                cfg.span_model,
                revision=cfg.span_model_revision or None,
                device=cfg.span_device,
                token_threshold=cfg.span_token_threshold,
                min_tokens=cfg.span_min_tokens,
                max_length=cfg.span_max_length,
            )
        return self._span_detector

    def detect_spans(self, context: str, response: str) -> SpanDetection:
        """Flag the hallucinated spans of *response* against *context*.

        Returns a
        :class:`~director_ai.core.scoring.span_detector.SpanDetection` with the
        character spans the token detector judged unsupported. Requires
        ``span_detection_enabled``.
        """
        return self.span_detector.detect(context, response)

    def check_multimodal(
        self,
        request: MultimodalCheckRequest,
        *,
        adapter: MultimodalVerifierAdapter | None = None,
    ) -> MultimodalCheckResult:
        """Verify a text claim against paired image / audio / video evidence.

        The in-process counterpart of the ``/v1/multimodal/check`` endpoint:
        scores a
        :class:`~director_ai.core.multimodal_guard.MultimodalCheckRequest` with
        the configured (or supplied) adapter and returns a tenant-safe
        :class:`~director_ai.core.multimodal_guard.MultimodalCheckResult` carrying
        the shared allow/warn/halt decision. Modalities enabled but not in
        ``multimodal_benchmarked_modalities`` resolve to ``warn`` rather than a
        scored decision, so an unbenchmarked modality never silently passes.
        """
        from director_ai.core.guard_control import RiskEnvelope

        cfg = self._config
        used = adapter if adapter is not None else self.multimodal_adapter
        risk_envelope = RiskEnvelope(
            action_category="multimodal",
            reversibility="reversible",
            domain="general",
            calibrated_threshold=getattr(cfg, "multimodal_calibrated_threshold", 0.5),
            no_go_threshold=getattr(cfg, "multimodal_no_go_threshold", 0.9),
        )
        policy_id = getattr(cfg, "multimodal_policy_id", "multimodal-default")
        return used.check(request, risk_envelope=risk_envelope, policy_id=policy_id)

    def compliance_engine(
        self, policy: CompliancePolicy
    ) -> NeuroSymbolicComplianceEngine:
        """Build a neuro-symbolic SMT compliance engine for ``policy``.

        ``policy`` is a
        :class:`~director_ai.core.neuro_symbolic.CompliancePolicy`. The engine
        checks structured output facts against the policy with Z3, returning a
        per-constraint verdict with counterexamples, and can cross-check that two
        formalisations are equivalent. Requires the ``[formal]`` extra (z3); the
        engine raises a clear error if z3 is unavailable.
        """
        from director_ai.core.neuro_symbolic import NeuroSymbolicComplianceEngine

        return NeuroSymbolicComplianceEngine(policy)

    def trajectory_monitor(
        self, specs: dict[str, object] | None = None
    ) -> TrajectorySafetyMonitor:
        """Return a fresh LTL safety monitor for one agent trajectory.

        Runs the built-in agent-safety specifications (tool calls eventually
        verified, handoffs coherence-checked, no output after an injection, fact
        claims eventually grounded) — the formal reading of EU AI Act Article 15
        "continuous monitoring". Feed one
        :class:`~director_ai.core.temporal_logic.StepObservation` per trajectory
        step; pass ``specs`` to override or extend the default set. Each call
        returns an independent monitor, so concurrent trajectories do not share
        state.
        """
        from director_ai.core.temporal_logic import TrajectorySafetyMonitor

        return TrajectorySafetyMonitor(specs)

    @property
    def scorer(self) -> CoherenceScorer:
        """Return the underlying coherence scorer."""
        return self._scorer

    @property
    def config(self) -> DirectorConfig:
        """Return the guard's resolved configuration."""
        return self._config
