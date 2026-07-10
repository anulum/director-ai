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
from collections.abc import Iterable, Sequence
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
from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.agent_preflight import AgentPreflightGuard
from director_ai.core.answer_bom import AnswerBOM, build_answer_bom
from director_ai.core.config import DirectorConfig
from director_ai.core.eval_trace import eval_record_from_guard, record_guard_decision
from director_ai.core.labelling_cockpit import ActiveLabellingCockpit
from director_ai.core.license import enforce_capability_tier
from director_ai.core.redactor import PIIRedactor
from director_ai.core.risk_threshold import (
    RiskAdaptiveThreshold,
    RiskFactors,
    RiskThresholdDecision,
)
from director_ai.core.scoring.verified_scorer import VerifiedScorer
from director_ai.core.types import CoherenceScore

if TYPE_CHECKING:
    # Advanced-tier (BUSL-1.1) types used only in annotations — runtime imports
    # are lazy inside the methods that need them, so the Apache core wheel does
    # not require these modules to be installed.
    from director_ai.core.calibration.runtime_governor import (
        RuntimeThresholdGovernor,
    )
    from director_ai.core.cyber_physical import (
        PhysicalConstraint,
        RobotCommandGuard,
    )
    from director_ai.core.execution_rings import ExecutionRingGate
    from director_ai.core.financial_services import BankingPolicyReport
    from director_ai.core.forecasting import (
        ForecastHistory,
        ForecastResult,
        HallucinationForecaster,
    )
    from director_ai.core.fuzzing import ContinuousFuzzer
    from director_ai.core.interpretability import HallucinationRootCauseAnalyzer
    from director_ai.core.ml_bom import MachineLearningBOM
    from director_ai.core.multimodal_guard import (
        MultimodalCheckRequest,
        MultimodalCheckResult,
        MultimodalVerifierAdapter,
    )
    from director_ai.core.neuro_symbolic import (
        CompliancePolicy,
        NeuroSymbolicComplianceEngine,
    )
    from director_ai.core.output_integrity import OutputIntegrityGuard
    from director_ai.core.output_trust import ZeroTrustOutputGuard
    from director_ai.core.rasp import RuntimeSelfProtection
    from director_ai.core.routing import EconomicDecision, HallucinationEconomics
    from director_ai.core.scoring.span_detector import (
        HallucinationSpanDetector,
        SpanDetection,
    )
    from director_ai.core.scoring.temporal_refresh import (
        RefreshReport,
        TemporalRefresher,
    )
    from director_ai.core.scoring.verified_scorer import VerificationResult
    from director_ai.core.self_healing import SelfHealingThresholdController
    from director_ai.core.temporal_consistency import TemporalConsistencyGraph
    from director_ai.core.temporal_logic import TrajectorySafetyMonitor
    from director_ai.core.threat_intel import ThreatIntelligenceMatcher

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
        self._preflight: AgentPreflightGuard | None = None
        self._risk_threshold: RiskAdaptiveThreshold | None = None
        self._labelling_cockpit: ActiveLabellingCockpit | None = None
        self._temporal_consistency: TemporalConsistencyGraph | None = None
        self._self_healing: SelfHealingThresholdController | None = None
        self._dp_retrieval = None
        self._root_cause: HallucinationRootCauseAnalyzer | None = None
        self._output_trust: ZeroTrustOutputGuard | None = None
        self._execution_rings: ExecutionRingGate | None = None
        self._output_integrity: OutputIntegrityGuard | None = None
        self._ml_bom: MachineLearningBOM | None = None
        self._rasp: RuntimeSelfProtection | None = None
        self._threat_intel: ThreatIntelligenceMatcher | None = None
        self._forecaster: HallucinationForecaster | None = None
        self._temporal_refresher: TemporalRefresher | None = None
        self._cross_model = None
        self._economics: HallucinationEconomics | None = None
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
    def preflight(self) -> AgentPreflightGuard:
        """Agent/MCP preflight guard wired to this guard's scorer.

        Provides the five seam gates (before/after tool call, before final
        answer, before handoff, before irreversible action); result plausibility
        is scored with this guard's coherence scorer.
        """
        if self._preflight is None:

            def _score(premise: str, hypothesis: str) -> float:
                return self._scorer.review(premise, hypothesis)[1].score

            self._preflight = AgentPreflightGuard(score_fn=_score)
        return self._preflight

    def risk_threshold(self, factors: RiskFactors) -> RiskThresholdDecision:
        """Compute a per-request approval threshold from a risk profile.

        Deterministically adapts the base coherence threshold up (stricter) for
        high-risk requests and down for a demonstrated high false-halt rate,
        recording every factor's contribution. The host applies the returned
        threshold; the guard does not mutate its own configured threshold.
        """
        if self._risk_threshold is None:
            from director_ai.core.risk_threshold import RiskThresholdPolicy

            self._risk_threshold = RiskAdaptiveThreshold(
                RiskThresholdPolicy(base_threshold=self._config.coherence_threshold)
            )
        return self._risk_threshold.evaluate(factors)

    @property
    def labelling_cockpit(self) -> ActiveLabellingCockpit:
        """Active-labelling cockpit at this guard's operating threshold.

        Rank items to label, measure false-halt vs missed-hallucination error,
        and recommend a threshold from reviewer-labelled outcomes.
        """
        if self._labelling_cockpit is None:
            self._labelling_cockpit = ActiveLabellingCockpit(
                threshold=self._config.coherence_threshold
            )
        return self._labelling_cockpit

    def eval_trace(
        self,
        result: GuardResult,
        *,
        model: str = "",
        tenant_id: str = "",
        domain: str = "",
        answer_id: str = "",
        emit_span: bool = True,
    ) -> dict[str, str | int | float | bool]:
        """Emit a guard decision as an OTel eval span and return its record.

        Builds the stable ``director.eval.*`` / ``gen_ai.*`` attribute record
        from the result and, when ``emit_span`` is set, opens an
        OpenTelemetry span carrying it (a no-op without the SDK). The returned
        dict is the same record, for tracers that take metadata rather than
        OTLP spans.
        """
        record = eval_record_from_guard(
            result,
            model=model,
            scorer=self._config.scorer_backend,
            tenant_id=tenant_id,
            domain=domain,
            answer_id=answer_id,
        )
        if emit_span:
            with record_guard_decision(record):
                pass
        return record

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
    def root_cause_analyzer(self) -> HallucinationRootCauseAnalyzer:
        """Prescriptive mechanistic hallucination root-cause analyzer.

        Consumes a
        :class:`~director_ai.core.interpretability.MechanisticAttributionReport`
        (from the ReDeEP attributor) and returns a tenant-safe diagnosis naming
        the dominant failure mode (parametric-knowledge override, attention
        ignoring evidence, underactive copying heads) with targeted fine-tuning
        recommendations for the Customer Model Factory.
        """
        if self._root_cause is None:
            from director_ai.core.interpretability import (
                HallucinationRootCauseAnalyzer,
            )

            self._root_cause = HallucinationRootCauseAnalyzer()
        return self._root_cause

    @property
    def output_trust(self) -> ZeroTrustOutputGuard:
        """Zero-trust output handling for untrusted model output (OWASP LLM05).

        Encodes a model output for the specific
        :class:`~director_ai.core.output_trust.OutputSink` it will enter (HTML,
        shell argument, SQL identifier, filesystem path, JSON, URL query, email
        header, log line) so one string cannot be an XSS payload in one context
        and a command-injection payload in another, and flags constructs that
        must never be executed or deserialised. Stateless; persists on the guard
        only to avoid re-instantiation.
        """
        if self._output_trust is None:
            from director_ai.core.output_trust import ZeroTrustOutputGuard

            self._output_trust = ZeroTrustOutputGuard()
        return self._output_trust

    def execution_rings(
        self, *, cooling_period_seconds: float = 86_400.0
    ) -> ExecutionRingGate:
        """Graduated human authorisation for agent actions (execution rings).

        Classifies an action into an ordered
        :class:`~director_ai.core.execution_rings.ExecutionRing` (read → write →
        delete → execute → exfiltrate) and allows it only when the human
        authorisation factors that ring demands (operator approval, cooling
        period, second operator, CISO notification) have been collected — so a
        prompt-injected agent cannot delete or exfiltrate without out-of-band
        confirmation. ``cooling_period_seconds`` defaults to 24 hours.
        """
        if self._execution_rings is None:
            from director_ai.core.execution_rings import ExecutionRingGate

            self._execution_rings = ExecutionRingGate(
                cooling_period_seconds=cooling_period_seconds
            )
        return self._execution_rings

    def output_integrity(
        self, *, signing_seed: bytes | None = None
    ) -> OutputIntegrityGuard:
        """Cryptographic integrity + non-repudiation for model outputs (ML09).

        Signs an output with a detached Ed25519 signature a third party can
        verify with only the public key, and records its digest in an append-only
        tamper-evident
        :class:`~director_ai.core.output_integrity.TamperEvidentLedger`. The
        ledger is stdlib-only and always available; signing needs the optional
        ``cryptography`` backend (``pip install director-ai[crypto]``). Supply a
        32-byte ``signing_seed`` from a secret manager for a stable identity.
        """
        if self._output_integrity is None:
            from director_ai.core.output_integrity import OutputIntegrityGuard

            self._output_integrity = OutputIntegrityGuard(signing_seed=signing_seed)
        return self._output_integrity

    @property
    def ml_bom(self) -> MachineLearningBOM:
        """Supply-chain bill of materials for the ML system (OWASP ASVS).

        Record each model, dataset, and dependency with a SHA-256 digest and
        provenance via
        :class:`~director_ai.core.ml_bom.MachineLearningBOM`, then
        :meth:`~director_ai.core.ml_bom.MachineLearningBOM.verify` the deployed
        artefacts to detect a swapped or poisoned component. The BOM carries its
        own digest so the inventory is itself tamper-evident. Persists on the
        guard so components accumulate across calls.
        """
        if self._ml_bom is None:
            from director_ai.core.ml_bom import MachineLearningBOM

            self._ml_bom = MachineLearningBOM()
        return self._ml_bom

    def _ensure_forecaster(self) -> HallucinationForecaster:
        if self._forecaster is None:
            from director_ai.core.forecasting import (
                ForecastHistory,
                HallucinationForecaster,
            )

            self._forecaster = HallucinationForecaster(history=ForecastHistory())
        return self._forecaster

    def forecast(self, prompt: str) -> ForecastResult:
        """Forecast a prompt's hallucination risk *before* generation.

        Runs one step ahead of every response-side guard: scores the incoming
        *prompt* against three signals — under-specification (ambiguity), lexical
        coverage by the facts this guard's :class:`GroundTruthStore` would
        retrieve, and the online hallucination rate of past prompts of the same
        shape — and returns a
        :class:`~director_ai.core.forecasting.ForecastResult` with a risk in
        ``[0, 1]`` plus a ``proceed`` / ``ground`` / ``human_review``
        recommendation. The forecaster (and its outcome history) persists across
        calls so the pattern signal accumulates; feed observed outcomes back with
        ``guard.forecast_history.record(prompt, hallucinated=...)``.
        """
        return self._ensure_forecaster().forecast(prompt, store=self._store)

    @property
    def forecast_history(self) -> ForecastHistory:
        """Return the forecaster's online outcome memory (created on first use)."""
        forecaster = self._ensure_forecaster()
        # _ensure_forecaster always constructs the forecaster with a history.
        assert forecaster.history is not None
        return forecaster.history

    @property
    def economics(self) -> HallucinationEconomics:
        """Cost-risk guard-action selector (created on first use).

        A :class:`~director_ai.core.routing.HallucinationEconomics` over the
        default action tiers; set ``self._economics`` directly or call with a
        custom menu for per-deployment cost models.
        """
        if self._economics is None:
            from director_ai.core.routing import HallucinationEconomics

            self._economics = HallucinationEconomics()
        return self._economics

    def guard_economics(
        self, risk: float, *, hallucination_cost: float | None = None
    ) -> EconomicDecision:
        """Pick the expected-cost-minimising guard action for a request.

        Treats guarding as an economic decision: given the request's
        hallucination *risk* in ``[0, 1]`` (for example
        ``guard.forecast(prompt).risk``) and the business cost of a hallucination
        reaching the user, returns a
        :class:`~director_ai.core.routing.EconomicDecision` naming the cheapest
        action (skip / heuristic / NLI / escalate / human review), its expected
        cost, the per-action breakdown, and the *value* — the expected loss
        guarding avoids versus doing nothing — so the guard can be justified as a
        value driver. Override the per-request stakes with *hallucination_cost*
        (e.g. higher for medical or financial domains).
        """
        return self.economics.decide(risk, hallucination_cost=hallucination_cost)

    def new_threshold_governor(
        self,
        *,
        candidate_thresholds: tuple[float, ...] | None = None,
        max_step: float = 0.05,
        auto_apply: bool = False,
        with_uncertainty_router: bool = True,
    ) -> RuntimeThresholdGovernor:
        """Build a runtime threshold governor seeded from this guard's threshold.

        Wires the per-segment
        :class:`~director_ai.core.calibration.SegmentedThresholdLearner` into a
        :class:`~director_ai.core.calibration.runtime_governor.RuntimeThresholdGovernor`
        that applies learned thresholds to the live runtime under a
        change-management overlay: bounded stepping (``max_step``), human-approval
        gating (unless ``auto_apply``), and an audit history. ``current_threshold``
        is this guard's ``coherence_threshold``; the candidate grid defaults to
        ``0.1 … 0.9``. With ``with_uncertainty_router`` (default) the governor's
        ``effective_threshold`` tightens on a wide/unreliable conformal interval.
        Returns a fresh governor so each deployment keeps its own live state and
        audit trail.
        """
        from director_ai.core.calibration.runtime_governor import (
            RuntimeThresholdGovernor,
        )
        from director_ai.core.calibration.segmented_threshold import (
            SegmentedThresholdLearner,
        )

        current = self._config.coherence_threshold
        if candidate_thresholds is None:
            candidate_thresholds = tuple(round(0.1 * i, 1) for i in range(1, 10))
        learner = SegmentedThresholdLearner(
            candidate_thresholds=candidate_thresholds, current_threshold=current
        )
        router = None
        if with_uncertainty_router:
            from director_ai.core.routing import UncertaintyRouter

            router = UncertaintyRouter()
        return RuntimeThresholdGovernor(
            learner=learner,
            current_threshold=current,
            max_step=max_step,
            auto_apply=auto_apply,
            uncertainty_router=router,
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

    @property
    def rasp(self) -> RuntimeSelfProtection:
        """Runtime application self-protection from behavioural anomalies.

        The last line of defence once input filters and guardrails are bypassed:
        feed per-request behavioural metrics (request rate, payload size, halt
        rate) to
        :meth:`~director_ai.core.rasp.RuntimeSelfProtection.observe` and read back
        a tenant-safe ok/watch/alert verdict scored by a dependency-free robust
        (median/MAD) detector. Persists across calls so each metric's baseline
        accumulates; the host decides whether to shed load or block.
        """
        if self._rasp is None:
            from director_ai.core.rasp import RuntimeSelfProtection

            self._rasp = RuntimeSelfProtection()
        return self._rasp

    def continuous_fuzzer(self, *, seed: int = 0) -> ContinuousFuzzer:
        """Mutation-based fuzzer that hunts for guard bypasses.

        Where the static adversarial suite checks a fixed list, the returned
        :class:`~director_ai.core.fuzzing.ContinuousFuzzer` mutates a seed corpus
        of attacks (homoglyph, zero-width, leetspeak, delimiter, …) against a
        ``predicate`` you supply — ``True`` = the guard flags it — and reports
        every obfuscation that slipped through plus any seed the guard missed
        outright. The ``seed`` makes a found bypass replayable as a regression.
        """
        from director_ai.core.fuzzing import ContinuousFuzzer

        return ContinuousFuzzer(seed=seed)

    @property
    def threat_intel(self) -> ThreatIntelligenceMatcher:
        """Threat-intelligence IOC matching with attribution (STIX-aligned).

        Register indicators directly or import them from a STIX 2.1 feed with
        :func:`~director_ai.core.threat_intel.from_stix_bundle`, then
        :meth:`~director_ai.core.threat_intel.ThreatIntelligenceMatcher.match`
        prompts/responses against them to get every triggered indicator with its
        attribution and severity — block *and* report "matches the APT29 kit".
        Persists across calls so the indicator set accumulates.
        """
        if self._threat_intel is None:
            from director_ai.core.threat_intel import ThreatIntelligenceMatcher

            self._threat_intel = ThreatIntelligenceMatcher()
        return self._threat_intel

    def robot_command_guard(
        self, constraints: Sequence[PhysicalConstraint] = (), **kwargs: Any
    ) -> RobotCommandGuard:
        """Build an embodied-AI guard for an LLM-planned robot command sequence.

        Verifies a whole plan before execution against per-action physical
        constraints plus temporal caps (bounded step displacement and path
        length). Warn-only by default; pass ``high_risk_enabled=True`` to block an
        unsafe plan. Returns a fresh
        :class:`~director_ai.core.cyber_physical.RobotCommandGuard`; pass
        ``model``, ``high_risk_enabled``, ``max_step_displacement``, or
        ``max_path_length`` via ``kwargs``.
        """
        from director_ai.core.cyber_physical import RobotCommandGuard

        return RobotCommandGuard(constraints, **kwargs)

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

    @property
    def self_healing(self) -> SelfHealingThresholdController:
        """Self-healing threshold controller seeded at the configured threshold.

        Closes the calibration loop safely: feed labelled outcomes
        (:class:`~director_ai.core.self_healing.LabelledOutcome`), call
        ``propose()`` to deploy a better threshold only when it beats the current
        one on a held-out split, and ``evaluate_regression()`` to auto-roll-back a
        deployed update that later regresses. Persists across calls on this guard
        so observations accumulate; every change is audited. The host applies the
        controller's ``threshold`` — the guard does not mutate its own config.
        """
        if self._self_healing is None:
            from director_ai.core.self_healing import SelfHealingThresholdController

            self._self_healing = SelfHealingThresholdController(
                self._config.coherence_threshold
            )
        return self._self_healing

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

    def answer_bom(
        self,
        result: GuardResult,
        *,
        model: str = "",
        tenant: str = "",
        answer_id: str | None = None,
        freshness: str = "",
        policy_refs: Iterable[str] = (),
    ) -> AnswerBOM:
        """Build an Answer Bill of Materials from a :class:`GuardResult`.

        Records the model/scorer/threshold header and a per-claim evidence
        record built from the scorer's claim-level provenance. The threshold is
        the calibrated threshold when calibration is enabled, otherwise the
        configured coherence threshold.
        """
        threshold = (
            result.calibrated_threshold
            if result.calibrated_threshold is not None
            else self._config.coherence_threshold
        )
        return build_answer_bom(
            result.coherence,
            model=model,
            scorer=self._config.scorer_backend,
            threshold=threshold,
            tenant=tenant,
            answer_id=answer_id,
            freshness=freshness,
            policy_refs=tuple(policy_refs),
        )
