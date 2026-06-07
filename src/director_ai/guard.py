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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.agent_preflight import AgentPreflightGuard
from director_ai.core.answer_bom import AnswerBOM, build_answer_bom
from director_ai.core.canary import (
    CanaryDetector,
    CanaryFact,
    CanaryRegistry,
    CanarySignal,
)
from director_ai.core.config import DirectorConfig
from director_ai.core.eval_trace import eval_record_from_guard, record_guard_decision
from director_ai.core.financial_services import (
    BankingPolicyReport,
    assess_banking_response,
)
from director_ai.core.labelling_cockpit import ActiveLabellingCockpit
from director_ai.core.risk_threshold import (
    RiskAdaptiveThreshold,
    RiskFactors,
    RiskThresholdDecision,
)
from director_ai.core.scoring.verified_scorer import VerifiedScorer
from director_ai.core.streaming_repair import RepairResult
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
    sector_policy_report: BankingPolicyReport | None = None
    uncertainty_action: str | None = None


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
        # Calibration pieces are lazily installed by
        # :meth:`enable_calibration`; declare them as optionals
        # up-front so the later assignments do not narrow.
        self._calibrator: Any = None
        self._conformal: Any = None
        self._feedback: Any = None
        self._uncertainty_router: Any = None
        self._injection_detector: InjectionDetector | None = None
        self._canary_registry: CanaryRegistry | None = None
        self._canary_detector: CanaryDetector | None = None
        self._preflight: AgentPreflightGuard | None = None
        self._risk_threshold: RiskAdaptiveThreshold | None = None
        self._labelling_cockpit: ActiveLabellingCockpit | None = None
        self._temporal_consistency: object | None = None
        self._self_healing: object | None = None

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
                require_model_backed_nli=getattr(
                    cfg,
                    "injection_require_model_backed_nli",
                    False,
                ),
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

    def repair_stream(
        self,
        prompt: str,
        response: str,
        *,
        tenant_id: str = "",
        request_id: str = "",
        rewrite_fn: Callable[[str, list[str]], str] | None = None,
        threshold: float | None = None,
    ) -> RepairResult:
        """Repair unsupported clauses in a generated response.

        Turns a coherence halt into a corrective pass: each clause is scored
        against the knowledge base, and an unsupported clause is rewritten from
        retrieved corrective evidence (when ``rewrite_fn`` is supplied and
        evidence is found) or redacted, leaving the supported clauses intact.
        Returns a :class:`RepairResult` with the corrected text, per-clause
        actions, and a tenant-safe repair event per fix.
        """
        from director_ai.core.streaming_repair import StreamingRepairer

        def _score_clause(clause: str) -> float:
            return self._scorer.review(prompt, clause, tenant_id=tenant_id)[1].score

        def _retrieve(clause: str) -> list[Any]:
            getter = getattr(self._store, "retrieve_context_with_chunks", None)
            if getter is None:
                return []
            return list(getter(clause, tenant_id=tenant_id))

        repairer = StreamingRepairer(
            _score_clause,
            threshold=(
                threshold if threshold is not None else self._config.coherence_threshold
            ),
            retrieve_fn=_retrieve,
            rewrite_fn=rewrite_fn,
        )
        return repairer.repair(response, tenant_id=tenant_id, request_id=request_id)

    def _ensure_canary(self) -> CanaryDetector:
        if self._canary_detector is None:
            self._canary_registry = CanaryRegistry()
            self._canary_detector = CanaryDetector(
                self._canary_registry,
                alert=lambda s: logger.warning(
                    "canary tripped: id=%s tenant=%s signal=%s",
                    s.canary_id,
                    s.tenant_id,
                    s.signal,
                ),
            )
        return self._canary_detector

    def plant_canary(
        self,
        tenant_id: str,
        *,
        template: str = "Internal reference marker {token}: do not disclose.",
        token: str | None = None,
    ) -> CanaryFact:
        """Mint a tenant-scoped canary, plant it in the KB, and return it.

        The canary text is added to the knowledge base so retrieval can surface
        it under attack; its sentinel token must never appear in a legitimate
        answer. Detect trips with :meth:`scan_canaries`.
        """
        self._ensure_canary()
        assert self._canary_registry is not None
        fact = self._canary_registry.mint(tenant_id, template=template, token=token)
        self._store.add(fact.canary_id, fact.text)
        return fact

    def scan_canaries(
        self,
        response: str,
        tenant_id: str,
        *,
        evidence: Iterable[Any] = (),
    ) -> list[CanarySignal]:
        """Scan a response (and optional evidence chunks) for tripped canaries.

        Returns a :class:`CanarySignal` for each canary token found in the
        response (leakage) and each canary chunk present in ``evidence``
        (citation).
        """
        detector = self._ensure_canary()
        return detector.scan(response, tenant_id, evidence=list(evidence))

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
    def temporal_consistency(self):
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

    def compliance_engine(self, policy):
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
    def self_healing(self):
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

    def trajectory_monitor(self, specs: dict[str, object] | None = None):
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
        return self._scorer

    @property
    def config(self) -> DirectorConfig:
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
