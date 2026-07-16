# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Production Guard Decision Quality
"""Decision-quality and calibration-lifecycle surface of the production guard.

:class:`DecisionQualityMixin` carries the capabilities of
:class:`~director_ai.guard.ProductionGuard` that measure, price, and
improve the guard's own decisions: risk-adaptive thresholds, the active
labelling cockpit, OTel eval traces, pre-generation hallucination
forecasting, guard-action economics, the runtime threshold governor,
the self-healing threshold controller, root-cause analysis, and the
Answer Bill of Materials. Stateful engines are built lazily on first
use and persist on the guard; the advanced-tier modules are imported
inside the methods so the Apache core wheel does not require them.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from director_ai.core.risk_threshold import (
    RiskAdaptiveThreshold,
    RiskFactors,
    RiskThresholdDecision,
)

if TYPE_CHECKING:
    from director_ai.core import GroundTruthStore
    from director_ai.core.answer_bom import AnswerBOM
    from director_ai.core.calibration.runtime_governor import (
        RuntimeThresholdGovernor,
    )
    from director_ai.core.config import DirectorConfig
    from director_ai.core.forecasting import (
        ForecastHistory,
        ForecastResult,
        HallucinationForecaster,
    )
    from director_ai.core.interpretability import HallucinationRootCauseAnalyzer
    from director_ai.core.labelling_cockpit import ActiveLabellingCockpit
    from director_ai.core.routing import EconomicDecision, HallucinationEconomics
    from director_ai.core.self_healing import SelfHealingThresholdController
    from director_ai.guard import GuardResult

__all__ = ["DecisionQualityMixin"]


class DecisionQualityMixin:
    """Risk thresholds, labelling, eval traces, forecasting, economics, healing.

    All state is initialised by :class:`~director_ai.guard.ProductionGuard`'s
    ``__init__``; the configuration and knowledge base come from the composing
    guard through the contracts declared below.
    """

    _risk_threshold: RiskAdaptiveThreshold | None
    _labelling_cockpit: ActiveLabellingCockpit | None
    _forecaster: HallucinationForecaster | None
    _economics: HallucinationEconomics | None
    _self_healing: SelfHealingThresholdController | None
    _root_cause: HallucinationRootCauseAnalyzer | None

    if TYPE_CHECKING:
        # Provided by the composing ProductionGuard.
        _config: DirectorConfig
        _store: GroundTruthStore

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
            from director_ai.core.labelling_cockpit import ActiveLabellingCockpit

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
        from director_ai.core.eval_trace import (
            eval_record_from_guard,
            record_guard_decision,
        )

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
        from director_ai.core.answer_bom import build_answer_bom

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
