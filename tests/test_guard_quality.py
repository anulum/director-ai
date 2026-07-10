# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Production guard decision-quality mixin contracts

"""Contract tests for the guard decision-quality module.

``director_ai._guard_quality`` owns the self-measurement surface of
:class:`~director_ai.guard.ProductionGuard` (risk-adaptive thresholds,
labelling cockpit, eval traces, forecasting, guard economics, threshold
governor, self-healing, root cause, Answer BOM). These tests pin where
the methods live, the lazy-engine persistence, and the threshold
seeding from the composing guard's config; the behaviour matrices stay
in ``tests/test_risk_threshold.py``, ``tests/test_labelling_cockpit.py``,
``tests/test_eval_trace.py``, ``tests/test_hallucination_forecaster.py``,
``tests/test_hallucination_economics.py``,
``tests/test_runtime_governor.py``, ``tests/test_self_healing.py``,
``tests/test_root_cause.py``, and ``tests/test_answer_bom.py``.
"""

from __future__ import annotations

from director_ai._guard_quality import DecisionQualityMixin
from director_ai.core.types import CoherenceScore
from director_ai.guard import GuardResult, ProductionGuard


class TestMixinComposition:
    def test_production_guard_composes_the_mixin(self):
        assert issubclass(ProductionGuard, DecisionQualityMixin)

    def test_quality_methods_live_on_the_mixin_only(self):
        for name in (
            "risk_threshold",
            "labelling_cockpit",
            "eval_trace",
            "_ensure_forecaster",
            "forecast",
            "forecast_history",
            "economics",
            "guard_economics",
            "new_threshold_governor",
            "self_healing",
            "root_cause_analyzer",
            "answer_bom",
        ):
            assert name in vars(DecisionQualityMixin)
            assert name not in vars(ProductionGuard)

    def test_module_exports_only_the_mixin(self):
        import director_ai._guard_quality as module

        assert module.__all__ == ["DecisionQualityMixin"]


class TestConfigSeeding:
    def test_labelling_cockpit_seeds_from_the_guard_threshold(self):
        guard = ProductionGuard()
        cockpit = guard.labelling_cockpit
        assert guard.labelling_cockpit is cockpit
        assert cockpit.threshold == guard.config.coherence_threshold

    def test_self_healing_seeds_from_the_guard_threshold(self):
        guard = ProductionGuard()
        controller = guard.self_healing
        assert guard.self_healing is controller
        assert controller.threshold == guard.config.coherence_threshold


class TestLazyEngines:
    def test_forecaster_is_shared_between_forecast_and_history(self):
        guard = ProductionGuard()
        assert guard._forecaster is None
        history = guard.forecast_history
        guard.forecast("What is the maximum dose?")
        assert guard.forecast_history is history

    def test_eval_trace_reads_the_scorer_backend_from_config(self):
        guard = ProductionGuard()
        result = GuardResult(
            approved=True,
            score=0.9,
            coherence=CoherenceScore(
                score=0.9, approved=True, h_logical=0.0, h_factual=0.0
            ),
        )
        record = guard.eval_trace(result, emit_span=False)
        assert record["director.eval.scorer"] == guard.config.scorer_backend
