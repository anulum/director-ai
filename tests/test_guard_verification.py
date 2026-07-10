# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Production guard extended-verification mixin contracts

"""Contract tests for the guard extended-verification module.

``director_ai._guard_verification`` owns the verification modalities of
:class:`~director_ai.guard.ProductionGuard` beyond the core coherence
check (temporal consistency and refresh, span detection, multimodal
evidence, SMT compliance, LTL trajectory safety). These tests pin where
the methods live, the lazy-verifier persistence, and the opt-in gates;
the behaviour matrices stay in ``tests/test_temporal_consistency.py``,
``tests/test_temporal_refresh.py``, ``tests/test_span_detector.py``,
``tests/test_guard_multimodal.py``, ``tests/test_neuro_symbolic.py``,
and ``tests/test_temporal_logic.py``.
"""

from __future__ import annotations

import pytest

from director_ai._guard_verification import ExtendedVerificationMixin
from director_ai.guard import ProductionGuard


class TestMixinComposition:
    def test_production_guard_composes_the_mixin(self):
        assert issubclass(ProductionGuard, ExtendedVerificationMixin)

    def test_verification_methods_live_on_the_mixin_only(self):
        for name in (
            "temporal_consistency",
            "temporal_refresher",
            "refresh_temporal",
            "multimodal_adapter",
            "span_detector",
            "detect_spans",
            "check_multimodal",
            "compliance_engine",
            "trajectory_monitor",
        ):
            assert name in vars(ExtendedVerificationMixin)
            assert name not in vars(ProductionGuard)

    def test_module_exports_only_the_mixin(self):
        import director_ai._guard_verification as module

        assert module.__all__ == ["ExtendedVerificationMixin"]


class TestLazyVerifiers:
    def test_temporal_graph_and_refresher_persist(self):
        guard = ProductionGuard()
        assert guard._temporal_consistency is None
        assert guard._temporal_refresher is None
        graph = guard.temporal_consistency
        refresher = guard.temporal_refresher
        assert guard.temporal_consistency is graph
        assert guard.temporal_refresher is refresher

    def test_trajectory_monitors_are_independent(self):
        guard = ProductionGuard()
        assert guard.trajectory_monitor() is not guard.trajectory_monitor()


class TestOptInGates:
    def test_multimodal_adapter_requires_enabled_modalities(self):
        guard = ProductionGuard()
        assert not guard.config.multimodal_enabled_modalities
        with pytest.raises(RuntimeError, match="multimodal guard is disabled"):
            _ = guard.multimodal_adapter

    def test_span_detector_requires_the_config_flag(self):
        guard = ProductionGuard()
        assert not guard.config.span_detection_enabled
        with pytest.raises(RuntimeError, match="span detection is disabled"):
            _ = guard.span_detector
