# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Production guard canary mixin contracts

"""Contract tests for the guard canary-operations module.

``director_ai._guard_canary`` owns the canary surface of
:class:`~director_ai.guard.ProductionGuard` (lazy registry/detector
build, planting, scanning). These tests pin where the methods live, the
lazy-build idempotence, and the alert logging channel; the behaviour
matrix (minting, tenant isolation, leakage/citation signals) stays in
``tests/test_canary.py``.
"""

from __future__ import annotations

import logging

import director_ai.guard as guard_module
from director_ai._guard_canary import CanaryOperationsMixin
from director_ai.guard import ProductionGuard


class TestMixinComposition:
    def test_production_guard_composes_the_mixin(self):
        assert issubclass(ProductionGuard, CanaryOperationsMixin)

    def test_canary_methods_live_on_the_mixin_only(self):
        for name in ("_ensure_canary", "plant_canary", "scan_canaries"):
            assert name in vars(CanaryOperationsMixin)
            assert name not in vars(ProductionGuard)

    def test_module_exports_only_the_mixin(self):
        import director_ai._guard_canary as module

        assert module.__all__ == ["CanaryOperationsMixin"]

    def test_facade_no_longer_imports_the_canary_types(self):
        for name in ("CanaryDetector", "CanaryFact", "CanaryRegistry"):
            assert not hasattr(guard_module, name)


class TestLazyBuild:
    def test_ensure_canary_is_idempotent(self):
        guard = ProductionGuard()
        assert guard._canary_detector is None
        detector = guard._ensure_canary()
        assert guard._canary_registry is not None
        assert guard._ensure_canary() is detector

    def test_plant_before_scan_shares_one_registry(self):
        guard = ProductionGuard()
        fact = guard.plant_canary("acme", token="CANARY-CONTRACT")
        signals = guard.scan_canaries(f"leak {fact.token}", "acme")
        assert [s.canary_id for s in signals] == [fact.canary_id]


class TestAlertLogging:
    def test_trip_alert_logs_on_the_guard_logger(self, caplog):
        guard = ProductionGuard()
        fact = guard.plant_canary("acme", token="CANARY-ALERT")
        with caplog.at_level(logging.WARNING, logger="DirectorAI.Guard"):
            guard.scan_canaries(f"oops {fact.token}", "acme")
        assert "canary tripped" in caplog.text
        assert fact.canary_id in caplog.text
