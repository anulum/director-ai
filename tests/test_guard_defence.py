# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Production guard response-defence mixin contracts

"""Contract tests for the guard response-defence module.

``director_ai._guard_defence`` owns the response-side defences of
:class:`~director_ai.guard.ProductionGuard` (injection detection,
moderation, the firewall pass, tool verification, streaming repair) and
the :class:`FirewallDecision` dataclass. These tests pin where the
methods and the dataclass live, the facade re-export, and the lazy
moderation build; the behaviour matrices stay in
``tests/test_guard_firewall.py``, ``tests/test_injection_integration.py``,
and ``tests/test_streaming_repair.py``.
"""

from __future__ import annotations

import director_ai._guard_defence as guard_defence
import director_ai.guard as guard_module
from director_ai._guard_defence import ResponseDefenceMixin
from director_ai.guard import FirewallDecision, ProductionGuard


class TestMixinComposition:
    def test_production_guard_composes_the_mixin(self):
        assert issubclass(ProductionGuard, ResponseDefenceMixin)

    def test_defence_methods_live_on_the_mixin_only(self):
        for name in (
            "check_injection",
            "set_moderation_detectors",
            "_ensure_moderation",
            "firewall",
            "verify_tool",
            "repair_stream",
        ):
            assert name in vars(ResponseDefenceMixin)
            assert name not in vars(ProductionGuard)

    def test_module_exports_the_decision_and_the_mixin(self):
        assert guard_defence.__all__ == ["FirewallDecision", "ResponseDefenceMixin"]


class TestFirewallDecisionPlacement:
    def test_facade_re_exports_the_same_dataclass(self):
        assert FirewallDecision is guard_defence.FirewallDecision
        assert guard_module.FirewallDecision is guard_defence.FirewallDecision

    def test_dataclass_home_module_is_the_defence_module(self):
        assert FirewallDecision.__module__ == "director_ai._guard_defence"


class TestLazyModeration:
    def test_default_moderation_pair_is_built_lazily(self):
        guard = ProductionGuard()
        assert guard._moderation_detectors is None
        detectors = guard._ensure_moderation()
        assert [type(d).__name__ for d in detectors] == [
            "RegexPIIDetector",
            "KeywordToxicityDetector",
        ]

    def test_set_moderation_detectors_overrides_the_default(self):
        class _Detector:
            def analyse(self, text):
                raise NotImplementedError

        guard = ProductionGuard()
        override = _Detector()
        guard.set_moderation_detectors([override])
        assert guard._ensure_moderation() == [override]
