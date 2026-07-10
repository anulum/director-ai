# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Production guard runtime-hardening mixin contracts

"""Contract tests for the guard runtime-hardening module.

``director_ai._guard_hardening`` owns the security-hardening surface of
:class:`~director_ai.guard.ProductionGuard` (preflight seam gates,
zero-trust output encoding, execution rings, output integrity, ML-BOM,
RASP, the bypass fuzzer, threat intelligence, robot command guard).
These tests pin where the methods live, the lazy-guard persistence, and
the preflight wiring to the composing guard's scorer; the behaviour
matrices stay in ``tests/test_agent_preflight.py``,
``tests/test_output_trust.py``, ``tests/test_execution_rings.py``,
``tests/test_output_integrity.py``, ``tests/test_ml_bom.py``,
``tests/test_rasp.py``, ``tests/test_fuzzing.py``,
``tests/test_threat_intel.py``, and ``tests/test_robot_command_guard.py``.
"""

from __future__ import annotations

from director_ai._guard_hardening import RuntimeHardeningMixin
from director_ai.guard import ProductionGuard


class TestMixinComposition:
    def test_production_guard_composes_the_mixin(self):
        assert issubclass(ProductionGuard, RuntimeHardeningMixin)

    def test_hardening_methods_live_on_the_mixin_only(self):
        for name in (
            "preflight",
            "output_trust",
            "execution_rings",
            "output_integrity",
            "ml_bom",
            "rasp",
            "continuous_fuzzer",
            "threat_intel",
            "robot_command_guard",
        ):
            assert name in vars(RuntimeHardeningMixin)
            assert name not in vars(ProductionGuard)

    def test_module_exports_only_the_mixin(self):
        import director_ai._guard_hardening as module

        assert module.__all__ == ["RuntimeHardeningMixin"]


class TestLazyGuards:
    def test_preflight_persists_and_scores_with_the_guard_scorer(self):
        guard = ProductionGuard()
        guard.load_facts({"fact": "The dose is 400mg."})
        preflight = guard.preflight
        assert guard.preflight is preflight
        # The score_fn closure must reach the composing guard's scorer.
        score = preflight._score_fn("The dose is 400mg.", "The dose is 400mg.")
        assert 0.0 <= score <= 1.0

    def test_stateful_guards_are_built_once(self):
        guard = ProductionGuard()
        assert guard._output_trust is None
        assert guard._rasp is None
        assert guard._threat_intel is None
        assert guard.output_trust is guard.output_trust
        assert guard.ml_bom is guard.ml_bom
        assert guard.rasp is guard.rasp
        assert guard.threat_intel is guard.threat_intel
        assert guard.execution_rings() is guard.execution_rings()
        assert guard.output_integrity() is guard.output_integrity()
