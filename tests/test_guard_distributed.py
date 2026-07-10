# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Production guard distributed-trust mixin contracts

"""Contract tests for the guard distributed-trust module.

``director_ai._guard_distributed`` owns the multi-party and multi-model
trust surface of :class:`~director_ai.guard.ProductionGuard` (DP
retrieval and DP-RAG, federated DP calibration and evidence, secure
aggregation, Byzantine quorums, cross-model consensus, swarm coherence).
These tests pin where the methods live and the lazy-engine persistence;
the behaviour matrices stay in ``tests/test_dp_rag.py``,
``tests/test_federated_dp.py``, ``tests/test_cross_model_consensus.py``,
``tests/test_byzantine_consensus.py``, ``tests/test_shamir.py``, and
``tests/test_swarm_coherence.py``.
"""

from __future__ import annotations

from director_ai._guard_distributed import DistributedTrustMixin
from director_ai.core.consensus import ModelResponse
from director_ai.guard import ProductionGuard


class TestMixinComposition:
    def test_production_guard_composes_the_mixin(self):
        assert issubclass(ProductionGuard, DistributedTrustMixin)

    def test_distributed_methods_live_on_the_mixin_only(self):
        for name in (
            "cross_model_consensus",
            "new_swarm_monitor",
            "secure_aggregator",
            "byzantine_consensus",
            "dp_retrieval",
            "dp_rag_pipeline",
            "federated_calibration",
            "federated_dp_evidence",
        ):
            assert name in vars(DistributedTrustMixin)
            assert name not in vars(ProductionGuard)

    def test_module_exports_only_the_mixin(self):
        import director_ai._guard_distributed as module

        assert module.__all__ == ["DistributedTrustMixin"]


class TestLazyEngines:
    def test_dp_retrieval_is_built_once_and_persists(self):
        guard = ProductionGuard()
        assert guard._dp_retrieval is None
        engine = guard.dp_retrieval
        assert guard.dp_retrieval is engine

    def test_cross_model_engine_persists_without_nli(self):
        guard = ProductionGuard()
        responses = [
            ModelResponse(model_id="a", text="Paris is the capital of France."),
            ModelResponse(model_id="b", text="Paris is the capital of France."),
        ]
        guard.cross_model_consensus(responses)
        engine = guard._cross_model
        guard.cross_model_consensus(responses)
        assert guard._cross_model is engine

    def test_federated_calibration_seeds_from_the_guard_threshold(self):
        guard = ProductionGuard()
        round_obj = guard.federated_calibration()
        assert round_obj.value == guard.config.coherence_threshold
