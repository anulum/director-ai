# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Production Guard Distributed Trust
"""Distributed-trust surface of the production guard.

:class:`DistributedTrustMixin` carries the capabilities of
:class:`~director_ai.guard.ProductionGuard` that establish trust across
several parties or several models: differentially private retrieval and
the DP-RAG pipeline, federated DP calibration with its formal evidence,
secure multi-party aggregation, Byzantine fault-tolerant verifier
quorums, cross-model consensus, and the multi-agent swarm-coherence
monitor. Stateful engines are built lazily on first use and persist on
the guard; the advanced-tier modules are imported inside the methods so
the Apache core wheel does not require them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from director_ai.core.config import DirectorConfig
    from director_ai.core.consensus import (
        ConsensusResult,
        CrossModelConsensus,
        ModelResponse,
    )
    from director_ai.core.dp_rag import DifferentiallyPrivateRetrieval, DPRagPipeline
    from director_ai.core.federated_dp import (
        FederatedCalibrationRound,
        FederatedDPEvidence,
    )
    from director_ai.core.federated_privacy import SecureAggregator
    from director_ai.core.scoring.consensus import ByzantineFaultTolerantConsensus
    from director_ai.core.scoring.contradiction import ContradictionScorer
    from director_ai.core.swarm_coherence import SwarmCoherenceMonitor

__all__ = ["DistributedTrustMixin"]


class DistributedTrustMixin:
    """DP retrieval, federated calibration, multi-party and multi-model trust.

    All state is initialised by :class:`~director_ai.guard.ProductionGuard`'s
    ``__init__``; the configuration comes from the composing guard through the
    ``_config`` contract declared below.
    """

    _dp_retrieval: DifferentiallyPrivateRetrieval | None
    _cross_model: CrossModelConsensus | None

    if TYPE_CHECKING:
        # Provided by the composing ProductionGuard.
        _config: DirectorConfig

    def cross_model_consensus(
        self,
        responses: Sequence[ModelResponse],
        *,
        nli: ContradictionScorer | None = None,
    ) -> ConsensusResult:
        """Measure agreement across several models' answers to one prompt.

        Scores a *panel* rather than a single response: given the same question
        answered by two or more models (each a
        :class:`~director_ai.core.consensus.ModelResponse` of ``model_id`` +
        ``text``), returns a
        :class:`~director_ai.core.consensus.ConsensusResult` with a consensus in
        ``[0, 1]``, an ``accept`` / ``review`` / ``escalate`` recommendation, the
        pairwise agreement matrix, and the specific diverging claim pairs as
        evidence.

        Pass an NLI contradiction scorer as *nli* (e.g.
        :meth:`director_ai.core.scoring.contradiction.ContradictionScorer.from_pretrained`)
        to get semantic, claim-level divergence with contradiction evidence;
        without one the consensus falls back to lexical Jaccard agreement and the
        single least-agreeing answer pair. The engine (and any supplied scorer)
        persists on the guard across calls.
        """
        from director_ai.core.consensus import CrossModelConsensus

        engine = self._cross_model
        if not isinstance(engine, CrossModelConsensus) or nli is not None:
            engine = CrossModelConsensus(nli=nli)
            self._cross_model = engine
        return engine.consensus(responses)

    def new_swarm_monitor(
        self,
        *,
        nli: ContradictionScorer | None = None,
        contradiction_threshold: float | None = None,
    ) -> SwarmCoherenceMonitor:
        """Create a stateful cross-agent cascade-coherence monitor.

        Returns a fresh
        :class:`~director_ai.core.swarm_coherence.SwarmCoherenceMonitor` for one
        multi-agent cascade: feed each agent's output to ``monitor.observe(agent_id,
        text)`` in turn and it checks the new claims against every earlier agent's
        established claims, halting the cascade (``update.halted``) on the first
        cross-agent contradiction so downstream agents never consume a poisoned
        context. Pass an NLI scorer as *nli* for contradiction detection (a fresh
        monitor per cascade keeps each conversation's state isolated).
        """
        from director_ai.core.swarm_coherence import SwarmCoherenceMonitor

        return SwarmCoherenceMonitor(
            nli=nli, contradiction_threshold=contradiction_threshold
        )

    def secure_aggregator(self, *, party_count: int) -> SecureAggregator:
        """Secure multi-party aggregation of scores (additive secret sharing).

        Each party secret-shares its value; the returned
        :class:`~director_ai.core.federated_privacy.SecureAggregator` sums the
        shares component-wise and reconstructs the multi-party total without ever
        materialising any single party's value — scoring across confidential
        knowledge bases without sharing the data. For dropout/threshold tolerance
        (any ``t`` of ``n`` parties), use the package's Shamir helpers
        (:func:`~director_ai.core.federated_privacy.shamir_split` /
        :func:`~director_ai.core.federated_privacy.shamir_reconstruct`).
        """
        from director_ai.core.federated_privacy import SecureAggregator

        return SecureAggregator(party_count=party_count)

    def byzantine_consensus(
        self, *, fault_tolerance: int = 1
    ) -> ByzantineFaultTolerantConsensus:
        """PBFT-style quorum over independent verifier votes.

        Tolerates up to ``fault_tolerance`` Byzantine (compromised/malicious)
        verifiers: it requires ``3f + 1`` independent votes and a ``2f + 1``
        quorum for the same verdict, so ``f`` adversarial verifiers can neither
        force a wrong decision nor block one the honest supermajority agrees on.
        Builds a fresh
        :class:`~director_ai.core.scoring.consensus.ByzantineFaultTolerantConsensus`
        each call (it is stateless); feed it
        :class:`~director_ai.core.scoring.consensus.BFTConsensusVote` objects.
        """
        from director_ai.core.scoring.consensus import (
            ByzantineFaultTolerantConsensus,
        )

        return ByzantineFaultTolerantConsensus(fault_tolerance=fault_tolerance)

    @property
    def dp_retrieval(self) -> DifferentiallyPrivateRetrieval:
        """Differentially private retrieval ranking with a per-tenant budget.

        Adds calibrated Laplace noise to retrieval similarity scores before
        ranking and meters each query against a per-tenant ``(ε, δ)`` budget
        (default cap 10.0), refusing a query that would exceed it. Persists
        across calls so the budget accumulates; construct
        :class:`~director_ai.core.dp_rag.DifferentiallyPrivateRetrieval` directly
        for a custom budget, sensitivity, or seed.
        """
        if self._dp_retrieval is None:
            from director_ai.core.dp_rag import DifferentiallyPrivateRetrieval

            self._dp_retrieval = DifferentiallyPrivateRetrieval(max_epsilon=10.0)
        return self._dp_retrieval

    def dp_rag_pipeline(
        self, max_epsilon: float = 10.0, **kwargs: Any
    ) -> DPRagPipeline:
        """Build a unified DP-RAG pipeline metering one per-tenant budget.

        Charges retrieval ranking, exponential-mechanism token decoding, and
        coherence-score release against a single per-tenant ``(ε)`` accountant,
        refusing any stage that would exceed ``max_epsilon`` before spending.
        Each call returns a fresh
        :class:`~director_ai.core.dp_rag.DPRagPipeline`; pass ``seed`` and the
        per-stage sensitivities through ``kwargs`` for reproducible tests or
        custom calibration.
        """
        from director_ai.core.dp_rag import DPRagPipeline

        return DPRagPipeline(max_epsilon=max_epsilon, **kwargs)

    def federated_calibration(
        self, initial_value: float | None = None, **kwargs: Any
    ) -> FederatedCalibrationRound:
        """Build a federated DP calibration round for a shared parameter.

        Tenants submit clipped local updates; the server aggregates them with
        Gaussian noise behind a minimum-cohort gate, so the shared parameter
        (default: this guard's coherence threshold) improves without any tenant's
        raw data leaving. Returns a fresh
        :class:`~director_ai.core.federated_dp.FederatedCalibrationRound`; pass
        ``clip_norm``, ``noise_multiplier``, ``min_cohort``, ``learning_rate``,
        ``value_bounds``, or ``seed`` via ``kwargs``.
        """
        from director_ai.core.federated_dp import FederatedCalibrationRound

        start = (
            self._config.coherence_threshold if initial_value is None else initial_value
        )
        return FederatedCalibrationRound(start, **kwargs)

    def federated_dp_evidence(
        self,
        calibration_round: FederatedCalibrationRound | None = None,
        **kwargs: Any,
    ) -> FederatedDPEvidence:
        """Build the formal-privacy + poisoning-resilience evidence for a round.

        Wraps a
        :class:`~director_ai.core.federated_dp.FederatedCalibrationRound` (the
        given one, or a fresh :meth:`federated_calibration` built from ``kwargs``)
        and produces its formal ``(ε, δ)`` bound (via the Rényi-DP accountant) and
        the certified worst-case poisoning shift from clipping, plus a simulated
        attack-vs-baseline check. Returns a
        :class:`~director_ai.core.federated_dp.FederatedDPEvidence`.
        """
        from director_ai.core.federated_dp import FederatedDPEvidence

        round_obj = (
            self.federated_calibration(**kwargs)
            if calibration_round is None
            else calibration_round
        )
        return FederatedDPEvidence(round_obj)
