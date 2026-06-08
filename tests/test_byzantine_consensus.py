# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Byzantine Consensus Tests

import pytest

from director_ai.core import (
    BFTConsensusVote,
    ByzantineFaultTolerantConsensus,
)


def test_pbft_quorum_accepts_with_one_fault_among_four():
    consensus = ByzantineFaultTolerantConsensus(fault_tolerance=1)
    votes = (
        BFTConsensusVote("nli-a", "allow", 0.1, "claim://1"),
        BFTConsensusVote("nli-b", "allow", 0.2, "claim://2"),
        BFTConsensusVote("symbolic", "allow", 0.0, "proof://1"),
        BFTConsensusVote("compromised", "halt", 0.9, "claim://bad"),
    )

    result = consensus.decide(votes, policy_id="policy.bft")

    assert result.decision == "allow"
    assert result.quorum_size == 3
    assert result.required_replicas == 4
    assert result.fault_tolerance == 1
    assert result.byzantine_resilient is True
    assert result.participating_verifiers == ("nli-a", "nli-b", "symbolic")


def test_no_quorum_warns_without_pretending_consensus():
    consensus = ByzantineFaultTolerantConsensus(fault_tolerance=1)
    votes = (
        BFTConsensusVote("a", "allow", 0.1),
        BFTConsensusVote("b", "warn", 0.5),
        BFTConsensusVote("c", "halt", 0.9),
        BFTConsensusVote("d", "allow", 0.2),
    )

    result = consensus.decide(votes, policy_id="policy.bft")

    assert result.decision == "warn"
    assert result.reason == "bft_no_quorum"
    assert result.byzantine_resilient is False


def test_insufficient_replicas_warns_for_requested_fault_tolerance():
    consensus = ByzantineFaultTolerantConsensus(fault_tolerance=2)
    votes = tuple(BFTConsensusVote(f"v{i}", "allow", 0.1) for i in range(6))

    result = consensus.decide(votes, policy_id="policy.bft")

    assert result.decision == "warn"
    assert result.reason == "bft_insufficient_replicas"
    assert result.required_replicas == 7
    assert result.byzantine_resilient is False


def test_duplicate_verifier_votes_are_rejected():
    consensus = ByzantineFaultTolerantConsensus(fault_tolerance=1)

    with pytest.raises(ValueError, match="duplicate verifier"):
        consensus.decide(
            (
                BFTConsensusVote("same", "allow", 0.1),
                BFTConsensusVote("same", "halt", 0.9),
                BFTConsensusVote("c", "allow", 0.1),
                BFTConsensusVote("d", "allow", 0.1),
            ),
            policy_id="policy.bft",
        )


def test_result_to_dict_is_tenant_safe():
    consensus = ByzantineFaultTolerantConsensus(fault_tolerance=1)
    result = consensus.decide(
        (
            BFTConsensusVote("a", "halt", 0.9, "secret://raw-prompt"),
            BFTConsensusVote("b", "halt", 0.8, "proof://2"),
            BFTConsensusVote("c", "halt", 0.7, "proof://3"),
            BFTConsensusVote("d", "allow", 0.1, "proof://4"),
        ),
        policy_id="policy.bft",
    )

    payload = result.to_dict()
    assert payload["decision"] == "halt"
    assert "raw-prompt" not in str(payload)
    assert payload["evidence_refs"] == ("redacted", "proof://2", "proof://3")


def test_production_guard_exposes_byzantine_consensus():
    from director_ai.core.config import DirectorConfig
    from director_ai.guard import ProductionGuard

    guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
    consensus = guard.byzantine_consensus(fault_tolerance=1)
    assert isinstance(consensus, ByzantineFaultTolerantConsensus)
    assert consensus.required_replicas == 4
    assert consensus.quorum_size == 3
    result = consensus.decide(
        (
            BFTConsensusVote("a", "allow", 0.1),
            BFTConsensusVote("b", "allow", 0.2),
            BFTConsensusVote("c", "allow", 0.0),
            BFTConsensusVote("compromised", "halt", 0.9),
        ),
        policy_id="policy.bft",
    )
    assert result.decision == "allow"
    assert result.byzantine_resilient is True
