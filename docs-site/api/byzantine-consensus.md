# Byzantine-Resilient Consensus

::: director_ai.core.scoring.consensus.ByzantineFaultTolerantConsensus

::: director_ai.core.scoring.consensus.BFTConsensusVote

::: director_ai.core.scoring.consensus.BFTConsensusResult

## Fault Model

`ByzantineFaultTolerantConsensus` implements a PBFT-style quorum over
independent verifier votes. For a configured fault tolerance `f`:

- at least `3f + 1` independent verifier votes are required
- a decision requires `2f + 1` votes for the same verdict
- duplicate verifier names are rejected
- missing quorum returns `warn`, not `allow`

```python
from director_ai.core import BFTConsensusVote, ByzantineFaultTolerantConsensus

consensus = ByzantineFaultTolerantConsensus(fault_tolerance=1)
result = consensus.decide(
    (
        BFTConsensusVote("nli-a", "allow", 0.1, "claim://1"),
        BFTConsensusVote("nli-b", "allow", 0.2, "claim://2"),
        BFTConsensusVote("symbolic", "allow", 0.0, "proof://1"),
        BFTConsensusVote("adversarial", "halt", 0.9, "claim://x"),
    ),
    policy_id="policy.bft",
)
```

This is a verifier-vote quorum layer. It does not prove that verifiers are
actually independent or uncompromised; deployment policy must justify that
assumption before claiming Byzantine tolerance.
