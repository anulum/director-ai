# Swarm-level coherence

> **Status: real-time cross-agent cascade halt.** The monitor watches a running
> multi-agent swarm and stops it the moment one agent contradicts another, before
> the contradiction propagates downstream. It is a swarm-orchestration guard built
> on the same NLI contradiction signal as the streaming halt.

In a swarm, agents run in sequence and each builds on what the earlier ones
produced. If one agent asserts something that contradicts an established claim,
every downstream agent inherits the error — the contradiction *cascades*.
`SwarmCoherenceMonitor` watches the swarm as it runs: as each agent emits text,
its claims are checked against the claims every earlier agent already
established, and the first real cross-agent contradiction **halts the cascade**
before the next agent consumes the poisoned context.

## How it works

- each agent's output is split into claims; claims accumulate, tagged by agent;
- a new claim is compared against every earlier agent's established claims with an
  injected NLI scorer, scored in both directions (the stronger `P(contradiction)`
  decides, because "A contradicts B" and "B contradicts A" are scored separately);
- claims from the **same** agent are never compared against each other — a single
  agent revising itself is not a swarm contradiction;
- the first claim pair at or above the threshold halts the cascade; the monitor
  stays halted until `reset()`, so downstream agents short-circuit.

Each flagged conflict is a `CascadeContradiction` carrying its evidence: which
agent contradicted which, the two claims, the contradiction strength, and their
lexical (topical) overlap via the Rust `rust_word_overlap` kernel.

## Usage

```python
from director_ai.guard import ProductionGuard
from director_ai.core.scoring.contradiction import ContradictionScorer

guard = ProductionGuard()
monitor = guard.new_swarm_monitor(nli=ContradictionScorer.from_pretrained())

for agent_id, output in run_swarm():           # your orchestrator loop
    update = monitor.observe(agent_id, output)
    if update.halted:
        c = update.contradictions[0]
        raise SwarmHalt(
            f"{c.new_agent} contradicts {c.prior_agent}: "
            f"{c.new_claim!r} vs {c.prior_claim!r} ({c.contradiction:.2f})"
        )
```

A fresh monitor per cascade keeps each conversation's state isolated. The running
`update.coherence` is `1 - contradicted_claims / total_claims`.

## Polyglot backend

The lexical topical-overlap behind each contradiction's `topical_overlap` runs
through the Rust `backfire_kernel.rust_word_overlap` extension when installed,
with a bit-for-bit identical pure-Python Jaccard fallback. See
[Rust Acceleration](rust-acceleration.md).

## Measured behaviour

On the bundled labelled cascades (`python -m benchmarks.swarm_coherence`):

| Metric | Value |
|---|---|
| Cascade-halt correctness (coherent vs poisoned) | 1.00 |
| Rust ↔ Python parity | exact |
| Throughput | ~210,000 observe/s |

These numbers come from the committed benchmark and
`benchmarks/results/swarm_coherence.json`; reproduce them with the command above.
The benchmark exercises the cascade machinery with a deterministic NLI stub and
the offline lexical backend; the contradiction judgement itself is the injected
`ContradictionScorer`, whose own quality is measured in the
[streaming-halt guide](streaming.md).
