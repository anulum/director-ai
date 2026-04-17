# Agent Trajectory Preflight

When every generation costs tokens, compute, and user attention,
catching a likely-to-fail prompt *before* the model runs saves all
three. The trajectory simulator runs a small fan-out of cheap
sampled trajectories, scores each, and returns a verdict — proceed,
warn, or halt — that the gateway can act on without ever calling
the full model.

## The flow

```
prompt ──► TrajectorySimulator.preflight
                │
                ├──► Actor.sample(prompt, seed=17)  ──┐
                ├──► Actor.sample(prompt, seed=18)    │  N draws
                ├──► …                                │  (default 8)
                └──► Actor.sample(prompt, seed=24)  ──┘
                         │
                         └──► CoherenceScorer.review(prompt, draw)
                                    │
                                    └──► halt_rate + CI + action
```

The actor is any object with a `sample(prompt, seed) -> list[str]`
method. In production it wraps a small distilled LLM; in tests and
smoke runs it can be a deterministic mock. The scorer is any
object with the standard `review(prompt, action)` interface — the
shipped `CoherenceScorer` plugs in unchanged.

## Minimal reproduction

```python
from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.trajectory import TrajectorySimulator
from director_ai.core.actor import MockGenerator


class MockActor:
    """Wrap the shipped MockGenerator to match the Actor protocol."""

    def __init__(self) -> None:
        self._gen = MockGenerator()

    def sample(self, prompt: str, seed: int) -> list[str]:
        # Mock generator is deterministic under a fixed seed. Real
        # actors draw from a distilled LLM with ``seed`` controlling
        # sampling noise.
        candidates = self._gen.generate_candidates(prompt, n=1)
        return candidates[0]["text"].split() if candidates else []


store = GroundTruthStore()
store.add("capital", "Paris is the capital of France.")
scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)

simulator = TrajectorySimulator(
    actor=MockActor(),
    scorer=scorer,
    n_simulations=8,
    halt_rate_warn=0.25,
    halt_rate_halt=0.50,
)

verdict = simulator.preflight("What is the capital of France?")
print(verdict.recommended)  # "proceed" / "warn" / "halt"
print(f"halt_rate={verdict.halt_rate:.2f} mean_coh={verdict.mean_coherence:.3f}")
for t in verdict.trajectories:
    print(f"  traj {t.trajectory_id}: coh={t.final_coherence:.3f} ok={t.approved}")
```

## Action bands

| Recommended action | Halt rate | What the gateway does |
| --- | --- | --- |
| `proceed` | < 0.25 | Run the real model with the cheaper scorer backend. |
| `warn` | 0.25 – 0.50 | Escalate to the NLI backend; stamp `X-Trajectory-Warn`. |
| `halt` | ≥ 0.50 | Return 422 before the upstream call. The prompt is likely to produce a hallucination that the streaming kernel would halt anyway. |

The thresholds are configurable per deployment. Medical and
financial domains typically tighten both bounds (`warn=0.10`,
`halt=0.25`) so marginal prompts get NLI scoring; creative or
chat domains widen them (`warn=0.40`, `halt=0.70`).

## Observing individual draws

```python
def log_trajectory(t):
    print(f"seed={t.seed} coh={t.final_coherence:.3f} {t.text!r}")

verdict = simulator.preflight(
    "Summarise ANULUM's 2025 performance.",
    on_trajectory=log_trajectory,
)
```

The `on_trajectory` callback fires once per draw as soon as the
scorer returns, before the aggregate verdict is ready. Exceptions
raised by the callback are caught and logged; a broken observer
cannot abort the preflight. This is the hook to wire into Langfuse
or an OTel tracer — send each trajectory as a sibling span under
the preflight span so the full fan-out is visible post hoc.

## Seeded determinism for forensics

Two preflight calls with the same prompt produce byte-identical
verdicts:

```python
a = simulator.preflight("Tell me about France.")
b = simulator.preflight("Tell me about France.")
assert [t.tokens for t in a.trajectories] == [t.tokens for t in b.trajectories]
```

The per-trajectory seed is `base_seed + i`; operators reconstruct
any historical preflight decision by replaying the same
`base_seed` and `n_simulations`. This is the single most useful
artefact for incident review — a week later you can show exactly
which draws the simulator saw and which one(s) triggered the
halt.

## Aggregate metrics

```python
verdict = simulator.preflight(prompt)

verdict.halt_rate        # fraction of draws that failed
verdict.mean_coherence   # arithmetic mean across draws
verdict.std_coherence    # stdev (zero with one draw)
verdict.ci_low           # 2.5% empirical quantile
verdict.ci_high          # 97.5% empirical quantile
verdict.min_coherence    # lowest draw
verdict.max_coherence    # highest draw
```

The CI is a plain empirical band, not a conformal prediction —
the simulator is foundation scope. Calibrate a proper conformal
threshold against historical traces once the fan-out has been
running in production long enough to collect a calibration set.

## Cost

Every preflight call is N extra scoring reviews. The default
`n_simulations=8` multiplies scoring cost by 8× — worthwhile for
deployments where a halted stream is more expensive than the
preflight fan-out (enterprise customer support, medical), not for
deployments where latency budget is tight and streaming halts
are cheap (demos, internal tools). Measure first, tune second.

## Not in this module

- **Distilled-actor integration.** The protocol accepts any
  ``Actor``; a production deployment wires a small seq-to-seq
  model here. The simulator is model-agnostic.
- **Conformal calibration.** The shipped CI is empirical quantiles;
  conformal bands need historical data.
- **Agent handoff.** `HandoffScorer` already covers the
  inter-agent edge; the trajectory simulator is for pre-execution
  single-turn prompts. The two compose: the gateway preflights,
  then the swarm guard catches anything that survives.
- **Rust fast-path.** The simulator loop is a thin orchestration
  layer; the hot path is the scorer itself, which already has a
  Rust backend. Parallelising the fan-out across threads is a
  v2 concern.

## See also

- [`docs-site/cookbook/streaming-halt-guide.md`](./streaming-halt-guide.md)
  — what happens when preflight says `proceed` and the stream
  drifts anyway.
- [`docs-site/cookbook/multi-agent-handoff-failures.md`](./multi-agent-handoff-failures.md)
  — catching the same failure mode at the handoff edge instead of
  at preflight.
- `ROADMAP.md` — the trajectory simulator is Tier 1 #1 of the
  2026-2030 roadmap; this cookbook covers the foundation shipped
  on 2026-04-17.
