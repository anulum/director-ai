# Runtime threshold governor

> **Status: change-management overlay for live thresholds.** The governor is the
> controlled bridge between the per-segment threshold *learner* and the live
> decision path — it applies learned thresholds gradually, with approval gating
> and an audit trail, and tightens on conformal uncertainty.

[Per-segment thresholds](segmented-thresholds.md) learn a separate halt threshold
for each domain, model, or tenant from feedback. But a recommendation is not
something to slam into production the instant it appears: an unguarded auto-tune
oscillates, over-fits a noisy hour, and leaves no audit trail.
`RuntimeThresholdGovernor` is the change-management overlay between the learner
and the runtime.

## Controlled change

The governor holds the threshold each segment is **currently using** and changes
it only through a controlled process:

- a change is proposed only once a segment has its *own* evidence (the
  recommendation source is `segment`, not the global cold-start pool) and the
  learner recommends a different threshold;
- each applied change moves at most `max_step` toward the recommendation, so the
  live threshold **ramps rather than jumps**;
- a recommendation that requires human approval is **held as pending** until
  `apply(segment, approve=True)` — unless the governor is built with
  `auto_apply=True` for self-tuning deployments;
- every applied change is recorded in `history()`.

## Conformal uncertainty tie-in

At decision time `effective_threshold(segment, interval=...)` returns the live
segment threshold, optionally tightened: when a
[conformal](threshold-tuning.md) `PredictionInterval` is supplied and the
`UncertaintyRouter` flags the estimate as unreliable or too wide, the threshold
is lowered by `uncertainty_penalty` (halt more readily) and the router's action is
surfaced — so an uncertain request is judged more conservatively than a confident
one.

## Usage

```python
from director_ai.guard import ProductionGuard

guard = ProductionGuard()
governor = guard.new_threshold_governor(max_step=0.05)   # seeded from the guard threshold

# feed labelled outcomes per segment
governor.observe(segment="medical", score=0.72, human_approved=False)

# propose a change (applied only if evidence + policy allow)
change = governor.propose("medical")
if change.requires_approval:
    governor.apply("medical", approve=True)              # human sign-off

# at decision time
eff = guard.... # your scorer
threshold = governor.effective_threshold("medical", interval=conformal_interval).threshold
```

## Measured behaviour

On the bundled scenarios (`python -m benchmarks.runtime_governor`):

| Metric | Value |
|---|---|
| Gating correctness (global / approval / auto-apply) | 1.00 |
| Bounded convergence (0.5 → 0.9 at max_step 0.05) | 8 steps, no jump |
| Throughput | ~100,000 proposals/s |

These numbers come from the committed benchmark and
`benchmarks/results/runtime_governor.json`; reproduce them with the command above.
