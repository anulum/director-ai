# Long-Context Intent-Drift Interlock

`IntentDriftInterlock` catches slow-burn jailbreaks that no single turn reveals.
A crescendo attack never trips a per-turn guard: each request drifts a little
further from the declared intent than the last, every step individually benign.
The signal lives in the **trajectory** — sustained drift, a rising slope,
accumulating cross-turn contradiction — not in any one turn.

## What it accumulates

The interlock folds three per-turn signals (all already produced during scoring)
into a fixed-size state that survives a 100k-token conversation without keeping
the raw history:

| Signal | Source |
| --- | --- |
| intent divergence | the cross-turn NLI divergence (`cross_turn_divergence`) |
| injection risk | the per-turn intent-grounded `injection_risk` |
| contradiction trend | the slope from the [contradiction tracker](temporal-consistency-graph.md) |

It keeps an exponential moving average of divergence and injection (old turns
decay rather than being dropped), a windowed slope of the divergence (the
gradual-escalation signal), and combines them into a single `drift_risk`. The
interlock **trips** when `drift_risk` crosses `trigger_threshold` after at least
`min_turns` turns — so a conversation creeping from 0.3 to 0.6 divergence over
six turns is halted even though no single turn ever clears a per-turn block, and
a lone off-topic opening turn never fires it.

```text
  divergence per turn:  0.2  0.3  0.4  0.5  0.6  0.7    (each below a 0.7 block)
  drift_risk:           .12  .12  .14  .35  .42  .49 ◄── trips here
```

## Enabling

The interlock is opt-in on the session and **off by default** — when absent, the
scorer leaves `intent_drift_risk` / `intent_drift_triggered` as `None`:

```python
from director_ai.core import CoherenceScorer
from director_ai.core.runtime.session import ConversationSession

scorer = CoherenceScorer(threshold=0.5, use_nli=True)
session = ConversationSession(track_intent_drift=True)

for prompt, response in conversation:
    approved, score = scorer.review(prompt, response, session=session)
    if score.intent_drift_triggered:
        halt("slow-burn intent drift detected")
```

Meaningful divergence signal requires a model-backed NLI backend; the interlock
itself is pure and numeric, so it is fully deterministic and testable without any
model.

## Tuning

```python
from director_ai.core.runtime.intent_drift import IntentDriftInterlock

session.intent_drift = IntentDriftInterlock(
    half_life_turns=4.0,     # EMA decay horizon
    window=8,                # slope estimate window (bounds memory)
    trigger_threshold=0.6,   # drift_risk at/above which it trips
    min_turns=3,             # turns required before it can trip
)
```

## Full API

::: director_ai.core.runtime.intent_drift.IntentDriftInterlock

::: director_ai.core.runtime.intent_drift.DriftState
