# Active Labelling Cockpit

A guard improves only as fast as it learns where it is wrong. The active
labelling cockpit turns a stream of scored guard decisions into a review
workflow: it surfaces the most informative items to label, measures the two
error types from the labels, shows the threshold trade-off, recommends an
operating threshold, and exports a packet for retraining.

## What it does

- **Rank what to label.** `rank_for_labelling` returns the unlabelled items whose
  score sits closest to the decision boundary — where a human label resolves the
  most uncertainty — so review effort goes where it pays off.
- **Measure error.** `error_breakdown` splits the labelled items, by the guard's
  actual decision, into **false halts** (grounded answers the guard blocked) and
  **missed hallucinations** (hallucinations the guard approved).
- **See the trade-off.** `tradeoff_curve` sweeps the threshold and reports both
  error counts at every decision boundary.
- **Recommend a threshold.** `recommend_threshold` picks the threshold (optionally
  per domain) that minimises the weighted error — tune `miss_weight` vs
  `false_halt_weight` for the domain's risk appetite.
- **Export for retraining.** `export_packet` emits a deterministic train/eval
  split of the labelled items, consumable by Lite Scorer v2 training.

```python
from director_ai.core.labelling_cockpit import ActiveLabellingCockpit, LabelItem

cockpit = ActiveLabellingCockpit(threshold=0.6)

# 1. Surface the top items to label
queue = cockpit.rank_for_labelling(scored_items, top_n=50)

# 2. After labelling, measure error
breakdown = cockpit.error_breakdown(labelled_items)
print(breakdown.false_halts, breakdown.missed_hallucinations)

# 3. Recommend a per-domain threshold
rec = cockpit.recommend_threshold(labelled_items, domain="finance", miss_weight=3.0)
print(rec.threshold, rec.point.total_errors)

# 4. Export a retraining packet
packet = cockpit.export_packet(labelled_items, eval_fraction=0.2)
```

A `LabelItem` carries the guard's score and decision, the domain, an optional
reviewer `label` (`"grounded"` / `"hallucination"`), and the prompt/response text
that flows into the exported packet.

## Through the guard

`ProductionGuard.labelling_cockpit` returns a cockpit at the guard's configured
threshold:

```python
from director_ai.guard import ProductionGuard

guard = ProductionGuard.from_profile("finance")
queue = guard.labelling_cockpit.rank_for_labelling(scored_items)
```

The export is deterministic — the same labels always produce the same split, so a
retraining run is reproducible.
