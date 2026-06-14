# Per-segment adaptive thresholds

> **Status: core (Apache-2.0), offline, human-gated.** Like the global
> [adaptive threshold learner](threshold-tuning.md) this only *recommends* — route
> the recommendation through change management before promotion.

The right halt threshold is not global. A clinical-domain answer and a casual chat
tolerate very different false-positive rates, and two models hallucinate at
different rates. `SegmentedThresholdLearner` wraps the global
`AdaptiveThresholdLearner` with a per-segment routing layer, so each segment — any
string key: a domain, a model id, a tenant id, or a composite — accumulates its
own human-labelled evidence and earns its own threshold.

## Cold start

A global pool sees **every** observation. A segment with fewer than
`promote_after` (default `min_samples`) observations of its own falls back to the
pooled recommendation, so a brand-new segment is never starved of guidance; once
it has earned enough evidence, its own learner takes over.

## Usage

```python
from director_ai.core import SegmentedThresholdLearner

learner = SegmentedThresholdLearner(
    candidate_thresholds=[0.3, 0.5, 0.7, 0.9],
    current_threshold=0.5,
    min_samples=20,
)

# Replay human-labelled feedback, tagged by segment.
learner.observe(score=0.91, human_approved=True, segment="clinical")
learner.observe(score=0.44, human_approved=True, segment="chat")

rec = learner.recommend(segment="clinical")
rec.source                          # "segment" once promoted, else "global"
rec.recommendation.recommended_threshold
rec.recommendation.to_profile_overlay(profile="clinical-adaptive")
```

`recommend` returns a `SegmentRecommendation` carrying the source, the per-segment
feedback count, and the underlying `AdaptiveThresholdRecommendation` (with its
safety constraints, expected lift, and rollback threshold).

## Measured

On a synthetic three-segment workload where each segment has a different latent
approval boundary (`python -m benchmarks.segmented_threshold`):

| Policy | Mean approval accuracy |
|---|---|
| Per-segment thresholds | 0.89 |
| Single global threshold | 0.83 |
| **Segmentation lift** | **+0.06** |

The largest gain is on the strict (clinical-style) segment, where the global
compromise threshold is far too lenient (0.91 vs 0.72 accuracy). A segment whose
optimum happens to coincide with the global one gains nothing — segmentation helps
exactly where segments diverge. Numbers come from the committed benchmark and
`benchmarks/results/segmented_threshold.json`.

## Backend

This layer adds only routing and bookkeeping; the Beta-posterior arithmetic and
its Rust `rust_beta_posterior_mean` fast path are inherited unchanged from
`AdaptiveThresholdLearner`. See [Rust Acceleration](rust-acceleration.md).
