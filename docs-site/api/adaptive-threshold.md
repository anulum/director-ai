# Adaptive Threshold Learning

::: director_ai.core.calibration.adaptive_threshold.AdaptiveThresholdLearner

::: director_ai.core.calibration.adaptive_threshold.ThresholdFeedback

::: director_ai.core.calibration.adaptive_threshold.AdaptiveThresholdArm

::: director_ai.core.calibration.adaptive_threshold.AdaptiveThresholdReport

::: director_ai.core.calibration.adaptive_threshold.AdaptiveThresholdRecommendation

## Safety Boundary

`AdaptiveThresholdLearner` is an offline recommender. It replays human-labelled
score feedback across fixed candidate thresholds, estimates each candidate with
Beta-Bernoulli posteriors, applies false-positive and false-negative safety
constraints, and returns a recommendation object.

It does not mutate `CoherenceScorer`, `DirectorConfig`, profile files, or live
runtime thresholds. Apply the returned profile overlay only after operator
approval and keep the rollback threshold in change-management records.

```python
from director_ai.core import AdaptiveThresholdLearner, ThresholdFeedback

learner = AdaptiveThresholdLearner(
    candidate_thresholds=[0.3, 0.4, 0.5, 0.6],
    current_threshold=0.4,
    max_false_negative_rate=0.05,
)
learner.observe_batch(
    [
        ThresholdFeedback(score=0.82, human_approved=True),
        ThresholdFeedback(score=0.28, human_approved=False),
    ]
)
recommendation = learner.recommend()

if recommendation.recommended_threshold is not None:
    overlay = recommendation.to_profile_overlay(profile="candidate")
```

For regulated deployments, use this together with `HumanReviewQueue`,
`OnlineCalibrator`, and drift reports. Treat candidate thresholds as a controlled
change, not as autonomous production policy.
