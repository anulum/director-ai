# Federated DP Calibration

Calibrate a shared guardrail parameter (for example the coherence threshold)
across tenants **without centralising their data**. Each tenant computes a local
update from its own labelled outcomes and submits only a single clipped scalar;
the server averages the clipped updates with added Gaussian noise behind a
minimum-cohort gate (DP-SGD-style), so the global parameter improves while no
tenant's raw data — or even its un-noised update — leaves its boundary, and no
single tenant can dominate a round.

This complements the federated *aggregators* in `core/federated_privacy` (counts
and safety signals) by aggregating *parameter updates*.

## Quick start

```python
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig

# A round seeded at the guard's coherence threshold.
rnd = ProductionGuard(DirectorConfig()).federated_calibration(
    clip_norm=0.05, noise_multiplier=1.0, min_cohort=5,
)

# Each tenant submits its clipped local update (computed on local data only).
rnd.submit_update(tenant_id="bank-a", update=+0.03)
rnd.submit_update(tenant_id="bank-b", update=+0.02)
# ... at least min_cohort tenants ...

result = rnd.aggregate()          # raises CohortTooSmallError below the gate
print(result.previous_value, result.new_value, result.cohort_size)
print(rnd.value)                  # the new shared parameter
```

`aggregate()` clips each tenant's update to `±clip_norm`, averages the cohort,
adds Gaussian noise of scale `noise_multiplier * clip_norm`, applies
`learning_rate`, clamps to `value_bounds`, and clears the pending updates. It
returns a `RoundResult` (`previous_value`, `new_value`, `cohort_size`,
`clipped_mean`, `noise_scale`); `to_dict()` carries no per-tenant updates.

## Privacy properties

- **Clipping** bounds any single tenant's influence (the DP sensitivity).
- **Gaussian noise** on the aggregate gives the differential-privacy guarantee
  (`noise_multiplier=0` disables it — for testing the arithmetic only, not
  private).
- **Minimum-cohort gate** — a round with fewer than `min_cohort` distinct tenants
  raises `CohortTooSmallError`; the value and pending updates are preserved.
- **One vote per tenant** — a resubmission overwrites that tenant's update.
- Only clipped scalars cross the boundary; raw labelled data never does.

## Direct construction

```python
from director_ai.core.federated_dp import FederatedCalibrationRound

rnd = FederatedCalibrationRound(
    initial_value=0.6,
    clip_norm=0.05,
    noise_multiplier=1.0,
    min_cohort=5,
    learning_rate=1.0,
    value_bounds=(0.0, 1.0),
    seed=None,          # system entropy in production; set for reproducible tests
)
```

## Notes

- Builds on `core/federated_privacy` mechanisms; this is DP-SGD-style federated
  averaging for a scalar calibration parameter.
- Run rounds repeatedly to track drift; the round object holds the current value
  and applies each aggregated step.
