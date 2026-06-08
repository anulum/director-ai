# Runtime Application Self-Protection (RASP)

The last line of defence once input filters and guardrails are bypassed: the
application watches its own behaviour for an attack already in progress — a
request-rate spike, an oversized payload, a halt-rate surge. RASP scores each
observation against a rolling baseline and reports an `ok` / `watch` / `alert`
verdict; the host decides whether to shed load, throttle, or block.

The detector is dependency-free and distribution-free: a streaming **robust**
z-score (median + median-absolute-deviation), so a few extreme values — the attack
itself — do not inflate the baseline and mask the next anomaly. (An IsolationForest
ensemble can be layered on later; the robust statistic is the always-available
floor.)

## Quick start

```python
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig

rasp = ProductionGuard(DirectorConfig()).rasp

# Feed per-request behavioural metrics as they happen.
for rate in normal_request_rates:
    rasp.observe("request_rate", rate)        # builds the baseline

verdict = rasp.observe("request_rate", 5000)  # a sudden flood
print(verdict.severity)          # "alert"
print(verdict.to_dict())         # metric, value, robust_z, severity, anomalous, cold_start

if rasp.under_attack(min_anomalies=3):
    ...  # shed load / escalate
```

## Scoring and severity

`observe(metric, value)` returns an `AnomalyVerdict`:

| Field | Meaning |
|-------|---------|
| `metric` | The metric name. |
| `value` | The observed value. |
| `robust_z` | Median/MAD robust z-score against the rolling baseline. |
| `severity` | `ok`, `watch`, or `alert`. |
| `anomalous` | True when `robust_z` exceeds the threshold (default 3.5). |
| `cold_start` | True while the baseline is still filling (under `min_samples`). |

Severity bands: `alert` once `robust_z` crosses the threshold, `watch` once it
enters the shoulder below it (`watch_fraction`, default 0.7), `ok` otherwise. A
cold-start observation is always `ok` — there is no baseline to judge against yet.

When the MAD is zero but the window has spread (its known weakness, when over half
the values equal the median), the detector falls back to the mean absolute
deviation so an in-range value is not flagged as an infinite outlier; only a
genuinely constant baseline scores any deviation as maximal.

## State

- `observe()` keeps a separate `StreamingRobustDetector` per metric, each with its
  own rolling window — request rate, payload size, and halt rate are judged
  independently.
- `recent_anomaly_count` is the number of anomalies in the recent window;
  `under_attack(min_anomalies=3)` is the convenience threshold over it.
- `tracked_metrics` lists the metrics with an active baseline.

All serialisation is tenant-safe — metric names and scores only, never request
content. RASP scores; the host owns the block decision (for action-level gating,
compose it with the [execution rings](execution-rings.md)).
