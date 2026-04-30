<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial licence available
Concepts 1996-2026 Miroslav Sotek. All rights reserved.
Code 2020-2026 Miroslav Sotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI - safety operations dashboard
-->

# Safety Operations Dashboard

The safety operations dashboard turns tenant-safe `SafetyEvent` JSONL and
calibration feedback into an immediate halt-rate view.

It is intended for the first response to drift, stale knowledge, and repeated
false-positive halts:

- per-tenant event count, halt count, halt rate, false-positive count, and alert
  state
- top contradiction sources across recent halts
- recent halt evidence with score, reason, source, and suggested operator action
- a ready-to-run retune command for recent labelled feedback

## Launch The UI

```bash
director-ai safety-dashboard
```

The same panel is also available inside:

```bash
director-ai wizard
```

Open the **Safety Ops** tab, paste `SafetyEvent` JSONL, paste optional feedback
JSONL, and render the tables.

## Text Mode

Use text mode when running over SSH or in CI:

```bash
director-ai safety-dashboard \
  --text \
  --events safety_events.jsonl \
  --feedback recent_feedback.jsonl
```

The command prints tenant halt rates, top contradiction sources, recent evidence,
and the retune command:

```bash
director-ai tune --dataset recent_feedback.jsonl --output director-ai-retuned.yaml
```

## Alert Thresholds

The defaults are intentionally conservative:

- halt-rate alert: `0.15`
- false-positive alert: `0.05`

Override them when a deployment has a known review cadence:

```bash
director-ai safety-dashboard \
  --text \
  --events safety_events.jsonl \
  --halt-alert-threshold 0.25 \
  --false-positive-alert-threshold 0.10
```

## Input Shape

Each event should be one tenant-safe JSON object per line. Native
`SafetyEvent.to_dict()` records work directly:

```json
{"event_id":"e1","tenant_id":"tenant-a","policy_decision":"halt","halt_reason":"contradiction","observed_score":0.22,"trace_attribution":{"fact_source":"kb://physics"},"tenant_safe_explanation":"Refresh the cited fact."}
```

Feedback rows can use the calibration format:

```json
{"event_id":"e1","tenant_id":"tenant-a","guardrail_approved":false,"human_approved":true,"source":"kb://physics"}
```

Rows where the guard rejected an answer but the human accepted it count as
false positives for the tenant.
