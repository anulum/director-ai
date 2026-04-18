# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — orchestrator regression detector

"""Per-metric regression detection against a baseline report.

A regression rule declares:

* ``metric`` — which :class:`MetricResult.name` to track.
* ``case_name`` — which :class:`SuiteEntry.name` the metric
  lives under (a metric name may repeat across cases, e.g.
  ``p99_latency_ms`` under the CPU case and again under GPU).
* ``absolute_tolerance`` / ``relative_tolerance`` — the budget
  for change before the rule fires. One of the two must be set;
  setting both means *either* exceeding fires.
* ``min_samples`` — optional guard that ignores the delta when
  the dataset this run actually measured is smaller than the
  baseline (e.g. a quick 100-sample spot-check against the full
  29 k AggreFact baseline).

The engine is intentionally deterministic — the same baseline +
same run always produces the same regression verdict. No
statistical machinery today; add one explicitly if / when the
team wants confidence-interval-aware gates (the schema is
extensible).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .schema import MetricResult, RunReport, validate_report


@dataclass(frozen=True)
class RegressionRule:
    """Single metric-level regression gate.

    Exactly one of ``absolute_tolerance`` and ``relative_tolerance``
    *must* be non-zero; both may be set (the rule fires when
    *either* budget is exceeded in the bad direction).
    """

    case_name: str
    metric: str
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0
    min_samples: int = 0
    severity: str = "high"

    def __post_init__(self) -> None:
        if not self.case_name or not self.metric:
            raise ValueError("case_name and metric are required")
        if self.absolute_tolerance == 0.0 and self.relative_tolerance == 0.0:
            raise ValueError(
                "regression rule must set absolute_tolerance and/or "
                "relative_tolerance; both zero means no gate"
            )
        if self.absolute_tolerance < 0 or self.relative_tolerance < 0:
            raise ValueError("tolerances must be non-negative")
        if self.severity not in {"low", "medium", "high"}:
            raise ValueError(f"severity must be low/medium/high; got {self.severity!r}")


@dataclass(frozen=True)
class RegressionFinding:
    """Populated when a :class:`RegressionRule` fires against a run."""

    rule: RegressionRule
    baseline_value: float
    current_value: float
    absolute_delta: float
    relative_delta: float
    reason: str

    @property
    def severity(self) -> str:
        return self.rule.severity


@dataclass(frozen=True)
class RegressionReport:
    """Aggregate answer from :func:`detect_regressions`.

    ``findings`` is empty when the run is within every rule's
    tolerance. ``skipped_rules`` lists rules that could not be
    evaluated (missing case / missing metric / sample-count
    guard) — a CI gate typically treats ``findings`` as failure
    and ``skipped_rules`` as warning.
    """

    findings: tuple[RegressionFinding, ...]
    skipped_rules: tuple[tuple[RegressionRule, str], ...] = field(default_factory=tuple)

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def high_severity(self) -> tuple[RegressionFinding, ...]:
        return tuple(f for f in self.findings if f.severity == "high")

    def to_json(self, indent: int | None = 2) -> str:
        payload = {
            "findings": [
                {
                    "rule": asdict(f.rule),
                    "baseline_value": f.baseline_value,
                    "current_value": f.current_value,
                    "absolute_delta": f.absolute_delta,
                    "relative_delta": f.relative_delta,
                    "reason": f.reason,
                    "severity": f.severity,
                }
                for f in self.findings
            ],
            "skipped_rules": [
                {"rule": asdict(r), "reason": why} for r, why in self.skipped_rules
            ],
        }
        return json.dumps(payload, indent=indent, sort_keys=True)

    def to_file(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        return p


def detect_regressions(
    current: RunReport,
    baseline: RunReport,
    rules: Iterable[RegressionRule],
) -> RegressionReport:
    """Evaluate every *rule* against the delta between *current*
    and *baseline*. Returns a :class:`RegressionReport`."""
    findings: list[RegressionFinding] = []
    skipped: list[tuple[RegressionRule, str]] = []

    for rule in rules:
        cur_entry = current.entry(rule.case_name)
        base_entry = baseline.entry(rule.case_name)
        if cur_entry is None or base_entry is None:
            skipped.append(
                (rule, f"case {rule.case_name!r} missing from one of the runs")
            )
            continue
        if cur_entry.status != "passed":
            skipped.append((rule, f"current case status is {cur_entry.status!r}"))
            continue
        cur_metric = cur_entry.metric(rule.metric)
        base_metric = base_entry.metric(rule.metric)
        if cur_metric is None or base_metric is None:
            skipped.append(
                (
                    rule,
                    f"metric {rule.metric!r} missing under case {rule.case_name!r}",
                )
            )
            continue

        if rule.min_samples > 0 and cur_entry.dataset_size < rule.min_samples:
            skipped.append(
                (
                    rule,
                    f"current dataset_size={cur_entry.dataset_size} "
                    f"< min_samples={rule.min_samples}",
                )
            )
            continue

        finding = _evaluate(rule, base_metric, cur_metric)
        if finding is not None:
            findings.append(finding)

    return RegressionReport(findings=tuple(findings), skipped_rules=tuple(skipped))


def _evaluate(
    rule: RegressionRule,
    baseline: MetricResult,
    current: MetricResult,
) -> RegressionFinding | None:
    if baseline.higher_is_better != current.higher_is_better:
        # Direction mismatch means one of the runs mis-reported
        # the metric's polarity — refuse to compare rather than
        # silently conclude the wrong answer.
        return RegressionFinding(
            rule=rule,
            baseline_value=baseline.value,
            current_value=current.value,
            absolute_delta=current.value - baseline.value,
            relative_delta=_relative_delta(baseline.value, current.value),
            reason="higher_is_better polarity differs between runs",
        )

    absolute_delta = current.value - baseline.value
    relative_delta = _relative_delta(baseline.value, current.value)

    # Bad direction = decreasing when higher is better, increasing
    # when lower is better. The regression rule's budget is checked
    # against the signed delta in the bad direction.
    bad_absolute = -absolute_delta if current.higher_is_better else absolute_delta
    bad_relative = -relative_delta if current.higher_is_better else relative_delta

    abs_triggered = (
        rule.absolute_tolerance > 0 and bad_absolute > rule.absolute_tolerance
    )
    rel_triggered = (
        rule.relative_tolerance > 0 and bad_relative > rule.relative_tolerance
    )

    if not (abs_triggered or rel_triggered):
        return None

    parts: list[str] = []
    if abs_triggered:
        parts.append(
            f"absolute delta {absolute_delta:+.6g} {current.unit} "
            f"exceeds budget {rule.absolute_tolerance:+.6g}"
        )
    if rel_triggered:
        parts.append(
            f"relative delta {relative_delta:+.2%} exceeds budget "
            f"{rule.relative_tolerance:+.2%}"
        )
    return RegressionFinding(
        rule=rule,
        baseline_value=baseline.value,
        current_value=current.value,
        absolute_delta=absolute_delta,
        relative_delta=relative_delta,
        reason="; ".join(parts),
    )


def _relative_delta(baseline: float, current: float) -> float:
    if baseline == 0.0:
        return 0.0
    return (current - baseline) / abs(baseline)


def load_baseline(path: str | Path) -> RunReport:
    """Load and validate the baseline :class:`RunReport` from disk.

    Raises :class:`ValueError` with a helpful message when the
    payload does not round-trip through the schema — easier than
    tracing a cryptic KeyError later in the regression pipeline.
    """
    p = Path(path)
    raw: Mapping[str, object] = json.loads(p.read_text())
    validate_report(raw)
    return RunReport.from_json(raw)


def default_rules(cases: Sequence[str] | None = None) -> tuple[RegressionRule, ...]:
    """Built-in regression gates for the standard suite.

    ``cases`` is an optional filter — rules whose
    ``case_name`` is not in the set are omitted. Operators
    typically call this with no argument and then override
    specific rules via a YAML config.
    """
    all_rules: tuple[RegressionRule, ...] = (
        RegressionRule(
            case_name="nli_tier5_aggrefact_macro",
            metric="balanced_accuracy",
            absolute_tolerance=0.02,
            severity="high",
        ),
        RegressionRule(
            case_name="nli_tier4_distilled_sanity",
            metric="sanity_pass_rate",
            absolute_tolerance=0.01,
            severity="high",
        ),
        RegressionRule(
            case_name="e2e_halueval_qa",
            metric="f1",
            absolute_tolerance=0.03,
            severity="high",
        ),
        RegressionRule(
            case_name="e2e_halueval_qa",
            metric="recall",
            absolute_tolerance=0.03,
            severity="high",
        ),
        RegressionRule(
            case_name="latency_scorer_cpu",
            metric="p99_latency_ms",
            relative_tolerance=0.15,
            severity="medium",
        ),
        RegressionRule(
            case_name="latency_scorer_gpu",
            metric="p99_latency_ms",
            relative_tolerance=0.15,
            severity="medium",
        ),
        RegressionRule(
            case_name="rust_parity_safety",
            metric="pass_count",
            absolute_tolerance=0.0,
            relative_tolerance=0.0001,
            severity="high",
        ),
        RegressionRule(
            case_name="injection_adversarial_suite",
            metric="detection_rate",
            absolute_tolerance=0.02,
            severity="high",
        ),
        RegressionRule(
            case_name="feature_matrix_safety_hooks",
            metric="pass_count",
            absolute_tolerance=0.0,
            relative_tolerance=0.0001,
            severity="high",
        ),
        RegressionRule(
            case_name="pytest_coverage",
            metric="coverage_percent",
            absolute_tolerance=1.0,
            severity="medium",
        ),
    )
    if cases is None:
        return all_rules
    wanted = set(cases)
    return tuple(r for r in all_rules if r.case_name in wanted)
