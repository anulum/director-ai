# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for the board-level KPI presentation/export layer.

Covers target validation, the lower/upper status bands (including the watch
shoulder), the per-domain status fan-out, overall-status precedence, and both
renderers — exercised against KpiReports built by ``compute_kpis`` so the
presentation layer stays faithful to the real data layer.
"""

from __future__ import annotations

import pytest

from director_ai.core.labelling_cockpit import LabelItem
from director_ai.core.observability import KpiReport, compute_kpis
from director_ai.core.observability.kpi_report import (
    ALERT,
    NOT_AVAILABLE,
    OK,
    WATCH,
    KpiTargets,
    _fmt,
    _status_lower,
    _status_upper,
    kpi_statuses,
    overall_status,
    render_markdown,
    render_text,
)


def _report(**overrides) -> KpiReport:
    base = dict(
        labelled_total=10,
        halt_rate=0.3,
        halt_precision=0.95,
        false_positive_rate=0.02,
        per_domain_false_positive_rate={},
        p95_scoring_latency_ms=20.0,
        tenant_boundary_violations=0,
        unsigned_kb_writes_rejected=0,
        security_exception_debt=0,
    )
    base.update(overrides)
    return KpiReport(**base)


class TestKpiTargets:
    def test_defaults_are_valid(self):
        t = KpiTargets()
        assert t.max_false_positive_rate == 0.10
        assert t.min_halt_precision == 0.80

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_false_positive_rate": 1.5}, "max_false_positive_rate"),
            ({"max_false_positive_rate": -0.1}, "max_false_positive_rate"),
            ({"min_halt_precision": 1.1}, "min_halt_precision"),
            ({"min_halt_precision": -0.1}, "min_halt_precision"),
            ({"max_p95_latency_ms": 0.0}, "max_p95_latency_ms"),
            ({"max_p95_latency_ms": -5.0}, "max_p95_latency_ms"),
            ({"watch_fraction": 0.0}, "watch_fraction"),
            ({"watch_fraction": 1.0}, "watch_fraction"),
        ],
    )
    def test_rejects_out_of_range(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            KpiTargets(**kwargs)


class TestStatusBands:
    def test_upper_none_is_not_available(self):
        assert _status_upper(None, 0.10, 0.8) == NOT_AVAILABLE

    def test_upper_above_limit_alerts(self):
        assert _status_upper(0.20, 0.10, 0.8) == ALERT

    def test_upper_in_watch_shoulder(self):
        # 0.09 > 0.10 * 0.8 (=0.08) but <= 0.10 -> watch.
        assert _status_upper(0.09, 0.10, 0.8) == WATCH

    def test_upper_well_below_is_ok(self):
        assert _status_upper(0.01, 0.10, 0.8) == OK

    def test_lower_none_is_not_available(self):
        assert _status_lower(None, 0.80, 0.8) == NOT_AVAILABLE

    def test_lower_below_floor_alerts(self):
        assert _status_lower(0.5, 0.80, 0.8) == ALERT

    def test_lower_in_watch_shoulder(self):
        # floor 0.80, watch band up to 0.80 + 0.20 * 0.2 = 0.84.
        assert _status_lower(0.82, 0.80, 0.8) == WATCH

    def test_lower_well_above_is_ok(self):
        assert _status_lower(0.99, 0.80, 0.8) == OK


class TestKpiStatuses:
    def test_healthy_report_all_ok(self):
        statuses = kpi_statuses(_report())
        assert set(statuses.values()) == {OK}

    def test_tenant_violation_alerts(self):
        statuses = kpi_statuses(_report(tenant_boundary_violations=1))
        assert statuses["tenant_boundary_violations"] == ALERT

    def test_security_debt_watches(self):
        statuses = kpi_statuses(_report(security_exception_debt=3))
        assert statuses["security_exception_debt"] == WATCH

    def test_per_domain_fan_out(self):
        statuses = kpi_statuses(
            _report(per_domain_false_positive_rate={"legal": 0.0, "med": 0.5})
        )
        assert statuses["false_positive_rate[legal]"] == OK
        assert statuses["false_positive_rate[med]"] == ALERT

    def test_explicit_targets_override_defaults(self):
        # A 0.05 FPR is ok by default but alerts under a strict 0.01 target.
        report = _report(false_positive_rate=0.05)
        strict = KpiTargets(max_false_positive_rate=0.01)
        assert kpi_statuses(report, strict)["false_positive_rate"] == ALERT


class TestOverallStatus:
    def test_alert_dominates(self):
        assert overall_status(_report(tenant_boundary_violations=1)) == ALERT

    def test_watch_when_no_alert(self):
        assert overall_status(_report(security_exception_debt=1)) == WATCH

    def test_ok_when_clean(self):
        assert overall_status(_report()) == OK


class TestFmt:
    def test_none(self):
        assert _fmt(None) == "n/a"

    def test_percent(self):
        assert _fmt(0.5, pct=True) == "50.00%"

    def test_suffix(self):
        assert _fmt(20.0, suffix=" ms") == "20 ms"


class TestRenderMarkdown:
    def test_includes_overall_and_table(self):
        out = render_markdown(_report(tenant_boundary_violations=1))
        assert "overall: ALERT" in out
        assert "| Metric | Value | Status |" in out
        assert "Tenant boundary violations | 1 | alert" in out

    def test_per_domain_section_present(self):
        out = render_markdown(
            _report(per_domain_false_positive_rate={"med": 0.5, "legal": 0.0})
        )
        assert "## Per-domain false-positive rate" in out
        # Sorted: legal before med.
        assert out.index("| legal |") < out.index("| med |")

    def test_per_domain_section_absent_when_empty(self):
        out = render_markdown(_report())
        assert "Per-domain" not in out


class TestRenderText:
    def test_full_text_render(self):
        out = render_text(_report(security_exception_debt=2))
        assert out.startswith("Guardrail KPIs (overall: watch)")
        assert "halt_precision: 95.00% [ok]" in out
        assert "security_exception_debt: 2 [watch]" in out

    def test_handles_missing_metrics(self):
        # No labelled items -> precision/FPR/latency are None ("n/a").
        report = compute_kpis([LabelItem("u1", 0.5, True)])
        out = render_text(report)
        assert "halt_precision: n/a" in out
        assert "false_positive_rate: n/a" in out
        assert "p95_scoring_latency: n/a" in out

    def test_not_available_status_propagates(self):
        report = compute_kpis([LabelItem("u1", 0.5, True)])
        statuses = kpi_statuses(report)
        assert statuses["halt_precision"] == NOT_AVAILABLE
        assert statuses["false_positive_rate"] == NOT_AVAILABLE
