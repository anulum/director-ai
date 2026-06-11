# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the board-level KPI computation."""

from __future__ import annotations

from director_ai.core.labelling_cockpit import LabelItem
from director_ai.core.observability import KpiReport, compute_kpis


def _items() -> list[LabelItem]:
    # grounded should pass (high score, approved); hallucination should halt.
    return [
        LabelItem("g1", 0.9, True, domain="finance", label="grounded"),
        LabelItem("g2", 0.4, False, domain="finance", label="grounded"),  # false halt
        LabelItem("h1", 0.2, False, domain="finance", label="hallucination"),  # correct
        LabelItem("h2", 0.8, True, domain="medical", label="hallucination"),  # missed
        LabelItem("u1", 0.5, True),  # unlabelled -> ignored
    ]


class TestComputeKpis:
    def test_counts_only_labelled(self):
        report = compute_kpis(_items())
        assert report.labelled_total == 4

    def test_halt_rate(self):
        # 2 of 4 labelled were halted (g2, h1).
        assert compute_kpis(_items()).halt_rate == 0.5

    def test_halt_precision(self):
        # Of 2 halts, 1 was a real hallucination (h1) -> 0.5.
        assert compute_kpis(_items()).halt_precision == 0.5

    def test_false_positive_rate(self):
        # 2 grounded; 1 wrongly halted (g2) -> 0.5.
        assert compute_kpis(_items()).false_positive_rate == 0.5

    def test_per_domain_fpr(self):
        report = compute_kpis(_items())
        # finance has g1 (pass) + g2 (false halt) -> 0.5; medical has no grounded.
        assert report.per_domain_false_positive_rate == {"finance": 0.5}

    def test_p95_latency(self):
        report = compute_kpis(_items(), latency_ms_samples=[10, 20, 30, 40, 100])
        assert report.p95_scoring_latency_ms == 100.0

    def test_p95_none_without_samples(self):
        assert compute_kpis(_items()).p95_scoring_latency_ms is None

    def test_counters_passed_through(self):
        report = compute_kpis(
            _items(),
            tenant_boundary_violations=3,
            unsigned_kb_writes_rejected=7,
            security_exception_debt=2,
        )
        assert report.tenant_boundary_violations == 3
        assert report.unsigned_kb_writes_rejected == 7
        assert report.security_exception_debt == 2

    def test_empty_input(self):
        report = compute_kpis([])
        assert report.labelled_total == 0
        assert report.halt_rate == 0.0
        assert report.halt_precision is None
        assert report.false_positive_rate is None

    def test_no_halts_precision_none(self):
        items = [LabelItem("g1", 0.9, True, label="grounded")]
        assert compute_kpis(items).halt_precision is None

    def test_to_dict_shape(self):
        payload = compute_kpis(_items()).to_dict()
        assert payload["labelled_total"] == 4
        assert "per_domain_false_positive_rate" in payload
        assert isinstance(payload, dict)

    def test_report_is_frozen(self):
        report = compute_kpis(_items())
        assert isinstance(report, KpiReport)
        import dataclasses

        try:
            report.halt_rate = 0.1  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError("KpiReport must be frozen")
