# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``director_ai.compliance.cost_analyser``.

Covers pricing, recording, cost estimation, reporting, per-agent
attribution, thread safety, and edge cases.
"""

from __future__ import annotations

import threading

from director_ai.compliance.cost_analyser import (
    CostAnalyser,
    CostRecord,
    ModelPricing,
)

# ── ModelPricing dataclass ─────────────────────────────────────────────


class TestModelPricing:
    def test_frozen(self):
        p = ModelPricing("gpt-4o", 0.0025, 0.01)
        assert p.model == "gpt-4o"
        assert p.input_per_1k == 0.0025

    def test_different_models(self):
        p1 = ModelPricing("a", 0.001, 0.002)
        p2 = ModelPricing("b", 0.003, 0.004)
        assert p1.model != p2.model


# ── CostRecord dataclass ──────────────────────────────────────────────


class TestCostRecord:
    def test_total_tokens(self):
        r = CostRecord(model="m", total_input_tokens=100, total_output_tokens=50)
        assert r.total_tokens == 150

    def test_defaults(self):
        r = CostRecord(model="m")
        assert r.call_count == 0
        assert r.total_tokens == 0


# ── Construction ───────────────────────────────────────────────────────


class TestConstruction:
    def test_default_currency(self):
        a = CostAnalyser()
        assert a._currency == "CHF"

    def test_custom_currency(self):
        a = CostAnalyser(currency="EUR")
        assert a._currency == "EUR"

    def test_default_pricing_loaded(self):
        a = CostAnalyser()
        assert "gpt-4o" in a._pricing
        assert "claude-3-5-sonnet" in a._pricing


# ── Pricing management ─────────────────────────────────────────────────


class TestPricing:
    def test_add_pricing(self):
        a = CostAnalyser()
        a.add_pricing("custom-model", 0.01, 0.02)
        assert "custom-model" in a._pricing

    def test_override_pricing(self):
        a = CostAnalyser()
        a.add_pricing("gpt-4o", 0.005, 0.02)
        assert a._pricing["gpt-4o"].input_per_1k == 0.005


# ── Recording ──────────────────────────────────────────────────────────


class TestRecording:
    def test_single_record(self):
        a = CostAnalyser()
        a.record("gpt-4o", input_tokens=500, output_tokens=100)
        r = a.report()
        assert r["total_tokens"] == 600

    def test_multiple_records_same_model(self):
        a = CostAnalyser()
        a.record("gpt-4o", input_tokens=500, output_tokens=100)
        a.record("gpt-4o", input_tokens=300, output_tokens=50)
        r = a.report()
        assert r["models"]["gpt-4o"]["call_count"] == 2
        assert r["models"]["gpt-4o"]["input_tokens"] == 800

    def test_multiple_models(self):
        a = CostAnalyser()
        a.record("gpt-4o", input_tokens=500, output_tokens=100)
        a.record("claude-3-5-sonnet", input_tokens=300, output_tokens=50)
        r = a.report()
        assert len(r["models"]) == 2

    def test_per_agent_attribution(self):
        a = CostAnalyser()
        a.record("gpt-4o", input_tokens=100, output_tokens=50, agent_id="agent-1")
        a.record("gpt-4o", input_tokens=200, output_tokens=80, agent_id="agent-2")
        r = a.report()
        assert "gpt-4o::agent-1" in r["models"]
        assert "gpt-4o::agent-2" in r["models"]
        assert r["models"]["gpt-4o::agent-1"]["input_tokens"] == 100


# ── Cost estimation ────────────────────────────────────────────────────


class TestCostEstimation:
    def test_estimate_gpt4o(self):
        a = CostAnalyser()
        cost = a.estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        # 1K input × 0.0025 + 0.5K output × 0.01 = 0.0025 + 0.005 = 0.0075
        assert abs(cost - 0.0075) < 1e-6

    def test_estimate_unknown_model(self):
        a = CostAnalyser()
        cost = a.estimate_cost("unknown-model", input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    def test_report_total_cost(self):
        a = CostAnalyser()
        a.record("gpt-4o", input_tokens=1000, output_tokens=500)
        r = a.report()
        assert r["total_cost"] > 0

    def test_report_currency(self):
        a = CostAnalyser(currency="EUR")
        r = a.report()
        assert r["currency"] == "EUR"


# ── Report ─────────────────────────────────────────────────────────────


class TestReport:
    def test_empty_report(self):
        a = CostAnalyser()
        r = a.report()
        assert r["total_cost"] == 0
        assert r["total_tokens"] == 0
        assert r["models"] == {}

    def test_report_structure(self):
        a = CostAnalyser()
        a.record("gpt-4o", input_tokens=100, output_tokens=50)
        r = a.report()
        assert "currency" in r
        assert "total_cost" in r
        assert "total_tokens" in r
        assert "models" in r
        m = r["models"]["gpt-4o"]
        assert "estimated_cost" in m
        assert "call_count" in m


# ── Reset ──────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_records(self):
        a = CostAnalyser()
        a.record("gpt-4o", input_tokens=100, output_tokens=50)
        a.reset()
        r = a.report()
        assert r["total_tokens"] == 0

    def test_reset_preserves_pricing(self):
        a = CostAnalyser()
        a.add_pricing("custom", 0.01, 0.02)
        a.reset()
        assert "custom" in a._pricing


# ── Thread safety ──────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_recording(self):
        a = CostAnalyser()
        errors: list[Exception] = []

        def record_batch() -> None:
            try:
                for _ in range(500):
                    a.record("gpt-4o", input_tokens=10, output_tokens=5)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=record_batch) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        r = a.report()
        assert r["models"]["gpt-4o"]["call_count"] == 2000
        assert r["total_tokens"] == 30000  # 2000 × (10+5)
