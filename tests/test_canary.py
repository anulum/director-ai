# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for counterfactual canary facts.

Covers registry minting and tenant scoping, leakage and citation detection,
alerting, metrics, and the ProductionGuard plant/scan integration.
"""

from __future__ import annotations

import itertools

import pytest

from director_ai.core.canary import (
    CANARY_FLAG,
    CanaryDetector,
    CanaryRegistry,
    CanarySignal,
)
from director_ai.core.metrics import metrics


def _registry() -> CanaryRegistry:
    counter = itertools.count()
    return CanaryRegistry(
        token_factory=lambda: f"CANARY-{next(counter):04d}",
        clock=lambda: "2026-06-06T00:00:00Z",
    )


# ── registry ────────────────────────────────────────────────────────────


class TestRegistry:
    def test_mint_embeds_token_in_text(self):
        fact = _registry().mint("acme")
        assert fact.token in fact.text
        assert fact.canary_id.startswith("canary_")
        assert fact.tenant_id == "acme"

    def test_mint_requires_tenant(self):
        with pytest.raises(ValueError, match="tenant_id is required"):
            _registry().mint("  ")

    def test_mint_requires_token_placeholder(self):
        with pytest.raises(ValueError, match=r"must contain '\{token\}'"):
            _registry().mint("acme", template="no placeholder here")

    def test_mint_rejects_blank_explicit_token(self):
        with pytest.raises(ValueError, match="token must be non-empty"):
            _registry().mint("acme", token="   ")

    def test_explicit_token_used(self):
        fact = _registry().mint("acme", token="CANARY-XYZ")
        assert fact.token == "CANARY-XYZ"
        assert "CANARY-XYZ" in fact.text

    def test_tenant_isolation(self):
        reg = _registry()
        reg.mint("acme")
        reg.mint("globex")
        assert len(reg.facts_for("acme")) == 1
        assert len(reg.facts_for("globex")) == 1
        assert reg.facts_for("unknown") == ()

    def test_tokens_for(self):
        reg = _registry()
        reg.mint("acme", token="CANARY-A")
        reg.mint("acme", token="CANARY-B")
        assert set(reg.tokens_for("acme")) == {"CANARY-A", "CANARY-B"}

    def test_find_by_id(self):
        reg = _registry()
        fact = reg.mint("acme")
        assert reg.find(fact.canary_id) is fact
        assert reg.find("canary_missing") is None

    def test_metadata_shape(self):
        fact = _registry().mint("acme")
        meta = fact.metadata()
        assert meta[CANARY_FLAG] is True
        assert meta["kb_canary_id"] == fact.canary_id
        assert meta["tenant_id"] == "acme"

    def test_default_clock_and_token(self):
        # Default registry uses the real UTC clock and a random token.
        fact = CanaryRegistry().mint("acme")
        assert fact.created_at.endswith("Z")
        assert fact.token.startswith("CANARY-")


# ── detector ────────────────────────────────────────────────────────────


class TestDetector:
    def test_leakage_detected(self):
        reg = _registry()
        fact = reg.mint("acme", token="CANARY-LEAK")
        signals = CanaryDetector(reg).scan_answer(
            "The secret marker CANARY-LEAK slipped out.", "acme"
        )
        assert len(signals) == 1
        assert signals[0].signal == "leakage"
        assert signals[0].canary_id == fact.canary_id

    def test_clean_answer_no_signal(self):
        reg = _registry()
        reg.mint("acme", token="CANARY-LEAK")
        assert CanaryDetector(reg).scan_answer("a normal answer", "acme") == []

    def test_leakage_tenant_isolated(self):
        reg = _registry()
        reg.mint("globex", token="CANARY-GLOBEX")
        # acme scan must not match globex's token.
        assert CanaryDetector(reg).scan_answer("CANARY-GLOBEX", "acme") == []

    def test_citation_detected(self):
        reg = _registry()
        fact = reg.mint("acme")
        chunks = [{"id": "c1", "text": "t", "metadata": fact.metadata()}]
        signals = CanaryDetector(reg).scan_evidence(chunks, "acme")
        assert len(signals) == 1
        assert signals[0].signal == "citation"
        assert signals[0].canary_id == fact.canary_id

    def test_non_canary_chunk_ignored(self):
        reg = _registry()
        reg.mint("acme")
        chunks = [{"id": "c1", "text": "t", "metadata": {"tenant_id": "acme"}}]
        assert CanaryDetector(reg).scan_evidence(chunks, "acme") == []

    def test_unregistered_canary_chunk_ignored(self):
        reg = _registry()
        reg.mint("acme")
        chunks = [
            {
                "id": "c1",
                "text": "t",
                "metadata": {CANARY_FLAG: True, "kb_canary_id": "canary_other"},
            }
        ]
        assert CanaryDetector(reg).scan_evidence(chunks, "acme") == []

    def test_non_mapping_metadata_ignored(self):
        reg = _registry()
        reg.mint("acme")
        chunks = [{"id": "c1", "text": "t", "metadata": "not-a-dict"}]
        assert CanaryDetector(reg).scan_evidence(chunks, "acme") == []

    def test_missing_metadata_ignored(self):
        reg = _registry()
        reg.mint("acme")
        assert (
            CanaryDetector(reg).scan_evidence([{"id": "c1", "text": "t"}], "acme") == []
        )

    def test_scan_combines_both(self):
        reg = _registry()
        fact = reg.mint("acme", token="CANARY-BOTH")
        chunks = [{"id": "c1", "text": "t", "metadata": fact.metadata()}]
        signals = CanaryDetector(reg).scan(
            "leaked CANARY-BOTH", "acme", evidence=chunks
        )
        assert {s.signal for s in signals} == {"leakage", "citation"}

    def test_alert_fired_per_signal(self):
        reg = _registry()
        reg.mint("acme", token="CANARY-A")
        seen: list[CanarySignal] = []
        detector = CanaryDetector(reg, alert=seen.append)
        detector.scan_answer("CANARY-A here", "acme")
        assert len(seen) == 1
        assert seen[0].signal == "leakage"

    def test_metrics_counted(self):
        metrics.reset()
        reg = _registry()
        reg.mint("acme", token="CANARY-M")
        CanaryDetector(reg).scan_answer("CANARY-M", "acme")
        snapshot = metrics.get_metrics()
        counter = snapshot["counters"]["canary_signals_total"]
        assert counter["multi_labels"].get('signal="leakage"') == 1.0

    def test_signal_to_dict(self):
        payload = CanarySignal("canary_1", "acme", "leakage").to_dict()
        assert payload == {
            "canary_id": "canary_1",
            "tenant_id": "acme",
            "signal": "leakage",
        }


# ── guard integration ───────────────────────────────────────────────────


class TestGuardIntegration:
    def test_plant_and_detect_leakage(self):
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard()
        fact = guard.plant_canary("acme", token="CANARY-PLANT")
        signals = guard.scan_canaries(f"oops {fact.token} leaked", "acme")
        assert [s.signal for s in signals] == ["leakage"]

    def test_clean_response_no_signal(self):
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard()
        guard.plant_canary("acme", token="CANARY-PLANT")
        assert guard.scan_canaries("a clean grounded answer", "acme") == []

    def test_citation_via_evidence(self):
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard()
        fact = guard.plant_canary("acme", token="CANARY-PLANT")
        evidence = [{"id": "c1", "text": "t", "metadata": fact.metadata()}]
        signals = guard.scan_canaries("clean", "acme", evidence=evidence)
        assert [s.signal for s in signals] == ["citation"]

    def test_planted_canary_text_in_store(self):
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard()
        fact = guard.plant_canary("acme", token="CANARY-PLANT")
        # The canary text was added to the KB so retrieval can surface it under
        # attack.
        assert fact.text in guard._store.facts.values()
