# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — unified firewall tests
"""Tests for the unified ProductionGuard.firewall() decision.

Covers the composition logic with controlled guards (clean pass, each single
guard firing, several firing together), the injection/moderation toggles, the
detector override, the tenant-safe ``to_dict`` shape, and one end-to-end pass
with the real heuristic guard that catches a hallucination and a PII leak in one
call. Coherence and injection are stubbed for the logic tests so no model loads.
"""

from __future__ import annotations

from types import SimpleNamespace

from director_ai.core.config import DirectorConfig
from director_ai.core.safety.moderation import ModerationResult
from director_ai.guard import FirewallDecision, ProductionGuard


def _guard() -> ProductionGuard:
    return ProductionGuard(config=DirectorConfig(use_nli=False))


def _stub_check(approved: bool, score: float):
    return lambda prompt, response, **kw: SimpleNamespace(
        approved=approved, score=score, coherence=SimpleNamespace(score=score)
    )


def _stub_injection(detected: bool, risk: float):
    return lambda **kw: SimpleNamespace(
        injection_detected=detected, injection_risk=risk
    )


class _FlagDetector:
    """Moderation detector that flags any text under a fixed name."""

    def __init__(self, name: str, *, flag: bool = True):
        self._name = name
        self._flag = flag

    def analyse(self, text: str) -> ModerationResult:
        matches = [object()] if self._flag else []
        return ModerationResult(detector=self._name, matches=matches)  # type: ignore[arg-type]


class TestComposition:
    def test_all_clean_allows(self, monkeypatch):
        g = _guard()
        monkeypatch.setattr(g, "check", _stub_check(True, 0.95))
        monkeypatch.setattr(g, "check_injection", _stub_injection(False, 0.0))
        g.set_moderation_detectors([_FlagDetector("pii", flag=False)])
        d = g.firewall("q", "clean response")
        assert isinstance(d, FirewallDecision)
        assert d.blocked is False
        assert d.reasons == ()
        assert d.moderation_flags == ()

    def test_hallucination_blocks(self, monkeypatch):
        g = _guard()
        monkeypatch.setattr(g, "check", _stub_check(False, 0.1))
        monkeypatch.setattr(g, "check_injection", _stub_injection(False, 0.0))
        g.set_moderation_detectors([])
        d = g.firewall("q", "r")
        assert d.blocked is True
        assert d.approved is False
        assert any("hallucination" in r for r in d.reasons)

    def test_injection_blocks(self, monkeypatch):
        g = _guard()
        monkeypatch.setattr(g, "check", _stub_check(True, 0.95))
        monkeypatch.setattr(g, "check_injection", _stub_injection(True, 0.88))
        g.set_moderation_detectors([])
        d = g.firewall("q", "r")
        assert d.blocked is True
        assert d.injection_detected is True
        assert d.injection_risk == 0.88
        assert any("injection" in r for r in d.reasons)

    def test_moderation_blocks_and_names_flag(self, monkeypatch):
        g = _guard()
        monkeypatch.setattr(g, "check", _stub_check(True, 0.95))
        monkeypatch.setattr(g, "check_injection", _stub_injection(False, 0.0))
        g.set_moderation_detectors([_FlagDetector("toxicity")])
        d = g.firewall("q", "r")
        assert d.blocked is True
        assert d.moderation_flags == ("toxicity",)
        assert any("toxicity" in r for r in d.reasons)

    def test_multiple_guards_all_reported(self, monkeypatch):
        g = _guard()
        monkeypatch.setattr(g, "check", _stub_check(False, 0.2))
        monkeypatch.setattr(g, "check_injection", _stub_injection(True, 0.9))
        g.set_moderation_detectors([_FlagDetector("pii"), _FlagDetector("toxicity")])
        d = g.firewall("q", "r")
        assert d.blocked is True
        assert set(d.moderation_flags) == {"pii", "toxicity"}
        assert len(d.reasons) == 4  # hallucination + injection + 2 moderation


class TestToggles:
    def test_injection_skipped_when_disabled(self, monkeypatch):
        g = _guard()
        monkeypatch.setattr(g, "check", _stub_check(True, 0.95))

        def _boom(**kw):  # must not be called
            raise AssertionError("check_injection ran when disabled")

        monkeypatch.setattr(g, "check_injection", _boom)
        g.set_moderation_detectors([])
        d = g.firewall("q", "r", check_injection=False)
        assert d.injection_detected is False
        assert d.blocked is False

    def test_moderation_skipped_when_disabled(self, monkeypatch):
        g = _guard()
        monkeypatch.setattr(g, "check", _stub_check(True, 0.95))
        monkeypatch.setattr(g, "check_injection", _stub_injection(False, 0.0))
        g.set_moderation_detectors([_FlagDetector("pii")])  # would flag if run
        d = g.firewall("q", "r", moderate=False)
        assert d.moderation_flags == ()
        assert d.blocked is False


class TestSerialisation:
    def test_to_dict_is_tenant_safe(self, monkeypatch):
        g = _guard()
        monkeypatch.setattr(g, "check", _stub_check(False, 0.3))
        monkeypatch.setattr(g, "check_injection", _stub_injection(True, 0.7))
        g.set_moderation_detectors([_FlagDetector("pii")])
        data = g.firewall("secret prompt", "secret response with PII").to_dict()
        assert set(data) == {
            "blocked",
            "approved",
            "coherence_score",
            "injection_detected",
            "injection_risk",
            "moderation_flags",
            "reasons",
        }
        # No raw prompt/response text leaks into the summary.
        blob = str(data)
        assert "secret prompt" not in blob
        assert "secret response" not in blob


class TestDefaultDetectors:
    def test_default_detectors_catch_hallucination_and_pii(self):
        # End-to-end with the real heuristic guard (no NLI model): a contradictory
        # answer that also leaks a phone number must block on both axes.
        g = _guard()
        d = g.firewall(
            "What is the capital of France?",
            "The capital of France is Berlin. Call me at 555-123-4567.",
            check_injection=False,
        )
        assert d.blocked is True
        assert "pii_regex" in d.moderation_flags
