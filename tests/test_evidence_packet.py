# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the one-command evidence packet.

Covers packet assembly from a guard (Answer BOM + eval-trace per decision),
SHA-256 integrity sealing, tamper detection, demo-expectation verification, and
the bundled demo knowledge base. The guard's scoring is driven deterministically
so the test does not depend on a model-backed scorer being installed.
"""

from __future__ import annotations

from director_ai.core.evidence_packet import (
    DEMO_FACTS,
    EVIDENCE_PACKET_VERSION,
    build_evidence_packet,
    verify_evidence_packet,
)
from director_ai.core.types import CoherenceScore
from director_ai.guard import GuardResult, ProductionGuard


def _coherence(score: float) -> CoherenceScore:
    return CoherenceScore(
        score=score, approved=score >= 0.6, h_logical=0.1, h_factual=0.1
    )


def _demo_guard() -> ProductionGuard:
    """Real guard with check() driven to approve grounded, block hallucinated."""
    guard = ProductionGuard()
    results = iter(
        [
            GuardResult(approved=True, score=0.92, coherence=_coherence(0.92)),
            GuardResult(approved=False, score=0.20, coherence=_coherence(0.20)),
        ]
    )
    guard.check = lambda *a, **k: next(results)  # type: ignore[method-assign]
    return guard


class TestBuildPacket:
    def test_packet_structure(self):
        packet = build_evidence_packet(_demo_guard())
        content = packet["content"]
        assert content["schema_version"] == EVIDENCE_PACKET_VERSION
        assert content["knowledge_base_size"] == len(DEMO_FACTS)
        assert content["checks"]["grounded_approved"] is True
        assert content["checks"]["hallucinated_blocked"] is True
        assert packet["integrity"]["algorithm"] == "sha256"
        assert len(packet["integrity"]["digest"]) == 64

    def test_packet_carries_bom_and_trace(self):
        content = build_evidence_packet(_demo_guard())["content"]
        assert "answer_bom" in content["grounded"]
        assert "eval_trace" in content["grounded"]
        assert content["grounded"]["eval_trace"]["gen_ai.system"] == "director-ai"
        assert content["grounded"]["eval_trace"]["director.eval.decision"] == "allow"
        assert content["hallucinated"]["eval_trace"]["director.eval.decision"] == "halt"

    def test_custom_facts(self):
        packet = build_evidence_packet(_demo_guard(), facts={"a": "x", "b": "y"})
        assert packet["content"]["knowledge_base_size"] == 2


class TestVerifyPacket:
    def test_valid_packet_verifies(self):
        packet = build_evidence_packet(_demo_guard())
        ok, reason = verify_evidence_packet(packet)
        assert ok
        assert reason == "ok"

    def test_tamper_detected(self):
        packet = build_evidence_packet(_demo_guard())
        packet["content"]["question"] = "tampered"
        ok, reason = verify_evidence_packet(packet)
        assert not ok
        assert reason == "digest_mismatch"

    def test_malformed_packet(self):
        ok, reason = verify_evidence_packet({"nope": True})
        assert not ok
        assert reason == "malformed_packet"

    def test_unsupported_version(self):
        packet = build_evidence_packet(_demo_guard())
        packet["content"]["schema_version"] = "director.evidence_packet.v0"
        # Recompute digest so it is the version check, not the digest, that trips.
        from director_ai.core.evidence_packet.packet import _canonical_digest

        packet["integrity"]["digest"] = _canonical_digest(packet["content"])
        ok, reason = verify_evidence_packet(packet)
        assert not ok
        assert reason == "unsupported_schema_version"

    def test_grounded_not_approved_fails(self):
        guard = ProductionGuard()
        results = iter(
            [
                GuardResult(approved=False, score=0.2, coherence=_coherence(0.2)),
                GuardResult(approved=False, score=0.2, coherence=_coherence(0.2)),
            ]
        )
        guard.check = lambda *a, **k: next(results)  # type: ignore[method-assign]
        packet = build_evidence_packet(guard)
        ok, reason = verify_evidence_packet(packet)
        assert not ok
        assert reason == "grounded_not_approved"

    def test_hallucinated_not_blocked_fails(self):
        guard = ProductionGuard()
        results = iter(
            [
                GuardResult(approved=True, score=0.9, coherence=_coherence(0.9)),
                GuardResult(approved=True, score=0.9, coherence=_coherence(0.9)),
            ]
        )
        guard.check = lambda *a, **k: next(results)  # type: ignore[method-assign]
        packet = build_evidence_packet(guard)
        ok, reason = verify_evidence_packet(packet)
        assert not ok
        assert reason == "hallucinated_not_blocked"


class TestDemoFacts:
    def test_demo_facts_nonempty(self):
        assert len(DEMO_FACTS) >= 5
        assert all(
            isinstance(k, str) and isinstance(v, str) for k, v in DEMO_FACTS.items()
        )


class TestCli:
    def test_evidence_emit_and_verify(self, tmp_path, monkeypatch, capsys):
        import director_ai.guard as guard_mod
        from director_ai.cli import main

        monkeypatch.setattr(
            guard_mod.ProductionGuard,
            "from_profile",
            staticmethod(lambda *a, **k: _demo_guard()),
        )
        out = tmp_path / "evidence"
        main(["evidence", "--emit", str(out)])
        assert (out / "evidence_packet.json").exists()
        assert "grounded answer approved:    True" in capsys.readouterr().out

        main(["verify-evidence", str(out)])
        assert "VERIFIED" in capsys.readouterr().out

    def test_verify_evidence_missing(self, tmp_path):
        import pytest

        from director_ai.cli import main

        with pytest.raises(SystemExit) as exc:
            main(["verify-evidence", str(tmp_path / "nope.json")])
        assert exc.value.code == 1

    def test_verify_evidence_tampered(self, tmp_path, capsys):
        import json

        import pytest

        from director_ai.cli import main

        packet = build_evidence_packet(_demo_guard())
        packet["content"]["question"] = "tampered"
        path = tmp_path / "evidence_packet.json"
        path.write_text(json.dumps(packet), encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            main(["verify-evidence", str(path)])
        assert exc.value.code == 1
        assert "INVALID" in capsys.readouterr().out
