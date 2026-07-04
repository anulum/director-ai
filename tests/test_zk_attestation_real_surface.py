# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ZK attestation real-surface tests
"""Real public-surface coverage for cross-org attestation passports."""

from __future__ import annotations

from collections.abc import Mapping

from director_ai.core.zk_attestation import (
    CommitmentBackend,
    MaximumHaltRate,
    NoBreakoutEvents,
    PassportIssuer,
    PassportVerifier,
)
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

_KEY = b"z" * 32
_ISSUER = "org://source.example"
_AGENT = "agent://support-42"
_SAMPLES: tuple[Mapping[str, object], ...] = (
    {"halted": False, "breakout": False, "domain": "support"},
    {"halted": False, "breakout": False, "domain": "support"},
    {"halted": False, "breakout": False, "domain": "billing"},
    {"halted": True, "breakout": False, "domain": "billing"},
)


def _backend() -> CommitmentBackend:
    """Return the public commitment backend used by issuer and verifier."""
    return CommitmentBackend(key=_KEY, challenge_size=2)


def _statements() -> tuple[NoBreakoutEvents, MaximumHaltRate]:
    """Return spot-check-sound passport statements for the sample history."""
    return (
        NoBreakoutEvents(name="no_breakouts", samples_min=4),
        MaximumHaltRate(name="halt_budget", max_rate=0.25, samples_min=4),
    )


def test_zk_attestation_unit_guard_has_real_surface_companion() -> None:
    """The helper-heavy ZK attestation guard needs public workflow coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_zk_attestation.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_zk_attestation_real_surface.py" in category


def test_public_passport_issue_and_verify_accepts_safe_handoff() -> None:
    """A real issuer/verifier pair should accept a proved safe handoff."""
    backend = _backend()
    issuer = PassportIssuer(
        key=_KEY,
        issuing_org=_ISSUER,
        default_backend=backend,
        clock=lambda: 1_800_000_000.0,
    )
    passport = issuer.issue(
        agent_id=_AGENT,
        samples=_SAMPLES,
        statements=_statements(),
    )
    verifier = PassportVerifier(
        issuer_keys={_ISSUER: _KEY},
        backends={backend.kind: backend},
    )

    verdict = verifier.verify(passport)

    assert passport.agent_id == _AGENT
    assert passport.issuing_org == _ISSUER
    assert passport.created_at == 1_800_000_000
    assert {entry.backend_kind for entry in passport.entries} == {"commitment"}
    assert verdict.accepted is True
    assert verdict.signature_ok is True
    assert verdict.failures == ()
    assert verdict.summary() == "all statements proved"
    assert verdict.safety_event is not None
    assert verdict.safety_event.hook_id == "zk_attestation.passport"
    assert verdict.safety_event.policy_decision == "allow"
    assert verdict.safety_event.evidence_refs == ()
    assert verdict.safety_event.attributes == {
        "failure_count": "0",
        "signature_ok": "true",
    }


def test_public_passport_verifier_blocks_bad_issuer_key() -> None:
    """A receiver with the wrong issuer key should fail closed."""
    backend = _backend()
    issuer = PassportIssuer(
        key=_KEY,
        issuing_org=_ISSUER,
        default_backend=backend,
        clock=lambda: 1_800_000_000.0,
    )
    passport = issuer.issue(
        agent_id=_AGENT,
        samples=_SAMPLES,
        statements=_statements(),
    )
    verifier = PassportVerifier(
        issuer_keys={_ISSUER: b"w" * 32},
        backends={backend.kind: backend},
    )

    verdict = verifier.verify(passport)

    assert verdict.accepted is False
    assert verdict.signature_ok is False
    assert verdict.failures == (("_passport", "mac_mismatch"),)
    assert verdict.summary() == "passport signature failed"
    assert verdict.safety_event is not None
    assert verdict.safety_event.policy_decision == "block"
    assert verdict.safety_event.evidence_refs == ("attestation:_passport:mac_mismatch",)
    assert verdict.safety_event.attributes == {
        "failure_count": "1",
        "signature_ok": "false",
    }
