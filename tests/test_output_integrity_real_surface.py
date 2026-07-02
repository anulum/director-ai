# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — output-integrity real-surface tests
"""Real public-facade coverage for cryptographic output integrity."""

from __future__ import annotations

import dataclasses
from typing import cast

import pytest

pytest.importorskip(
    "cryptography",
    reason="Ed25519 output-integrity real-surface coverage requires cryptography",
)

import director_ai
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig
from director_ai.core.output_integrity import (
    SignedOutput,
    TamperEvidentLedger,
    verify_signed_output,
)
from director_ai.guard import GuardResult
from director_ai.guard import ProductionGuard as GuardModuleFacade
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _guard_config() -> DirectorConfig:
    """Return a local production-guard configuration for facade coverage."""
    return DirectorConfig(use_nli=False, llm_provider="mock")


def test_output_integrity_unit_guard_has_real_surface_companion() -> None:
    """The output-integrity unit guard should have public facade coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_output_integrity.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_output_integrity_real_surface.py" in category


def test_package_root_exposes_documented_production_guard_facade() -> None:
    """The package root should expose the documented ProductionGuard facade."""
    assert ProductionGuard is GuardModuleFacade
    assert director_ai.GuardResult is GuardResult
    assert "ProductionGuard" in director_ai.__all__
    assert "GuardResult" in director_ai.__all__


def test_package_root_reports_moved_and_unknown_symbols() -> None:
    """The package root should keep explicit errors for invalid lazy exports."""
    with pytest.raises(ImportError, match="moved to director_ai.enterprise"):
        director_ai.__getattr__("TenantRouter")

    with pytest.raises(AttributeError, match="NoSuchFacade"):
        director_ai.__getattr__("NoSuchFacade")


def test_production_guard_output_integrity_signs_records_and_verifies_exports() -> None:
    """ProductionGuard should expose signing and ledger integrity end to end."""
    integrity = ProductionGuard(_guard_config()).output_integrity(
        signing_seed=bytes(range(32))
    )
    metadata = {
        "tenant": "tenant-alpha",
        "request_id": "req-output-integrity-001",
        "model": "factcg-local",
    }
    output = "Bratislava is the capital of Slovakia."

    signed = integrity.sign(output, metadata)
    entry = integrity.record(output, metadata)

    signed_payload = signed.to_dict()
    entry_payload = entry.to_dict()

    assert integrity.verify(signed) is True
    assert verify_signed_output(signed) is True
    assert integrity.verify_ledger() is True
    assert TamperEvidentLedger.verify_entries(integrity.ledger.entries) is True
    assert signed_payload["output"] == output
    assert signed_payload["metadata"] == metadata
    assert cast(str, signed_payload["public_key"]) == integrity.public_key_hex
    assert entry_payload["index"] == 0
    assert output not in cast(str, entry_payload["payload_digest"])
    assert metadata["tenant"] not in cast(str, entry_payload["payload_digest"])


def test_public_output_integrity_verifier_rejects_tampered_envelopes() -> None:
    """Public verification should reject altered signed-output envelopes."""
    integrity = ProductionGuard(_guard_config()).output_integrity(
        signing_seed=bytes(range(32, 64))
    )
    signed = integrity.sign("Release claim: local packet only.", {"claim": "local"})

    output_tamper = dataclasses.replace(signed, output="Release claim: public ready.")
    metadata_tamper = dataclasses.replace(signed, metadata={"claim": "public"})
    key_tamper = dataclasses.replace(signed, public_key="00" * 32)
    envelope_tamper = SignedOutput(**{**signed.to_dict(), "algorithm": "rsa"})

    assert integrity.verify(output_tamper) is False
    assert integrity.verify(metadata_tamper) is False
    assert integrity.verify(key_tamper) is False
    assert verify_signed_output(envelope_tamper) is False
