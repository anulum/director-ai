# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Cryptographic output integrity tests

"""Multi-angle tests for output integrity: tamper-evident ledger + Ed25519 signing.

The ledger tests are stdlib-only and always run; the signing tests require the
optional ``cryptography`` backend and skip cleanly without it. Covers chain
construction and tamper detection (payload, hash, reorder, index), signature
round-trip and forgery rejection (output, metadata, algorithm, public key),
deterministic seeds, the missing-backend error, and ProductionGuard wiring.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys

import pytest

from director_ai.core.output_integrity import (
    GENESIS_HASH,
    LedgerEntry,
    MissingCryptoBackendError,
    OutputIntegrityGuard,
    TamperEvidentLedger,
)
from director_ai.core.output_integrity import signing as signing_mod

requires_crypto = pytest.mark.skipif(
    importlib.util.find_spec("cryptography") is None,
    reason="Ed25519 signing requires the optional cryptography backend",
)


class TestTamperEvidentLedger:
    def test_empty_head_is_genesis(self):
        assert TamperEvidentLedger().head == GENESIS_HASH

    def test_append_chains_entries(self):
        ledger = TamperEvidentLedger()
        e0 = ledger.append({"output": "a"})
        e1 = ledger.append({"output": "b"})
        assert e0.index == 0 and e1.index == 1
        assert e0.prev_hash == GENESIS_HASH
        assert e1.prev_hash == e0.entry_hash
        assert ledger.head == e1.entry_hash

    def test_intact_chain_verifies(self):
        ledger = TamperEvidentLedger()
        for i in range(5):
            ledger.append({"output": f"o{i}", "metadata": {"i": i}})
        assert ledger.verify() is True

    def test_distinct_payloads_distinct_digests(self):
        ledger = TamperEvidentLedger()
        a = ledger.append({"output": "x"})
        b = ledger.append({"output": "y"})
        assert a.payload_digest != b.payload_digest

    def test_tampered_payload_digest_detected(self):
        ledger = TamperEvidentLedger()
        ledger.append({"output": "a"})
        ledger.append({"output": "b"})
        entries = list(ledger.entries)
        entries[0] = dataclasses.replace(entries[0], payload_digest="ff" * 32)
        assert TamperEvidentLedger.verify_entries(entries) is False

    def test_tampered_entry_hash_detected(self):
        ledger = TamperEvidentLedger()
        ledger.append({"output": "a"})
        entries = list(ledger.entries)
        entries[0] = dataclasses.replace(entries[0], entry_hash="ab" * 32)
        assert TamperEvidentLedger.verify_entries(entries) is False

    def test_broken_prev_link_detected(self):
        ledger = TamperEvidentLedger()
        ledger.append({"output": "a"})
        ledger.append({"output": "b"})
        entries = list(ledger.entries)
        entries[1] = dataclasses.replace(entries[1], prev_hash="00" * 32)
        assert TamperEvidentLedger.verify_entries(entries) is False

    def test_reordered_index_detected(self):
        ledger = TamperEvidentLedger()
        ledger.append({"output": "a"})
        ledger.append({"output": "b"})
        entries = list(reversed(ledger.entries))
        assert TamperEvidentLedger.verify_entries(entries) is False

    def test_entry_to_dict(self):
        entry = TamperEvidentLedger().append({"output": "a"})
        d = entry.to_dict()
        assert set(d) == {"index", "payload_digest", "prev_hash", "entry_hash"}

    def test_empty_ledger_verifies(self):
        assert TamperEvidentLedger().verify() is True


@requires_crypto
class TestEd25519Signing:
    def test_sign_verify_roundtrip(self):
        signer = signing_mod.OutputSigner()
        signed = signer.sign("The sky is blue.", {"tenant": "t1"})
        assert signed.algorithm == "ed25519"
        assert signing_mod.verify_signed_output(signed) is True

    def test_tampered_output_rejected(self):
        signed = signing_mod.OutputSigner().sign("original", {"a": 1})
        forged = dataclasses.replace(signed, output="forged")
        assert signing_mod.verify_signed_output(forged) is False

    def test_tampered_metadata_rejected(self):
        signed = signing_mod.OutputSigner().sign("o", {"role": "user"})
        forged = dataclasses.replace(signed, metadata={"role": "admin"})
        assert signing_mod.verify_signed_output(forged) is False

    def test_wrong_algorithm_rejected(self):
        signed = signing_mod.OutputSigner().sign("o")
        assert (
            signing_mod.verify_signed_output(
                dataclasses.replace(signed, algorithm="rsa")
            )
            is False
        )

    def test_malformed_signature_rejected(self):
        signed = signing_mod.OutputSigner().sign("o")
        assert (
            signing_mod.verify_signed_output(
                dataclasses.replace(signed, signature="zz")
            )
            is False
        )

    def test_seed_must_be_32_bytes(self):
        with pytest.raises(ValueError, match="32 bytes"):
            signing_mod.OutputSigner(seed=b"short")

    def test_seed_gives_stable_identity(self):
        seed = bytes(range(32))
        assert (
            signing_mod.OutputSigner(seed=seed).public_key_hex
            == signing_mod.OutputSigner(seed=seed).public_key_hex
        )

    def test_sign_rejects_non_string(self):
        with pytest.raises(TypeError, match="must be a string"):
            signing_mod.OutputSigner().sign(123)

    def test_signed_output_to_dict(self):
        d = signing_mod.OutputSigner().sign("o", {"k": "v"}).to_dict()
        assert set(d) == {"output", "metadata", "signature", "public_key", "algorithm"}


class TestMissingBackend:
    def test_missing_cryptography_raises(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "cryptography.hazmat.primitives.asymmetric", None
        )
        with pytest.raises(MissingCryptoBackendError, match="director-ai\\[crypto\\]"):
            signing_mod._load_ed25519()


@requires_crypto
class TestOutputIntegrityGuard:
    def test_sign_and_verify(self):
        guard = OutputIntegrityGuard()
        signed = guard.sign("answer", {"model": "x"})
        assert guard.verify(signed) is True
        assert len(guard.public_key_hex) == 64

    def test_record_appends_digest_only(self):
        guard = OutputIntegrityGuard()
        entry = guard.record("sensitive answer", {"tenant": "t1"})
        assert isinstance(entry, LedgerEntry)
        # Only a digest is retained, never the raw output.
        assert "sensitive" not in entry.payload_digest
        assert guard.verify_ledger() is True

    def test_ledger_property_exposed(self):
        guard = OutputIntegrityGuard()
        guard.record("a")
        assert len(guard.ledger.entries) == 1

    def test_seeded_guard_stable_key(self):
        seed = bytes(range(31, 63))
        a = OutputIntegrityGuard(signing_seed=seed)
        b = OutputIntegrityGuard(signing_seed=seed)
        assert a.public_key_hex == b.public_key_hex


@requires_crypto
class TestGuardWiring:
    def test_production_guard_exposes_output_integrity(self):
        from director_ai.core.config import DirectorConfig
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
        oi = guard.output_integrity()
        assert isinstance(oi, OutputIntegrityGuard)
        assert guard.output_integrity() is oi  # cached
        signed = oi.sign("hello")
        assert oi.verify(signed) is True
