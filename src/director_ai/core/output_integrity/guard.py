# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Output integrity guard

"""Compose Ed25519 signing with the tamper-evident ledger.

:class:`OutputIntegrityGuard` is the single entry point for output integrity: it
signs an output for non-repudiation and records its digest in an append-only
tamper-evident ledger for an audit trail. The ledger is stdlib-only and always
available; signing lazily loads the optional ``cryptography`` backend, so a
caller can keep an audit ledger even where signing is not configured.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .ledger import LedgerEntry, TamperEvidentLedger
from .signing import OutputSigner, SignedOutput, verify_signed_output

__all__ = ["OutputIntegrityGuard"]


class OutputIntegrityGuard:
    """Sign outputs and append their digests to a tamper-evident ledger."""

    def __init__(self, *, signing_seed: bytes | None = None):
        self._signer = OutputSigner(seed=signing_seed)
        self._ledger = TamperEvidentLedger()

    @property
    def ledger(self) -> TamperEvidentLedger:
        """The append-only tamper-evident ledger of recorded outputs."""
        return self._ledger

    @property
    def public_key_hex(self) -> str:
        """The signer's public key (hex) for distribution to verifiers."""
        return self._signer.public_key_hex

    def sign(
        self, output: str, metadata: Mapping[str, Any] | None = None
    ) -> SignedOutput:
        """Sign ``output`` for non-repudiation (requires the crypto backend)."""
        return self._signer.sign(output, metadata)

    def verify(self, signed: SignedOutput) -> bool:
        """Verify a :class:`SignedOutput` against its embedded public key."""
        return verify_signed_output(signed)

    def record(
        self, output: str, metadata: Mapping[str, Any] | None = None
    ) -> LedgerEntry:
        """Append a tenant-safe digest of ``output`` + ``metadata`` to the ledger.

        Only the canonical digest is stored — never the raw output — so the ledger
        proves an interaction occurred and was not altered without retaining its
        content. Does not require the crypto backend.
        """
        return self._ledger.append({"output": output, "metadata": dict(metadata or {})})

    def verify_ledger(self) -> bool:
        """Report whether the recorded ledger chain is intact."""
        return self._ledger.verify()
