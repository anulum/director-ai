# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — cryptographic output integrity

"""Cryptographic integrity and non-repudiation for model outputs (OWASP ML09).

Two composable controls:

* :class:`TamperEvidentLedger` — an append-only, hash-chained log of output
  digests; altering any past entry breaks the chain. Stdlib only, always
  available, tenant-safe (digests, never raw payloads).
* :class:`OutputSigner` / :func:`verify_signed_output` — detached Ed25519
  signatures so a third party can verify, with only the public key, that an
  output and its metadata are authentic and unaltered. Needs the optional
  ``cryptography`` backend (``pip install director-ai[crypto]``).

:class:`OutputIntegrityGuard` composes both.
"""

from .guard import OutputIntegrityGuard
from .ledger import GENESIS_HASH, LedgerEntry, TamperEvidentLedger
from .signing import (
    MissingCryptoBackendError,
    OutputSigner,
    SignedOutput,
    verify_signed_output,
)

__all__ = [
    "GENESIS_HASH",
    "LedgerEntry",
    "MissingCryptoBackendError",
    "OutputIntegrityGuard",
    "OutputSigner",
    "SignedOutput",
    "TamperEvidentLedger",
    "verify_signed_output",
]
