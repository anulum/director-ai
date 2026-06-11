# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Ed25519 output signing

"""Detached Ed25519 signatures over LLM outputs for non-repudiation.

A symmetric HMAC proves integrity to anyone holding the shared key; an Ed25519
signature proves *non-repudiation* — a third party can verify, with only the
public key, that this exact output and metadata were produced by the holder of
the private key and not altered since. That is what finance and healthcare audit
trails require. :class:`OutputSigner` signs; :func:`verify_signed_output` checks a
:class:`SignedOutput` against its embedded public key.

Ed25519 needs the optional ``cryptography`` backend (``pip install
director-ai[crypto]``); :class:`MissingCryptoBackendError` is raised with that
hint when it is absent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MissingCryptoBackendError",
    "OutputSigner",
    "SignedOutput",
    "verify_signed_output",
]

_ALGORITHM = "ed25519"


class MissingCryptoBackendError(RuntimeError):
    """Raised when Ed25519 signing is requested without the ``cryptography`` backend."""


def _load_ed25519():
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:
        raise MissingCryptoBackendError(
            "Ed25519 output signing requires the cryptography backend: "
            "pip install director-ai[crypto]"
        ) from exc
    return ed25519


def _signing_payload(output: str, metadata: Mapping[str, Any]) -> bytes:
    # Canonical, order-independent encoding so verification is deterministic.
    canonical = json.dumps(
        {"output": output, "metadata": dict(metadata)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return canonical.encode("utf-8")


@dataclass(frozen=True)
class SignedOutput:
    """An output bound to a detached Ed25519 signature and its public key."""

    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    signature: str = ""
    public_key: str = ""
    algorithm: str = _ALGORITHM

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (the full signed envelope)."""
        return {
            "output": self.output,
            "metadata": dict(self.metadata),
            "signature": self.signature,
            "public_key": self.public_key,
            "algorithm": self.algorithm,
        }


class OutputSigner:
    """Sign LLM outputs with a held Ed25519 key.

    With no seed a fresh key is generated; supply a 32-byte ``seed`` to reproduce
    a stable identity across processes (load it from a secret manager, never hard
    code it).
    """

    def __init__(self, seed: bytes | None = None):
        if seed is not None and len(seed) != 32:
            raise ValueError("Ed25519 seed must be exactly 32 bytes")
        self._seed = seed
        self._private_key: Any = None

    def _key(self) -> Any:
        if self._private_key is None:
            ed25519 = _load_ed25519()
            if self._seed is None:
                self._private_key = ed25519.Ed25519PrivateKey.generate()
            else:
                self._private_key = ed25519.Ed25519PrivateKey.from_private_bytes(
                    self._seed
                )
        return self._private_key

    @property
    def public_key_hex(self) -> str:
        """The hex-encoded raw public key for distribution to verifiers."""
        from cryptography.hazmat.primitives import serialization

        raw = (
            self._key()
            .public_key()
            .public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        )
        return str(raw.hex())

    def sign(
        self, output: str, metadata: Mapping[str, Any] | None = None
    ) -> SignedOutput:
        """Produce a :class:`SignedOutput` over ``output`` and ``metadata``."""
        if not isinstance(output, str):
            raise TypeError("output to sign must be a string")
        meta = dict(metadata or {})
        signature = self._key().sign(_signing_payload(output, meta))
        return SignedOutput(
            output=output,
            metadata=meta,
            signature=signature.hex(),
            public_key=self.public_key_hex,
            algorithm=_ALGORITHM,
        )


def verify_signed_output(signed: SignedOutput) -> bool:
    """Verify a :class:`SignedOutput` against its embedded public key."""
    if signed.algorithm != _ALGORITHM:
        return False
    ed25519 = _load_ed25519()
    try:
        from cryptography.exceptions import InvalidSignature

        public_key = ed25519.Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(signed.public_key)
        )
        public_key.verify(
            bytes.fromhex(signed.signature),
            _signing_payload(signed.output, signed.metadata),
        )
    except (InvalidSignature, ValueError):
        return False
    return True
