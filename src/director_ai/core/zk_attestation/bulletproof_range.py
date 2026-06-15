# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Bulletproof range-proof attestation backend

"""Prove an aggregate meets a public threshold without revealing the aggregate.

The :class:`~director_ai.core.zk_attestation.SchnorrAttestationBackend` hides the
individual sample values but still reveals the aggregate. This backend hides the
aggregate too: it proves `Σ vᵢ ≥ threshold` while disclosing neither the
individual values nor their sum — only the public threshold and the pass/fail
decision.

It binds a Bulletproof range proof to the data. Each sample value is sealed in a
Ristretto Pedersen commitment; by homomorphism the published commitments sum to a
commitment of the aggregate, and the proof shows that this aggregate commitment,
shifted by the threshold, hides a non-negative value in ``[0, 2^bits)``. Because
the verifier recomputes the aggregate commitment from the published per-sample
commitments, the range proof cannot be forged against fabricated data — it is the
real committed aggregate that is shown to clear the bar.

This backend is **Rust-only**: the Bulletproof and Ristretto arithmetic live in
the ``backfire_kernel`` extension (dalek ``bulletproofs``). There is no pure
-Python fallback, because a correct, constant-time Bulletproof implementation in
Python would be neither safe nor practical; construction raises if the kernel is
absent. The threshold is public (the agreed compliance bar); the values and the
aggregate are hidden.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from .statements import AttestationStatement, HistorySample, MinimumCoherence

__all__ = [
    "RangeAttestation",
    "BulletproofRangeBackend",
    "min_aggregate_for",
]

_FIXED_POINT_SCALE = 1_000_000
_DEFAULT_BITS = 32


def _kernel():
    """Return the backfire_kernel module or raise a clear error."""
    try:
        import backfire_kernel
    except ImportError as exc:  # pragma: no cover - exercised on kernel-less installs
        raise RuntimeError(
            "BulletproofRangeBackend requires the compiled backfire_kernel "
            "extension (dalek bulletproofs); install/build the kernel to use it."
        ) from exc
    return backfire_kernel


def min_aggregate_for(
    statement: AttestationStatement, total_samples: int, scale: int
) -> int:
    """Minimum scaled aggregate that satisfies *statement* over *total_samples*.

    Encodes the statement's public threshold as the lower bound the range proof
    must clear. Currently supports :class:`MinimumCoherence` (mean ≥ threshold ⇔
    sum ≥ threshold·n); other statement shapes raise until their bound is added.
    """
    if isinstance(statement, MinimumCoherence):
        return math.ceil(statement.threshold * total_samples * scale)
    raise NotImplementedError(
        f"range-proof threshold not defined for statement kind {statement.kind!r}; "
        "use SchnorrAttestationBackend or add a bound to min_aggregate_for"
    )


@dataclass(frozen=True)
class RangeAttestation:
    """Published artefact: range proof + per-sample commitments (no aggregate)."""

    proof: bytes
    commitments: tuple[bytes, ...]
    threshold_scaled: int
    bits: int
    scale: int
    statement_kind: str
    samples_min: int


@dataclass
class BulletproofRangeBackend:
    """Zero-knowledge backend hiding both the sample values and the aggregate.

    Parameters
    ----------
    bits:
        Range-proof bit width (8, 16, 32, or 64); the scaled aggregate-minus
        -threshold difference must fit in ``[0, 2^bits)``. Defaults to 32.
    scale:
        Fixed-point scale applied to each ``evaluate_sample`` contribution.
    """

    bits: int = _DEFAULT_BITS
    scale: int = _FIXED_POINT_SCALE
    kind: str = field(default="bulletproof-range", init=False)

    def __post_init__(self) -> None:
        if self.bits not in (8, 16, 32, 64):
            raise ValueError("bits must be one of 8, 16, 32, 64")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        _kernel()  # fail fast if the extension is missing

    def prove(
        self,
        statement: AttestationStatement,
        samples: Sequence[HistorySample],
        *,
        context: bytes = b"",
    ) -> RangeAttestation:
        """Prove the statement holds, hiding every value and the aggregate.

        Raises ``ValueError`` (from the kernel) when the data does not meet the
        threshold — a false statement cannot be proven.
        """
        if not samples:
            raise ValueError("samples must be non-empty")
        values: list[int] = []
        for sample in samples:
            scaled = round(statement.evaluate_sample(sample) * self.scale)
            if scaled < 0:
                raise ValueError("statement contributions must be non-negative")
            values.append(scaled)
        threshold = min_aggregate_for(statement, len(samples), self.scale)
        samples_min = getattr(statement, "samples_min", 0)
        proof, commitments = _kernel().rust_bulletproof_prove_threshold(
            values, threshold, self.bits, context
        )
        return RangeAttestation(
            proof=bytes(proof),
            commitments=tuple(bytes(c) for c in commitments),
            threshold_scaled=threshold,
            bits=self.bits,
            scale=self.scale,
            statement_kind=statement.kind,
            samples_min=int(samples_min),
        )

    def verify(
        self,
        statement: AttestationStatement,
        proof: object,
        *,
        context: bytes = b"",
    ) -> tuple[bool, str]:
        """Verify the range proof and the public sample-count floor."""
        if not isinstance(proof, RangeAttestation):
            return False, "wrong_proof_type"
        if proof.statement_kind != statement.kind:
            return False, "statement_kind_mismatch"
        if proof.bits not in (8, 16, 32, 64):
            return False, "invalid_bits"
        if not proof.commitments:
            return False, "no_commitments"
        expected_threshold = min_aggregate_for(
            statement, len(proof.commitments), proof.scale
        )
        if proof.threshold_scaled != expected_threshold:
            return False, "threshold_mismatch"
        if len(proof.commitments) < proof.samples_min:
            return False, "too_few_samples"
        ok = _kernel().rust_bulletproof_verify_threshold(
            proof.proof,
            list(proof.commitments),
            proof.threshold_scaled,
            proof.bits,
            context,
        )
        if not ok:
            return False, "range_proof_invalid"
        return True, ""
