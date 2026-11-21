# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Post-Swap Verification Records

"""What a chained deployment's post-swap check observed, and its evidence.

After a deployment publishes or seals generation ``base_generation + 1``, its
verifier reads the chain and the serving start record and reports an effect
(whether the swap is visible), a result (whether it is verified) and one
observation from a fixed set, with the evidence it read. These records are
the typed return of ``ChainedDeploymentPort.verify_post_swap``.

Each observation fixes exactly one effect and one result, as the chained
deployment contract's table gives them; ``OBSERVATION_OUTCOMES`` mirrors the
mapping Core implements for the same verifier. A verified swap names the
slot record's and the start record's digests and the observed start
process, and the target is always the generation after the base.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._checks import require_choice, require_int, require_sha256, require_text
from .errors import DeploymentRecordError

E = DeploymentRecordError
PostSwapEffect = Literal["unknown", "absent", "observed"]
PostSwapResult = Literal["verified", "not_verified", "not_checked"]
PostSwapObservation = Literal[
    "chain_unreadable",
    "base_differs",
    "not_published",
    "sealed_before_publication",
    "slot_taken_by_another_attempt",
    "artefact_differs",
    "published_not_converged",
    "start_record_not_recognised",
    "published_and_converged",
    "converged_artefact_differs",
    "superseded_before_convergence_seen",
]
POST_SWAP_EFFECTS: frozenset[str] = frozenset({"unknown", "absent", "observed"})
POST_SWAP_RESULTS: frozenset[str] = frozenset(
    {"verified", "not_verified", "not_checked"}
)
OBSERVATION_OUTCOMES: dict[str, tuple[str, str]] = {
    "chain_unreadable": ("unknown", "not_checked"),
    "base_differs": ("observed", "not_verified"),
    "not_published": ("absent", "not_verified"),
    "sealed_before_publication": ("absent", "not_verified"),
    "slot_taken_by_another_attempt": ("absent", "not_verified"),
    "artefact_differs": ("observed", "not_verified"),
    "published_not_converged": ("observed", "not_checked"),
    "start_record_not_recognised": ("observed", "not_checked"),
    "published_and_converged": ("observed", "verified"),
    "converged_artefact_differs": ("observed", "not_verified"),
    "superseded_before_convergence_seen": ("observed", "not_checked"),
}
"""Each observation's one ``(effect, result)``, mirroring Core's mapping."""
POST_SWAP_OBSERVATIONS: frozenset[str] = frozenset(OBSERVATION_OUTCOMES)


@dataclass(frozen=True, slots=True)
class PostSwapEvidence:
    """Immutable evidence bundle recorded during chained post-swap verification."""

    chain_dir: str
    start_record_path: str
    requested_revision: str
    base_generation: int
    target_generation: int
    chain_head_generation: int | None
    slot_record_sha256: str | None
    start_record_sha256: str | None
    start_record_mtime_ns: str | None
    observed_start_pid: int | None


@dataclass(frozen=True, slots=True)
class PostSwapVerification:
    """Typed verification record returned by ChainedDeploymentPort.verify_post_swap."""

    schema: Literal["post-swap-verification.v1"]
    operation_id: str
    attempt_id: str
    effect: PostSwapEffect
    result: PostSwapResult
    observation: PostSwapObservation
    evidence: PostSwapEvidence


def validate_post_swap_evidence(evidence: PostSwapEvidence) -> PostSwapEvidence:
    """Check a post-swap evidence bundle's shape.

    Parameters
    ----------
    evidence : PostSwapEvidence
        The evidence to check.

    Returns
    -------
    PostSwapEvidence
        The same evidence.

    Raises
    ------
    DeploymentRecordError
        If the target is not the generation after the base, a count or
        identifier is malformed, or the start record's time is not digits.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(evidence, PostSwapEvidence):
        raise E("a PostSwapEvidence is required")
    for name in ("chain_dir", "start_record_path", "requested_revision"):
        require_text(getattr(evidence, name), name, E)
    base = require_int(evidence.base_generation, "base_generation", E, minimum=0)
    if (
        require_int(evidence.target_generation, "target_generation", E, minimum=1)
        != base + 1
    ):
        raise E("the target generation is the one after the base generation")
    if evidence.chain_head_generation is not None:
        require_int(
            evidence.chain_head_generation, "chain_head_generation", E, minimum=0
        )
    for name in ("slot_record_sha256", "start_record_sha256"):
        if getattr(evidence, name) is not None:
            require_sha256(getattr(evidence, name), name)
    mtime = evidence.start_record_mtime_ns
    if mtime is not None and not (
        isinstance(mtime, str) and mtime.isascii() and mtime.isdigit()
    ):
        raise E("start_record_mtime_ns is decimal digits")
    if evidence.observed_start_pid is not None:
        require_int(evidence.observed_start_pid, "observed_start_pid", E, minimum=1)
    return evidence


def validate_post_swap_verification(
    verification: PostSwapVerification,
) -> PostSwapVerification:
    """Check a post-swap verification record's shape.

    Parameters
    ----------
    verification : PostSwapVerification
        The record to check.

    Returns
    -------
    PostSwapVerification
        The same record.

    Raises
    ------
    DeploymentRecordError
        If a literal is unknown, the effect and result are not the ones the
        observation fixes, a verified swap lacks its slot or start record
        digest or start process, or a field is malformed.
    MalformedDigestError
        If an evidence digest is malformed.
    """
    if not isinstance(verification, PostSwapVerification):
        raise E("a PostSwapVerification is required")
    require_choice(verification.schema, "schema", {"post-swap-verification.v1"}, E)
    require_text(verification.operation_id, "operation_id", E)
    require_text(verification.attempt_id, "attempt_id", E)
    effect = require_choice(verification.effect, "effect", POST_SWAP_EFFECTS, E)
    result = require_choice(verification.result, "result", POST_SWAP_RESULTS, E)
    observation = require_choice(
        verification.observation, "observation", POST_SWAP_OBSERVATIONS, E
    )
    if (effect, result) != OBSERVATION_OUTCOMES[observation]:
        raise E(
            f"{observation} fixes the effect and result {OBSERVATION_OUTCOMES[observation]}"
        )
    evidence = validate_post_swap_evidence(verification.evidence)
    if result == "verified" and None in (
        evidence.slot_record_sha256,
        evidence.start_record_sha256,
        evidence.observed_start_pid,
    ):
        raise E("a verified swap names its slot and start record digests and start pid")
    return verification
