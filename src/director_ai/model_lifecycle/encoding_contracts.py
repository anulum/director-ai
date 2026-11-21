# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Encoding Validators

"""Validators of tokeniser, normaliser and embedding space records.

A tokeniser's special tokens have distinct ids in ascending order below its
vocabulary size; its end, padding and any start token are special; its stop
tokens are special, ascending and include the end token. NFKC folding needs
a written reason. :func:`validate_tokeniser_and_normaliser_disjointness`
then requires that a tokeniser or normaliser was fitted on training shards
of its own split audit only.
"""

from __future__ import annotations

from ._checks import (
    require_bool,
    require_choice,
    require_finite,
    require_int,
    require_sha256,
    require_text,
    require_tuple,
    require_unique,
)
from .architecture_specs import MAX_SEED
from .dataset_contracts import validate_dedup_report
from .dataset_records import DatasetDeduplicationReport
from .digests import record_digest
from .encoding_records import (
    POOLING_MODES,
    SCALER_KINDS,
    TOKENISER_ALGORITHMS,
    UNICODE_NORMALISATIONS,
    DynamicsNormaliserEnvelope,
    EmbeddingSpaceIdentity,
    TokeniserArtefactEnvelope,
)
from .errors import EncodingContractError, SplitLeakageError

E = EncodingContractError


def _shards(value: object, field: str) -> tuple[object, ...]:
    """Return a non-empty tuple of distinct shard digests."""
    items = require_tuple(value, field, E)
    if not items:
        raise E(f"{field} needs at least one shard")
    for item in items:
        require_sha256(item, field)
    require_unique(items, field, E)
    return items


def _named_pairs(value: object, field: str) -> list[tuple[str, object]]:
    """Return ``(name, value)`` pairs with distinct, non-empty names."""
    pairs: list[tuple[str, object]] = []
    for pair in require_tuple(value, field, E):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise E(f"{field} must hold (name, value) pairs")
        pairs.append((require_text(pair[0], f"{field} name", E), pair[1]))
    require_unique([name for name, _ in pairs], f"{field} names", E)
    return pairs


def _check_special_tokens(envelope: TokeniserArtefactEnvelope) -> None:
    """Check special-token ids, the end, padding and start tokens, and stop tokens."""
    vocab = require_int(envelope.vocab_size, "vocab_size", E, minimum=1)
    ids = [
        require_int(token_id, "special token id", E, minimum=0)
        for _, token_id in _named_pairs(envelope.special_tokens, "special_tokens")
    ]
    if any(a >= b for a, b in zip(ids, ids[1:], strict=False)):
        raise E("special token ids must be distinct and ascending")
    if ids and ids[-1] >= vocab:
        raise E("special token ids must lie below vocab_size")
    special = set(ids)
    named = {
        "eos_token_id": envelope.eos_token_id,
        "pad_token_id": envelope.pad_token_id,
    }
    if envelope.bos_token_id is not None:
        named["bos_token_id"] = envelope.bos_token_id
    for name, token_id in named.items():
        if require_int(token_id, name, E, minimum=0) not in special:
            raise E(f"{name} must be a special token")
    stops = [
        require_int(token_id, "stop token id", E, minimum=0)
        for token_id in require_tuple(envelope.stop_token_ids, "stop_token_ids", E)
    ]
    if not stops or any(a >= b for a, b in zip(stops, stops[1:], strict=False)):
        raise E("stop_token_ids must be non-empty, distinct and ascending")
    if not set(stops) <= special:
        raise E("every stop token must be a special token")
    if envelope.eos_token_id not in stops:
        raise E("the end token must be a stop token")


def validate_tokeniser_envelope(
    envelope: TokeniserArtefactEnvelope,
) -> TokeniserArtefactEnvelope:
    """Check a tokeniser envelope's special tokens, normalisation and digests.

    Parameters
    ----------
    envelope : TokeniserArtefactEnvelope
        The envelope to check.

    Returns
    -------
    TokeniserArtefactEnvelope
        The same envelope.

    Raises
    ------
    EncodingContractError
        If a special-token rule fails, NFKC has no reason or another
        normalisation has one, or a field is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(envelope, TokeniserArtefactEnvelope):
        raise E("a TokeniserArtefactEnvelope is required")
    require_choice(envelope.schema, "schema", {"tokeniser-artefact-envelope.v1"}, E)
    require_choice(envelope.algorithm, "algorithm", TOKENISER_ALGORITHMS, E)
    normalisation = require_choice(
        envelope.unicode_normalisation,
        "unicode_normalisation",
        UNICODE_NORMALISATIONS,
        E,
    )
    if normalisation == "NFKC":
        require_text(envelope.nfkc_justification, "nfkc_justification", E)
    elif envelope.nfkc_justification is not None:
        raise E("nfkc_justification is only for NFKC")
    require_bool(envelope.byte_fallback_enabled, "byte_fallback_enabled", E)
    _check_special_tokens(envelope)
    if envelope.chat_template_sha256 is not None:
        require_sha256(envelope.chat_template_sha256, "chat_template_sha256")
    for name in (
        "vocabulary_sha256",
        "merges_or_scores_sha256",
        "dedup_report_digest",
        "scrubber_report_digest",
        "delimiter_injection_report_digest",
    ):
        require_sha256(getattr(envelope, name), name)
    _shards(envelope.training_corpus_shard_digests, "training_corpus_shard_digests")
    for _, fertility in _named_pairs(
        envelope.fertility_tokens_per_byte, "fertility_tokens_per_byte"
    ):
        if require_finite(fertility, "fertility_tokens_per_byte", E) <= 0.0:
            raise E("fertility_tokens_per_byte values must be above zero")
    return envelope


def validate_normaliser_envelope(
    envelope: DynamicsNormaliserEnvelope,
) -> DynamicsNormaliserEnvelope:
    """Check a dynamics normaliser envelope's channels, step and digests.

    Parameters
    ----------
    envelope : DynamicsNormaliserEnvelope
        The envelope to check.

    Returns
    -------
    DynamicsNormaliserEnvelope
        The same envelope.

    Raises
    ------
    EncodingContractError
        If there is no channel, a channel name repeats, a unit is empty, the
        step is not above zero, or a field is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(envelope, DynamicsNormaliserEnvelope):
        raise E("a DynamicsNormaliserEnvelope is required")
    require_choice(envelope.schema, "schema", {"dynamics-normaliser-envelope.v1"}, E)
    channels = _named_pairs(
        envelope.channel_names_and_ucum_units, "channel_names_and_ucum_units"
    )
    if not channels:
        raise E("a normaliser needs at least one channel")
    for _, unit in channels:
        require_text(unit, "channel unit", E)
    if (
        envelope.dt_seconds is not None
        and require_finite(envelope.dt_seconds, "dt_seconds", E) <= 0.0
    ):
        raise E("dt_seconds must be above zero")
    require_choice(envelope.scaler_kind, "scaler_kind", SCALER_KINDS, E)
    require_sha256(envelope.parameters_sha256, "parameters_sha256")
    require_sha256(envelope.dedup_report_digest, "dedup_report_digest")
    _shards(
        envelope.training_trajectory_shard_digests, "training_trajectory_shard_digests"
    )
    return envelope


def validate_embedding_space_identity(
    identity: EmbeddingSpaceIdentity,
) -> EmbeddingSpaceIdentity:
    """Check an embedding space identity.

    Parameters
    ----------
    identity : EmbeddingSpaceIdentity
        The identity to check.

    Returns
    -------
    EmbeddingSpaceIdentity
        The same identity.

    Raises
    ------
    EncodingContractError
        If a field is malformed or the projection seed is outside ``[0, 2**63)``.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(identity, EmbeddingSpaceIdentity):
        raise E("an EmbeddingSpaceIdentity is required")
    require_choice(identity.schema, "schema", {"embedding-space-identity.v1"}, E)
    require_sha256(identity.model_content_digest, "model_content_digest")
    require_sha256(identity.tokeniser_envelope_digest, "tokeniser_envelope_digest")
    require_int(identity.embed_dim, "embed_dim", E, minimum=1)
    require_choice(identity.pooling_mode, "pooling_mode", POOLING_MODES, E)
    require_bool(identity.l2_normalised, "l2_normalised", E)
    if identity.projection_seed is not None and (
        require_int(identity.projection_seed, "projection_seed", E, minimum=0)
        >= MAX_SEED
    ):
        raise E("projection_seed must be below 2**63")
    return identity


def validate_tokeniser_and_normaliser_disjointness(
    envelope: TokeniserArtefactEnvelope | DynamicsNormaliserEnvelope,
    dedup: DatasetDeduplicationReport,
) -> TokeniserArtefactEnvelope | DynamicsNormaliserEnvelope:
    """Require that a tokeniser or normaliser was fitted on training shards only.

    Parameters
    ----------
    envelope : TokeniserArtefactEnvelope or DynamicsNormaliserEnvelope
        The fitted tokeniser or normaliser.
    dedup : DatasetDeduplicationReport
        The split audit it names.

    Returns
    -------
    TokeniserArtefactEnvelope or DynamicsNormaliserEnvelope
        The same envelope.

    Raises
    ------
    SplitLeakageError
        If a fitting shard is not a training shard of the audit.
    EncodingContractError
        If the envelope is malformed, names another audit, or its audit is
        of another domain.
    DatasetContractError
        If the audit is malformed.
    """
    validate_dedup_report(dedup)
    if isinstance(envelope, TokeniserArtefactEnvelope):
        validate_tokeniser_envelope(envelope)
        domain, shards = "language_corpus", envelope.training_corpus_shard_digests
    else:
        validate_normaliser_envelope(envelope)
        domain, shards = (
            "dynamics_trajectories",
            envelope.training_trajectory_shard_digests,
        )
    if dedup.domain_kind != domain:
        raise E(f"the envelope needs a {domain} audit")
    if envelope.dedup_report_digest != record_digest(dedup):
        raise E("the envelope names another dedup report")
    train = dict(dedup.partition_shard_digests)["train"]
    outside = sorted(set(shards) - set(train))
    if outside:
        raise SplitLeakageError(
            f"fitting shards outside the training partition: {outside}"
        )
    return envelope
