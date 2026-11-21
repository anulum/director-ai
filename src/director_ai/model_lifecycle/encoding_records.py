# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Encoding Records

"""Records of how a model reads its inputs: tokenisers, normalisers, embeddings.

A tokeniser or a dynamics normaliser is fitted on training shards only, and
names the split audit those shards come from. An embedding space identity
binds a model's weights, its tokeniser and its pooling, so a vector index
built with one embedding space is never queried with another. The rules
are in :mod:`.encoding_contracts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TOKENISER_ALGORITHMS: frozenset[str] = frozenset(
    {"byte_level_bpe", "unigram_sentencepiece", "tiktoken_bpe"}
)
UNICODE_NORMALISATIONS: frozenset[str] = frozenset(
    {"NFC", "NFKC", "none_raw_utf8_bytes"}
)
SCALER_KINDS: frozenset[str] = frozenset(
    {"zscore_clipped", "robust_iqr", "dimensionless_physical_characteristic"}
)
POOLING_MODES: frozenset[str] = frozenset(
    {"cls", "mean_non_pad", "last_eos_token", "matryoshka_prefix"}
)


@dataclass(frozen=True, slots=True)
class TokeniserArtefactEnvelope:
    """Train-only tokeniser provenance, special-token invariants, and safety digest."""

    schema: Literal["tokeniser-artefact-envelope.v1"]
    algorithm: Literal["byte_level_bpe", "unigram_sentencepiece", "tiktoken_bpe"]
    vocab_size: int
    unicode_normalisation: Literal["NFC", "NFKC", "none_raw_utf8_bytes"]
    nfkc_justification: str | None
    byte_fallback_enabled: bool
    special_tokens: tuple[tuple[str, int], ...]
    bos_token_id: int | None
    eos_token_id: int
    pad_token_id: int
    stop_token_ids: tuple[int, ...]
    chat_template_sha256: str | None
    vocabulary_sha256: str
    merges_or_scores_sha256: str
    dedup_report_digest: str
    training_corpus_shard_digests: tuple[str, ...]
    scrubber_report_digest: str
    delimiter_injection_report_digest: str
    fertility_tokens_per_byte: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class DynamicsNormaliserEnvelope:
    """Train-only physical channel normaliser envelope with UCUM units."""

    schema: Literal["dynamics-normaliser-envelope.v1"]
    channel_names_and_ucum_units: tuple[tuple[str, str], ...]
    dt_seconds: float | None
    scaler_kind: Literal[
        "zscore_clipped",
        "robust_iqr",
        "dimensionless_physical_characteristic",
    ]
    parameters_sha256: str
    dedup_report_digest: str
    training_trajectory_shard_digests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingSpaceIdentity:
    """Vector-index compatibility identity binding weights, tokeniser, and pooling."""

    schema: Literal["embedding-space-identity.v1"]
    model_content_digest: str
    tokeniser_envelope_digest: str
    embed_dim: int
    pooling_mode: Literal[
        "cls",
        "mean_non_pad",
        "last_eos_token",
        "matryoshka_prefix",
    ]
    l2_normalised: bool
    projection_seed: int | None
