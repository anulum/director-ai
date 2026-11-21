# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Data Ports

"""Ports that prepare a study's data: curation, tokenisers and normalisers.

The kernel defines these interfaces and checks what they return; an adapter
outside the kernel implements them. A curator splits raw sources into
cluster-disjoint splits; a tokeniser trainer and a dynamics normaliser fit
only on the train split and verify their artefacts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from .dataset_records import DatasetDeduplicationReport
from .encoding_records import DynamicsNormaliserEnvelope, TokeniserArtefactEnvelope


@runtime_checkable
class DatasetCuratorPort(Protocol):
    """Deduplicates raw corpora/trajectories into cluster-disjoint splits."""

    def curate_splits(
        self,
        *,
        raw_sources: Sequence[Path],
        domain_kind: Literal[
            "language_corpus",
            "dynamics_trajectories",
            "spiking_stimuli",
            "quantum_hamiltonians",
        ],
        cluster_key_kind: Literal[
            "document_cluster",
            "episode_id",
            "tokamak_shot_or_discharge_id",
            "molecule_or_hamiltonian_family_id",
        ],
        scrubber_report_digest: str,
        output_dir: Path,
    ) -> DatasetDeduplicationReport:
        """Split raw sources into cluster-disjoint splits and report it.

        Parameters
        ----------
        raw_sources : Sequence[Path]
            The raw corpora or trajectories.
        domain_kind : str
            What the sources are.
        cluster_key_kind : str
            The key that keeps a cluster inside one split.
        scrubber_report_digest : str
            SHA-256 of the report that scrubbed the sources.
        output_dir : Path
            Where the splits are written.

        Returns
        -------
        DatasetDeduplicationReport
            The deduplication report of the splits.
        """
        ...


@runtime_checkable
class TokeniserTrainerPort(Protocol):
    """Fits train-only tokenisers and verifies untrusted delimiter isolation."""

    def train_tokeniser(
        self,
        *,
        dedup_report: DatasetDeduplicationReport,
        train_shard_paths: Mapping[str, Path],
        vocab_size: int,
        algorithm: Literal[
            "byte_level_bpe",
            "unigram_sentencepiece",
            "tiktoken_bpe",
        ],
        unicode_normalisation: Literal[
            "NFC",
            "NFKC",
            "none_raw_utf8_bytes",
        ] = "NFC",
        nfkc_justification: str | None = None,
        special_tokens: Sequence[str],
        output_dir: Path,
    ) -> TokeniserArtefactEnvelope:
        """Fit a tokeniser on the train shards only.

        Parameters
        ----------
        dedup_report : DatasetDeduplicationReport
            The report whose train partition the shards belong to.
        train_shard_paths : Mapping[str, Path]
            Train shard digest to path.
        vocab_size : int
            The vocabulary size.
        algorithm : str
            The tokeniser algorithm.
        unicode_normalisation : str
            The Unicode normalisation, NFC by default.
        nfkc_justification : str or None
            Why NFKC is used, required with it.
        special_tokens : Sequence[str]
            The special tokens.
        output_dir : Path
            Where the tokeniser is written.

        Returns
        -------
        TokeniserArtefactEnvelope
            The tokeniser's envelope.
        """
        ...

    def encode_untrusted_text(
        self,
        *,
        tokeniser_dir: Path,
        untrusted_text: str,
    ) -> tuple[int, ...]:
        """Encode untrusted text so that it can never produce a delimiter.

        Parameters
        ----------
        tokeniser_dir : Path
            The tokeniser.
        untrusted_text : str
            The text to encode.

        Returns
        -------
        tuple[int, ...]
            The token ids.
        """
        ...

    def verify_untrusted_delimiter_safety(
        self,
        *,
        tokeniser_dir: Path,
        output_report_path: Path,
    ) -> str:
        """Verify that untrusted text cannot inject a delimiter, and report it.

        Parameters
        ----------
        tokeniser_dir : Path
            The tokeniser.
        output_report_path : Path
            Where the report is written.

        Returns
        -------
        str
            SHA-256 of the delimiter-injection report.
        """
        ...


@runtime_checkable
class DynamicsNormaliserPort(Protocol):
    """Fits and verifies train-only physical channel normalisers."""

    def fit_normaliser(
        self,
        *,
        dedup_report: DatasetDeduplicationReport,
        train_trajectory_paths: Mapping[str, Path],
        channels: tuple[tuple[str, str], ...],
        dt_seconds: float | None,
        scaler_kind: Literal[
            "zscore_clipped",
            "robust_iqr",
            "dimensionless_physical_characteristic",
        ],
        output_dir: Path,
    ) -> DynamicsNormaliserEnvelope:
        """Fit a channel normaliser on the train trajectories only.

        Parameters
        ----------
        dedup_report : DatasetDeduplicationReport
            The report whose train partition the trajectories belong to.
        train_trajectory_paths : Mapping[str, Path]
            Train trajectory digest to path.
        channels : tuple[tuple[str, str], ...]
            ``(channel, unit)`` pairs.
        dt_seconds : float or None
            The sampling step, if fixed.
        scaler_kind : str
            The scaler.
        output_dir : Path
            Where the normaliser is written.

        Returns
        -------
        DynamicsNormaliserEnvelope
            The normaliser's envelope.
        """
        ...

    def verify_normaliser(
        self,
        *,
        envelope: DynamicsNormaliserEnvelope,
        path: Path,
    ) -> bool:
        """Return whether the normaliser at ``path`` matches its envelope.

        Parameters
        ----------
        envelope : DynamicsNormaliserEnvelope
            The normaliser's envelope.
        path : Path
            The normaliser.

        Returns
        -------
        bool
            Whether it matches.
        """
        ...
