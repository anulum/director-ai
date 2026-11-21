# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Dataset and Encoding Tests

"""Split audits, distillation manifests, tokenisers, normalisers and embeddings."""

from __future__ import annotations

from dataclasses import replace

import pytest

from director_ai.model_lifecycle import (
    DatasetContractError,
    DatasetDeduplicationReport,
    DynamicsNormaliserEnvelope,
    EmbeddingSpaceIdentity,
    EncodingContractError,
    LicenceRecordError,
    MalformedDigestError,
    OutputTrainingNotPermittedError,
    SplitLeakageError,
    SyntheticDistillationManifest,
    TokeniserArtefactEnvelope,
    canonical_json_bytes,
    from_canonical_json_bytes,
    record_digest,
    validate_dedup_and_distillation,
    validate_dedup_report,
    validate_distillation_manifest,
    validate_embedding_space_identity,
    validate_normaliser_envelope,
    validate_tokeniser_and_normaliser_disjointness,
    validate_tokeniser_envelope,
)
from tests.model_lifecycle_support import digest_of, licence

TRAIN = (digest_of("train-1"), digest_of("train-2"))
SELECTION = (digest_of("selection-1"),)
SEALED = (digest_of("sealed-1"),)


def dedup(**changes: object) -> DatasetDeduplicationReport:
    """Return a valid, clean language split audit with ``changes`` applied."""
    base = DatasetDeduplicationReport(
        schema="dataset-dedup-report.v1",
        domain_kind="language_corpus",
        cluster_key_kind="document_cluster",
        partition_shard_digests=(
            ("train", TRAIN),
            ("selection", SELECTION),
            ("sealed", SEALED),
        ),
        partition_cluster_counts=(("train", 100), ("selection", 10), ("sealed", 10)),
        partition_model_visible_bytes=(
            ("train", 10**6),
            ("selection", 10**5),
            ("sealed", 0),
        ),
        cross_split_shared_cluster_count=0,
        cross_split_near_duplicate_window_count=0,
        synthetic_non_train_row_count=0,
        real_train_cluster_fraction=0.8,
        train_template_family_ids=("template-a", "template-b"),
        non_train_template_family_ids=("template-c",),
        scrubber_report_digest=digest_of("scrubber"),
        contains_human_biosignal_or_phi=False,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


DYNAMICS = {"domain_kind": "dynamics_trajectories", "cluster_key_kind": "episode_id"}


def manifest(
    report: DatasetDeduplicationReport, **changes: object
) -> SyntheticDistillationManifest:
    """Return a valid distillation manifest naming ``report``."""
    base = SyntheticDistillationManifest(
        "synthetic-distillation-manifest.v1",
        digest_of("teacher"),
        licence(),
        "grader-exact-match",
        record_digest(report),
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def tokeniser(
    report: DatasetDeduplicationReport, **changes: object
) -> TokeniserArtefactEnvelope:
    """Return a valid tokeniser envelope fitted on ``report``'s first training shard."""
    base = TokeniserArtefactEnvelope(
        schema="tokeniser-artefact-envelope.v1",
        algorithm="byte_level_bpe",
        vocab_size=32000,
        unicode_normalisation="NFC",
        nfkc_justification=None,
        byte_fallback_enabled=True,
        special_tokens=(("<pad>", 0), ("<s>", 1), ("</s>", 2), ("<|im_end|>", 3)),
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        stop_token_ids=(2, 3),
        chat_template_sha256=digest_of("chat"),
        vocabulary_sha256=digest_of("vocab"),
        merges_or_scores_sha256=digest_of("merges"),
        dedup_report_digest=record_digest(report),
        training_corpus_shard_digests=TRAIN[:1],
        scrubber_report_digest=digest_of("scrubber"),
        delimiter_injection_report_digest=digest_of("delimiters"),
        fertility_tokens_per_byte=(("en", 0.25), ("sk", 0.31)),
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def normaliser(
    report: DatasetDeduplicationReport, **changes: object
) -> DynamicsNormaliserEnvelope:
    """Return a valid dynamics normaliser fitted on ``report``'s training shards."""
    base = DynamicsNormaliserEnvelope(
        "dynamics-normaliser-envelope.v1",
        (("plasma_current", "A"), ("loop_voltage", "V"), ("density", "m-3")),
        0.001,
        "robust_iqr",
        digest_of("parameters"),
        record_digest(report),
        TRAIN,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def embedding(**changes: object) -> EmbeddingSpaceIdentity:
    """Return a valid embedding space identity with ``changes`` applied."""
    base = EmbeddingSpaceIdentity(
        "embedding-space-identity.v1",
        digest_of("model"),
        digest_of("tokeniser"),
        768,
        "mean_non_pad",
        True,
        None,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


class TestSplitAudit:
    """The report's shape and the training gate."""

    def test_a_clean_audit_passes_the_gate(self) -> None:
        """A clean report passes with and without synthetic rows."""
        report = dedup()
        assert validate_dedup_and_distillation(report) == report
        assert validate_dedup_and_distillation(report, manifest(report)) == report

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"cross_split_shared_cluster_count": 1}, "clusters are shared"),
            ({"cross_split_near_duplicate_window_count": 2}, "near-duplicate"),
            ({"synthetic_non_train_row_count": 1}, "synthetic rows"),
            ({"non_train_template_family_ids": ("template-b",)}, "template family"),
        ],
        ids=["shared-cluster", "near-duplicate", "M-B2-DEDUP-3", "M-B2-DEDUP-1"],
    )
    def test_leakage_is_recorded_but_refused_by_the_gate(
        self, changes: dict[str, object], fragment: str
    ) -> None:
        """A report may record leakage; the gate refuses it by name."""
        report = validate_dedup_report(dedup(**changes))
        with pytest.raises(SplitLeakageError, match=fragment):
            validate_dedup_and_distillation(report)

    def test_training_needs_real_clusters(self) -> None:
        """No real training clusters fails; synthetic rows need a quarter real."""
        with pytest.raises(DatasetContractError, match="real clusters"):
            validate_dedup_and_distillation(dedup(real_train_cluster_fraction=0.0))
        low = dedup(real_train_cluster_fraction=0.2)
        validate_dedup_and_distillation(low)
        with pytest.raises(DatasetContractError, match="a quarter"):
            validate_dedup_and_distillation(low, manifest(low))
        edge = dedup(real_train_cluster_fraction=0.25)
        validate_dedup_and_distillation(edge, manifest(edge))
        full = dedup(real_train_cluster_fraction=1.0)
        validate_dedup_and_distillation(full, manifest(full))

    def test_m_b2_dedup_2_the_teacher_licence_decides(self) -> None:
        """B2-10: output training is read from the teacher's licence record."""
        report = dedup()
        closed = manifest(
            report, teacher_licence_record=licence(output_training_permitted=False)
        )
        validate_distillation_manifest(closed)
        with pytest.raises(OutputTrainingNotPermittedError):
            validate_dedup_and_distillation(report, closed)

    def test_a_manifest_of_another_audit_is_refused(self) -> None:
        """The manifest must name this report."""
        report = dedup()
        with pytest.raises(DatasetContractError, match="another dedup report"):
            validate_dedup_and_distillation(
                report, manifest(dedup(scrubber_report_digest=digest_of("x")))
            )

    @pytest.mark.parametrize(
        "changes",
        [
            {"schema": "dedup.v2"},
            {"domain_kind": "images"},
            {"cluster_key_kind": "row"},
            {"partition_shard_digests": [("train", TRAIN)]},
            {"partition_shard_digests": (("train", TRAIN), ("selection",))},
            {"partition_shard_digests": (("train", TRAIN), ("test", SELECTION))},
            {"partition_shard_digests": (("train", TRAIN), ("train", SELECTION))},
            {"partition_shard_digests": (("selection", SELECTION), ("sealed", SEALED))},
            {
                "partition_shard_digests": (
                    ("train", ()),
                    ("selection", SELECTION),
                    ("sealed", SEALED),
                )
            },
            {
                "partition_shard_digests": (
                    ("train", TRAIN + TRAIN[:1]),
                    ("selection", SELECTION),
                    ("sealed", SEALED),
                )
            },
            {
                "partition_shard_digests": (
                    ("train", TRAIN),
                    ("selection", TRAIN[:1]),
                    ("sealed", SEALED),
                )
            },
            {"partition_cluster_counts": (("train", 100), ("selection", 10))},
            {
                "partition_cluster_counts": (
                    ("train", -1),
                    ("selection", 10),
                    ("sealed", 10),
                )
            },
            {
                "partition_model_visible_bytes": (
                    ("train", 1),
                    ("selection", 1),
                    ("calibration", 1),
                )
            },
            {"cross_split_shared_cluster_count": -1},
            {"synthetic_non_train_row_count": 0.0},
            {"real_train_cluster_fraction": 1.5},
            {"real_train_cluster_fraction": -0.1},
            {"real_train_cluster_fraction": 1},
            {"train_template_family_ids": ("a", "a")},
            {"non_train_template_family_ids": ("",)},
            {"contains_human_biosignal_or_phi": True},
        ],
    )
    def test_report_shape_refusals(self, changes: dict[str, object]) -> None:
        """Each malformed report field is refused."""
        with pytest.raises(DatasetContractError):
            validate_dedup_report(dedup(**changes))

    def test_each_partition_table_rule_is_its_own(self) -> None:
        """Without train, or with a repeated partition, the table itself is refused."""
        no_train = dedup(
            partition_shard_digests=(("selection", SELECTION), ("sealed", SEALED)),
            partition_cluster_counts=(("selection", 10), ("sealed", 10)),
            partition_model_visible_bytes=(("selection", 1), ("sealed", 1)),
        )
        with pytest.raises(DatasetContractError, match="needs a train partition"):
            validate_dedup_report(no_train)
        repeated = dedup(
            partition_cluster_counts=(
                ("train", 100),
                ("selection", 10),
                ("sealed", 10),
                ("train", 100),
            ),
        )
        with pytest.raises(DatasetContractError, match="repeats partition 'train'"):
            validate_dedup_report(repeated)

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"scrubber_report_digest": "x"}, MalformedDigestError),
            ({"partition_shard_digests": (("train", ("x",)),)}, MalformedDigestError),
        ],
    )
    def test_report_digest_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Malformed digests are refused by name."""
        with pytest.raises(error):
            validate_dedup_report(dedup(**changes))

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"schema": "manifest.v2"}, DatasetContractError),
            ({"teacher_artefact_digest": "x"}, MalformedDigestError),
            (
                {"teacher_licence_record": licence(spdx_expression="")},
                LicenceRecordError,
            ),
            ({"verifier_grader_id": ""}, DatasetContractError),
            ({"dedup_report_digest": "x"}, MalformedDigestError),
        ],
    )
    def test_manifest_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Each malformed manifest field is refused by name."""
        with pytest.raises(error):
            validate_distillation_manifest(manifest(dedup(), **changes))

    def test_records_of_other_types_are_refused(self) -> None:
        """Each validator checks its own record type."""
        with pytest.raises(DatasetContractError):
            validate_dedup_report(embedding())  # type: ignore[arg-type]
        with pytest.raises(DatasetContractError):
            validate_distillation_manifest(embedding())  # type: ignore[arg-type]


class TestTokeniser:
    """Special tokens, normalisation and train-only fitting."""

    def test_a_valid_tokeniser_fitted_on_training_shards(self) -> None:
        """A tokeniser fitted on a training shard of its audit passes."""
        report = dedup()
        item = tokeniser(report)
        assert validate_tokeniser_and_normaliser_disjointness(item, report) == item
        validate_tokeniser_envelope(
            tokeniser(report, bos_token_id=None, chat_template_sha256=None)
        )
        validate_tokeniser_envelope(
            tokeniser(
                report,
                unicode_normalisation="NFKC",
                nfkc_justification="folds full-width digits",
            )
        )

    @pytest.mark.parametrize("shard", [SELECTION[0], SEALED[0], digest_of("unknown")])
    def test_m_b2_enc_1_fitting_shards_must_be_training_shards(
        self, shard: str
    ) -> None:
        """M-B2-ENC-1: a selection, sealed or unknown shard in the fitting set."""
        report = dedup()
        item = tokeniser(report, training_corpus_shard_digests=(TRAIN[0], shard))
        with pytest.raises(SplitLeakageError, match="outside the training partition"):
            validate_tokeniser_and_normaliser_disjointness(item, report)

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"stop_token_ids": (3,)}, "end token must be a stop token"),
            ({"unicode_normalisation": "NFKC"}, "nfkc_justification"),
            (
                {"unicode_normalisation": "NFKC", "nfkc_justification": ""},
                "nfkc_justification",
            ),
            ({"nfkc_justification": "because"}, "only for NFKC"),
            ({"special_tokens": (("<pad>", 0), ("<s>", 2), ("</s>", 1))}, "ascending"),
            (
                {"special_tokens": (("<pad>", 0), ("<s>", 1), ("</s>", 2), ("<x>", 2))},
                "ascending",
            ),
            (
                {
                    "special_tokens": (
                        ("<pad>", 0),
                        ("<s>", 1),
                        ("</s>", 2),
                        ("<x>", 32000),
                    )
                },
                "below vocab_size",
            ),
            ({"special_tokens": (("<pad>", 0), ("<s>", 1), ("<s>", 2))}, "names"),
            ({"special_tokens": (("<pad>", 0), ("<s>", 1))}, "eos_token_id"),
            ({"pad_token_id": 9}, "pad_token_id"),
            ({"bos_token_id": 9}, "bos_token_id"),
            ({"stop_token_ids": ()}, "stop_token_ids"),
            ({"stop_token_ids": (3, 2)}, "ascending"),
            ({"stop_token_ids": (2, 2)}, "ascending"),
            ({"stop_token_ids": (2, 9)}, "special token"),
            ({"special_tokens": (("<pad>", -1),)}, "special token id"),
            ({"special_tokens": (("<pad>", 0, 1),)}, "pairs"),
            ({"vocab_size": 0}, "vocab_size"),
            ({"algorithm": "wordpiece"}, "algorithm"),
            ({"byte_fallback_enabled": 1}, "byte_fallback_enabled"),
            ({"training_corpus_shard_digests": ()}, "at least one shard"),
            ({"training_corpus_shard_digests": (TRAIN[0], TRAIN[0])}, "repeat"),
            ({"fertility_tokens_per_byte": (("en", 0.0),)}, "above zero"),
            ({"fertility_tokens_per_byte": (("en", 1),)}, "fertility"),
            ({"fertility_tokens_per_byte": (("en", 0.2), ("en", 0.3))}, "names"),
            ({"schema": "tokeniser.v2"}, "schema"),
        ],
        ids=lambda value: str(value)[:40] if isinstance(value, str) else None,
    )
    def test_tokeniser_refusals(
        self, changes: dict[str, object], fragment: str
    ) -> None:
        """M-B2-ENC-2, M-B2-ENC-3 and every other tokeniser rule, by name."""
        with pytest.raises(EncodingContractError, match=fragment):
            validate_tokeniser_envelope(tokeniser(dedup(), **changes))

    @pytest.mark.parametrize(
        "field",
        [
            "chat_template_sha256",
            "vocabulary_sha256",
            "merges_or_scores_sha256",
            "dedup_report_digest",
            "scrubber_report_digest",
            "delimiter_injection_report_digest",
        ],
    )
    def test_tokeniser_digests(self, field: str) -> None:
        """Every tokeniser digest, the delimiter-injection report's included, is 64-hex."""
        with pytest.raises(MalformedDigestError):
            validate_tokeniser_envelope(tokeniser(dedup(), **{field: "x"}))

    def test_the_audit_must_be_the_named_language_audit(self) -> None:
        """A tokeniser names its own language audit."""
        report = dedup()
        with pytest.raises(EncodingContractError, match="another dedup report"):
            validate_tokeniser_and_normaliser_disjointness(
                tokeniser(report, dedup_report_digest=digest_of("other")), report
            )
        dynamics = dedup(**DYNAMICS)
        with pytest.raises(EncodingContractError, match="language_corpus"):
            validate_tokeniser_and_normaliser_disjointness(
                tokeniser(dynamics), dynamics
            )


class TestNormaliserAndEmbedding:
    """Dynamics normalisers and embedding space identities."""

    def test_a_normaliser_fitted_on_training_shards(self) -> None:
        """A normaliser fitted on its dynamics audit's training shards passes."""
        report = dedup(**DYNAMICS)
        item = normaliser(report)
        assert validate_tokeniser_and_normaliser_disjointness(item, report) == item
        validate_normaliser_envelope(normaliser(report, dt_seconds=None))
        leaking = normaliser(report, training_trajectory_shard_digests=(SEALED[0],))
        with pytest.raises(SplitLeakageError):
            validate_tokeniser_and_normaliser_disjointness(leaking, report)
        language = dedup()
        with pytest.raises(EncodingContractError, match="dynamics_trajectories"):
            validate_tokeniser_and_normaliser_disjointness(
                normaliser(language), language
            )

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"channel_names_and_ucum_units": ()}, EncodingContractError),
            (
                {"channel_names_and_ucum_units": (("ip", "A"), ("ip", "kA"))},
                EncodingContractError,
            ),
            ({"channel_names_and_ucum_units": (("ip", ""),)}, EncodingContractError),
            ({"dt_seconds": 0.0}, EncodingContractError),
            ({"dt_seconds": 1}, EncodingContractError),
            ({"scaler_kind": "minmax"}, EncodingContractError),
            ({"schema": "normaliser.v2"}, EncodingContractError),
            ({"training_trajectory_shard_digests": ()}, EncodingContractError),
            ({"parameters_sha256": "x"}, MalformedDigestError),
            ({"dedup_report_digest": "x"}, MalformedDigestError),
        ],
    )
    def test_normaliser_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Each malformed normaliser field is refused by name."""
        with pytest.raises(error):
            validate_normaliser_envelope(normaliser(dedup(**DYNAMICS), **changes))

    def test_valid_embedding_identities(self) -> None:
        """Seeds in ``[0, 2**63)`` and no seed are both valid."""
        for seed in (None, 0, 2**63 - 1):
            validate_embedding_space_identity(embedding(projection_seed=seed))

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"projection_seed": 2**63}, EncodingContractError),
            ({"projection_seed": -1}, EncodingContractError),
            ({"embed_dim": 0}, EncodingContractError),
            ({"pooling_mode": "max"}, EncodingContractError),
            ({"l2_normalised": None}, EncodingContractError),
            ({"schema": "embedding.v2"}, EncodingContractError),
            ({"model_content_digest": "x"}, MalformedDigestError),
            ({"tokeniser_envelope_digest": "x"}, MalformedDigestError),
        ],
    )
    def test_embedding_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Each malformed embedding field is refused by name."""
        with pytest.raises(error):
            validate_embedding_space_identity(embedding(**changes))

    def test_records_of_other_types_are_refused(self) -> None:
        """Each validator checks its own record type."""
        report = dedup()
        for check in (
            validate_tokeniser_envelope,
            validate_normaliser_envelope,
            validate_embedding_space_identity,
        ):
            with pytest.raises(EncodingContractError):
                check(report)  # type: ignore[arg-type]
        with pytest.raises(EncodingContractError):
            validate_tokeniser_and_normaliser_disjointness(embedding(), report)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "record",
    [
        dedup(),
        manifest(dedup()),
        tokeniser(dedup()),
        normaliser(dedup(**DYNAMICS)),
        embedding(projection_seed=7),
    ],
    ids=["dedup", "manifest", "tokeniser", "normaliser", "embedding"],
)
def test_every_record_round_trips_through_the_codec(record: object) -> None:
    """The five records decode from their canonical bytes to equal records."""
    assert (
        from_canonical_json_bytes(type(record), canonical_json_bytes(record)) == record
    )
