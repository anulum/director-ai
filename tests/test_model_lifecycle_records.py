# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — model lifecycle record tests

"""Canonical JSON, licences, envelopes, custody records and preregistrations."""

from __future__ import annotations

import typing
from dataclasses import dataclass, replace

import pytest

from director_ai.model_lifecycle import (
    SEALED_CUSTODY_KINDS,
    CustodyIngestRecord,
    CustodyRecordError,
    EnvelopeError,
    EvaluationContractError,
    LicenceRecordError,
    MalformedDigestError,
    ModelArtefactEnvelope,
    ModelLifecycleError,
    NonCanonicalJsonError,
    OutputTrainingNotPermittedError,
    PreregistrationApproval,
    PreregistrationRecord,
    SampleSizeError,
    canonical_json_bytes,
    decode_record,
    is_sealed_kind,
    record_digest,
    require_active_family,
    require_output_training_permitted,
    required_paired_sample_size,
    strict_json_loads,
    validate_base_checkpoint_record,
    validate_compute_budget,
    validate_custody_ingest_record,
    validate_envelope,
    validate_grader,
    validate_licence_record,
    validate_lineage_edge,
    validate_preregistration,
    validate_preregistration_approval,
    validate_sealed_split_pool,
)
from tests.model_lifecycle_support import (
    PROPOSER_A,
    UTC,
    base_checkpoint,
    budget,
    custody_record,
    digest_of,
    edge,
    envelope,
    grader,
    licence,
    pool,
    preregistration,
)


@dataclass(frozen=True, slots=True)
class _Probe:
    """A record exercising every annotation form the decoder supports."""

    name: str
    count: int
    ratio: float
    flag: bool
    note: str | None
    pair: tuple[str, int]
    items: tuple[int, ...]
    kind: typing.Literal["a", "b"]


class TestCanonicalJson:
    """The canonical encoding, record digests and strict decoding."""

    def test_encoding_sorts_keys_and_turns_tuples_into_arrays(self) -> None:
        """Dataclasses become sorted objects with compact separators."""
        probe = _Probe("é", 2, 0.5, True, None, ("x", 1), (1, 2), "a")
        assert (
            canonical_json_bytes(probe)
            == (
                '{"count":2,"flag":true,"items":[1,2],"kind":"a","name":"é",'
                '"note":null,"pair":["x",1],"ratio":0.5}'
            ).encode()
        )
        assert canonical_json_bytes({"b": [1], "a": {}}) == b'{"a":{},"b":[1]}'
        assert len(record_digest(probe)) == 64

    @pytest.mark.parametrize(
        "value", [float("nan"), {1: "x"}, object(), [float("inf")]]
    )
    def test_values_without_a_canonical_form_are_refused(self, value: object) -> None:
        """Non-finite floats, non-string keys and other types are refused."""
        with pytest.raises(NonCanonicalJsonError):
            canonical_json_bytes(value)

    def test_decoding_round_trips_a_record(self) -> None:
        """A canonical encoding decodes to an equal record."""
        probe = _Probe("n", 3, 1.25, False, "x", ("y", 2), (), "b")
        seen: list[_Probe] = []
        assert decode_record(_Probe, canonical_json_bytes(probe), seen.append) == probe
        assert seen == [probe]

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            (b"\xff", "not valid UTF-8"),
            (b"{", "malformed at offset"),
            (b'{"a":1,"a":2}', "appears twice"),
            (b'{"a":NaN}', "constant NaN"),
            (b"[]", "must be a JSON object"),
            (b'{"name":"n"}', "keys differ"),
        ],
    )
    def test_strict_decoding_refusals(self, raw: bytes, message: str) -> None:
        """Encoding, syntax, duplicates, constants and keys are strict."""
        with pytest.raises(NonCanonicalJsonError, match=message):
            decode_record(_Probe, raw, lambda _: None)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("count", True, "must be int"),
            ("count", 1.0, "must be int"),
            ("ratio", 1, "must be float"),
            ("flag", 1, "must be bool"),
            ("note", 3, "must be str"),
            ("pair", "x", "must be an array"),
            ("pair", ["x"], "must have 2 items"),
            ("items", [1, "2"], "must be int"),
            ("kind", "c", "must be one of"),
        ],
    )
    def test_type_coercion_is_refused(
        self, field: str, value: object, message: str
    ) -> None:
        """Booleans, integers and floats are never coerced into each other."""
        payload: dict[str, object] = {
            "name": "n",
            "count": 1,
            "ratio": 0.5,
            "flag": True,
            "note": None,
            "pair": ["x", 1],
            "items": [],
            "kind": "a",
        }
        payload[field] = value
        with pytest.raises(NonCanonicalJsonError, match=message):
            decode_record(_Probe, canonical_json_bytes(payload), lambda _: None)

    def test_unsupported_annotations_are_refused(self) -> None:
        """Only the annotation forms the records use can be decoded."""

        @dataclass(frozen=True)
        class Wide:
            """A record with a multi-member union."""

            value: int | str | None

        @dataclass(frozen=True)
        class Mapped:
            """A record with a mapping field."""

            value: dict[str, int]

        with pytest.raises(NonCanonicalJsonError, match="unsupported union"):
            decode_record(Wide, b'{"value":1}', lambda _: None)
        with pytest.raises(NonCanonicalJsonError, match="unsupported annotation"):
            decode_record(Mapped, b'{"value":{}}', lambda _: None)

    def test_the_validator_runs_after_decoding(self) -> None:
        """A decoded record that its validator refuses is not returned."""
        raw = canonical_json_bytes(licence(data_class="not_applicable"))
        with pytest.raises(LicenceRecordError):
            decode_record(type(licence()), raw, validate_licence_record)

    def test_lone_surrogates_are_refused_with_a_typed_error(self) -> None:
        """A string that cannot encode as UTF-8 is refused on both sides."""
        with pytest.raises(NonCanonicalJsonError, match="not valid Unicode"):
            strict_json_loads(b'{"a": "\\ud800"}')
        with pytest.raises(NonCanonicalJsonError, match="encode as UTF-8"):
            canonical_json_bytes({"a": "\ud800"})

    def test_strict_loads_returns_plain_data(self) -> None:
        """Plain JSON decodes unchanged."""
        assert strict_json_loads(b'{"a":[1,2.5,null]}') == {"a": [1, 2.5, None]}


class TestLicencesAndBudgets:
    """Dual-axis licence records, budgets and the output-training check."""

    def test_valid_records_pass(self) -> None:
        """The builders' records are valid."""
        assert validate_licence_record(licence()) == licence()
        assert validate_compute_budget(budget()) == budget()
        require_output_training_permitted(licence())

    def test_output_training_needs_explicit_permission(self) -> None:
        """M-J4-4: a teacher without the permission cannot train another model."""
        with pytest.raises(OutputTrainingNotPermittedError):
            require_output_training_permitted(licence(output_training_permitted=False))

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"schema": "dual-axis-licence-record.v2"}, LicenceRecordError),
            ({"spdx_expression": " Apache-2.0"}, LicenceRecordError),
            ({"weight_and_output_terms_digest": ""}, MalformedDigestError),
            ({"data_class": "not_applicable"}, LicenceRecordError),
            ({"tdm_opted_out": 0}, LicenceRecordError),
            ({"output_training_permitted": 1}, LicenceRecordError),
        ],
    )
    def test_licence_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Each malformed licence field is refused."""
        with pytest.raises(error):
            validate_licence_record(licence(**changes))

    @pytest.mark.parametrize(
        "changes",
        [
            {"schema": "budget"},
            {"max_wall_seconds": -1},
            {"max_sealed_looks": True},
            {"max_accelerator_hours": float("inf")},
            {"max_accelerator_hours": 2},
            {"max_accelerator_hours": -0.5},
        ],
    )
    def test_budget_refusals(self, changes: dict[str, object]) -> None:
        """Negative, non-finite or mistyped axes are refused."""
        with pytest.raises(LicenceRecordError):
            validate_compute_budget(budget(**changes))

    def test_wrong_record_types_are_refused(self) -> None:
        """Validators refuse an object of another type."""
        with pytest.raises(LicenceRecordError):
            validate_licence_record(budget())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it
        with pytest.raises(LicenceRecordError):
            validate_compute_budget(licence())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it


def _no_optional_or_untyped(annotation: object) -> bool:
    """Return whether an annotation tree holds no ``Any``, ``dict`` or ``None``."""
    if annotation is typing.Any or annotation is type(None) or annotation is dict:
        return False
    if typing.get_origin(annotation) is dict:
        return False
    return all(
        _no_optional_or_untyped(arg)
        for arg in typing.get_args(annotation)
        if arg is not Ellipsis
    )


class TestEnvelopes:
    """K-1: exact envelopes, real reservations and the registered families."""

    def test_envelope_annotations_have_no_any_dict_or_none(self) -> None:
        """K-1: no field is ``Any``, an untyped mapping or optional."""
        hints = typing.get_type_hints(ModelArtefactEnvelope)
        assert hints and all(_no_optional_or_untyped(hint) for hint in hints.values())

    @pytest.mark.parametrize(
        "family", ["language", "dynamics", "spiking", "quantum_variational"]
    )
    def test_registered_families_instantiate(self, family: str) -> None:
        """All four registered families validate."""
        assert validate_envelope(envelope(family=family)).family == family
        assert require_active_family(family) == family

    def test_an_unknown_family_is_refused(self) -> None:
        """An unknown family is a plain envelope error."""
        with pytest.raises(EnvelopeError):
            require_active_family("vision")

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"schema": "envelope"}, EnvelopeError),
            ({"artefact_kind": "gguf"}, EnvelopeError),
            ({"content_digest": "X" * 64}, MalformedDigestError),
            (
                {"lineage_edges": [edge(digest_of("a"), digest_of("candidate"))]},
                EnvelopeError,
            ),
            ({"lineage_edges": ("edge",)}, EnvelopeError),
            ({"lineage_edges": (edge(digest_of("a"), digest_of("b")),)}, EnvelopeError),
            ({"recipe_digest": ""}, MalformedDigestError),
            ({"licence_record": licence(spdx_expression="")}, LicenceRecordError),
            ({"budget_spent": budget(max_wall_seconds=-1)}, LicenceRecordError),
            ({"spend_reservation_digest": ""}, MalformedDigestError),
            ({"publishable": "no"}, EnvelopeError),
            ({"excluded_reviewer_vendors": ("a", "a")}, EnvelopeError),
            ({"excluded_reviewer_vendors": ("",)}, EnvelopeError),
            ({"excluded_reviewer_vendors": ["a"]}, EnvelopeError),
            ({"family_contract_schema": ""}, EnvelopeError),
            ({"family_contract_digest": ""}, MalformedDigestError),
        ],
    )
    def test_envelope_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Every envelope field is required and exact; J4-6 included."""
        with pytest.raises(error):
            validate_envelope(envelope(**changes))

    def test_a_non_envelope_is_refused(self) -> None:
        """The validator needs an envelope."""
        with pytest.raises(EnvelopeError):
            validate_envelope(licence())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"parent_digest": ""}, MalformedDigestError),
            ({"child_digest": digest_of("parent")}, EnvelopeError),
            ({"relation": "copied_from"}, EnvelopeError),
            ({"search_run_id": ""}, EnvelopeError),
            ({"declared_created_at_utc": "2026-11-21"}, EnvelopeError),
            ({"declared_created_at_utc": "2026-13-40T00:00:00Z"}, EnvelopeError),
        ],
    )
    def test_lineage_edge_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Edges need two distinct digests, a known relation and a real time."""
        with pytest.raises(error):
            validate_lineage_edge(
                edge(digest_of("parent"), digest_of("child"), **changes)
            )

    def test_a_non_edge_is_refused(self) -> None:
        """The edge validator needs an edge."""
        with pytest.raises(EnvelopeError):
            validate_lineage_edge("edge")  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it


class TestCustodyRecords:
    """The 17-key custody ingest record and its kind rules."""

    def test_records_have_exactly_seventeen_keys(self) -> None:
        """E10-4: one shape for every kind."""
        assert len(typing.get_type_hints(CustodyIngestRecord)) == 17

    @pytest.mark.parametrize(
        "kind",
        [
            "artefact",
            "base_checkpoint",
            "preregistration",
            "preregistration_approval",
            "split",
            "split_pool",
            "corpus",
            "grader_bundle",
        ],
    )
    def test_every_kind_has_a_valid_record(self, kind: str) -> None:
        """Each kind validates with its default data class and format."""
        record = custody_record(digest_of(kind), kind)
        assert validate_custody_ingest_record(record) == record
        assert is_sealed_kind(kind) == (kind in SEALED_CUSTODY_KINDS)

    def test_the_sealed_set_is_fixed(self) -> None:
        """P5-3a and P5-3b: corpus and grader bundles are sealed too."""
        assert (
            frozenset({"split", "split_pool", "corpus", "grader_bundle"})
            == SEALED_CUSTODY_KINDS
        )

    def test_a_none_format_artefact_needs_a_reason(self) -> None:
        """ONNX or GGUF weights record no tensor-set digest and a reason."""
        record = custody_record(
            digest_of("onnx"),
            tensor_set_digest=None,
            weight_format="none",
            weight_format_reason="ONNX export",
        )
        assert validate_custody_ingest_record(record) == record
        with pytest.raises(CustodyRecordError, match="weight_format_reason"):
            validate_custody_ingest_record(replace(record, weight_format_reason=None))

    @pytest.mark.parametrize(
        ("kind", "changes", "error"),
        [
            ("artefact", {"schema": "custody-ingest-record.v2"}, CustodyRecordError),
            ("artefact", {"programme": ""}, CustodyRecordError),
            ("model", {}, CustodyRecordError),
            ("artefact", {"content_digest": ""}, MalformedDigestError),
            ("artefact", {"custody_ingest_seq": -1}, CustodyRecordError),
            ("artefact", {"submitting_role": ""}, CustodyRecordError),
            ("artefact", {"submitting_holder": ""}, CustodyRecordError),
            ("artefact", {"submitting_uid": -1}, CustodyRecordError),
            ("artefact", {"submitting_pid": 0}, CustodyRecordError),
            ("artefact", {"role_authentication_mode": "trusted"}, CustodyRecordError),
            ("artefact", {"byte_length": -1}, CustodyRecordError),
            ("artefact", {"ingested_at_utc": "yesterday"}, CustodyRecordError),
            ("artefact", {"data_class": "not_applicable"}, CustodyRecordError),
            ("preregistration", {"data_class": "pilot"}, CustodyRecordError),
            ("artefact", {"weight_format": "onnx"}, CustodyRecordError),
            ("split", {"weight_format": "safetensors"}, CustodyRecordError),
            ("artefact", {"tensor_set_digest": None}, MalformedDigestError),
            ("split", {"tensor_set_digest": digest_of("t")}, CustodyRecordError),
            ("split", {"weight_format_reason": "text"}, CustodyRecordError),
            ("artefact", {"base_checkpoint": base_checkpoint()}, CustodyRecordError),
            ("base_checkpoint", {"base_checkpoint": None}, CustodyRecordError),
            (
                "base_checkpoint",
                {"tensor_set_digest": digest_of("other")},
                CustodyRecordError,
            ),
        ],
    )
    def test_record_refusals(
        self, kind: str, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Every field and kind rule refuses its violation."""
        with pytest.raises(error):
            validate_custody_ingest_record(
                custody_record(digest_of("x"), kind, 1, PROPOSER_A, **changes)
            )

    def test_a_base_checkpoint_file_digest_must_match(self) -> None:
        """The ingested file is the record's content."""
        record = custody_record(digest_of("base"), "base_checkpoint")
        with pytest.raises(CustodyRecordError, match="file digest"):
            validate_custody_ingest_record(
                replace(record, base_checkpoint=base_checkpoint())
            )

    def test_a_non_record_is_refused(self) -> None:
        """The record validator needs a record."""
        with pytest.raises(CustodyRecordError):
            validate_custody_ingest_record(licence())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it


class TestBaseCheckpointsAndApprovals:
    """Base checkpoint provenance and preregistration approvals."""

    @pytest.mark.parametrize(
        "changes",
        [
            {
                "upstream_revision_kind": "doi_version",
                "upstream_revision": "10.5281/zenodo.1.v2",
            },
            {
                "upstream_published_digest_kind": "sha384",
                "upstream_published_digest": "a" * 96,
            },
            {
                "upstream_published_digest_kind": "git_blob_oid",
                "upstream_published_digest": "b" * 40,
            },
            {"upstream_published_digest_kind": "none", "upstream_published_digest": ""},
            {"tensor_set_digest": None},
        ],
    )
    def test_valid_variants(self, changes: dict[str, object]) -> None:
        """Each revision and digest kind validates in its own form."""
        record = base_checkpoint(**changes)
        assert validate_base_checkpoint_record(record) == record

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"schema": "base"}, CustodyRecordError),
            ({"repository_url": "http://example.org/x"}, CustodyRecordError),
            ({"upstream_revision_kind": "tag"}, CustodyRecordError),
            ({"upstream_revision": "main"}, CustodyRecordError),
            ({"upstream_published_digest_kind": "md5"}, CustodyRecordError),
            ({"upstream_published_digest": "short"}, CustodyRecordError),
            ({"upstream_published_digest": None}, CustodyRecordError),
            ({"custody_ingested_file_sha256": ""}, MalformedDigestError),
            ({"tensor_set_digest": ""}, MalformedDigestError),
            ({"weight_spdx_expression": ""}, CustodyRecordError),
            ({"commercial_and_derivative_clear": "yes"}, CustodyRecordError),
        ],
    )
    def test_base_checkpoint_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Each provenance field is exact."""
        with pytest.raises(error):
            validate_base_checkpoint_record(base_checkpoint(**changes))

    def test_approvals_validate_and_refuse(self) -> None:
        """An approval names a digest, a pool and a real time."""
        approval = PreregistrationApproval(
            "preregistration-approval.v1", digest_of("p"), "pool-1", UTC
        )
        assert validate_preregistration_approval(approval) == approval
        for changes in ({"schema": "x"}, {"pool_id": ""}, {"approved_at_utc": "now"}):
            with pytest.raises(CustodyRecordError):
                validate_preregistration_approval(replace(approval, **changes))  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields
        with pytest.raises(MalformedDigestError):
            validate_preregistration_approval(
                replace(approval, preregistration_digest="")
            )

    def test_wrong_types_are_refused(self) -> None:
        """Both validators need their own record type."""
        with pytest.raises(CustodyRecordError):
            validate_base_checkpoint_record(licence())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it
        with pytest.raises(CustodyRecordError):
            validate_preregistration_approval(licence())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it


class TestGradersPoolsAndPreregistrations:
    """Graders, sealed split pools and preregistration records."""

    def test_valid_records_pass(self) -> None:
        """The builders' records are valid."""
        assert validate_grader(grader()) == grader()
        assert validate_sealed_split_pool(pool()) == pool()
        assert validate_preregistration(preregistration()) == preregistration()

    def test_a_non_exact_grader_needs_a_calibration_report(self) -> None:
        """F-P8-01a: calibration is required unless the grader is exact."""
        calibrated = grader(
            is_exact_deterministic=False, calibration_report_digest=digest_of("cal")
        )
        assert validate_grader(calibrated) == calibrated
        assert validate_grader(grader(calibration_report_digest=digest_of("cal")))
        with pytest.raises(MalformedDigestError):
            validate_grader(grader(is_exact_deterministic=False))

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"grader_id": ""}, EvaluationContractError),
            ({"implementation_digest": ""}, MalformedDigestError),
            ({"roles": ()}, EvaluationContractError),
            ({"roles": ["sealed"]}, EvaluationContractError),
            ({"roles": ("judge",)}, EvaluationContractError),
            ({"roles": ("sealed", "sealed")}, EvaluationContractError),
            ({"is_exact_deterministic": 1}, EvaluationContractError),
            ({"calibration_report_digest": "x"}, MalformedDigestError),
        ],
    )
    def test_grader_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Each grader field is exact."""
        with pytest.raises(error):
            validate_grader(grader(**changes))

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"pool_id": ""}, EvaluationContractError),
            (
                {"split_digests": (digest_of("1"), digest_of("2"))},
                EvaluationContractError,
            ),
            ({"split_digests": ()}, EvaluationContractError),
            ({"split_digests": (digest_of("1"),) * 3}, EvaluationContractError),
            ({"deduplication_report_digest": ""}, MalformedDigestError),
        ],
    )
    def test_pool_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """A pool holds at least three distinct splits and a dedup report."""
        with pytest.raises(error):
            validate_sealed_split_pool(pool(**changes))

    def test_a_sealed_grader_that_also_trains_is_refused(self) -> None:
        """M-J4-8 (graders): a sealed grader never trains or selects."""
        for roles in (("sealed", "training"), ("sealed", "selection")):
            prereg = preregistration(graders=(grader(roles=roles),))
            with pytest.raises(EvaluationContractError, match="may not also train"):
                validate_preregistration(prereg)
        mixed = (grader(), grader(grader_id="g2", roles=("sealed", "training")))
        with pytest.raises(EvaluationContractError, match="may not also train"):
            validate_preregistration(preregistration(graders=mixed))

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"schema": "prereg"}, EvaluationContractError),
            ({"study_id": ""}, EvaluationContractError),
            ({"declared_arms": ()}, EvaluationContractError),
            ({"declared_arms": ("a", "a")}, EvaluationContractError),
            ({"declared_arms": ("",)}, EvaluationContractError),
            ({"search_spaces_and_budgets_digest": ""}, MalformedDigestError),
            ({"split_pool": pool(pool_id="")}, EvaluationContractError),
            ({"graders": (grader(roles=("training",)),)}, EvaluationContractError),
            ({"graders": ("grader",)}, EvaluationContractError),
            ({"graders": (grader(), grader())}, EvaluationContractError),
            ({"non_inferiority_margins_bps": ()}, EvaluationContractError),
            (
                {"non_inferiority_margins_bps": (("accuracy",),)},
                EvaluationContractError,
            ),
            (
                {"non_inferiority_margins_bps": (("accuracy", 0),)},
                EvaluationContractError,
            ),
            ({"non_inferiority_margins_bps": (("", 10),)}, EvaluationContractError),
            (
                {"non_inferiority_margins_bps": (("a", 10), ("a", 20))},
                EvaluationContractError,
            ),
            ({"planned_sample_size": 0}, EvaluationContractError),
            ({"target_power_permille": 1000}, EvaluationContractError),
            ({"target_power_permille": 0}, EvaluationContractError),
            ({"ci_method": "t_test"}, EvaluationContractError),
            ({"resampling_unit": ""}, EvaluationContractError),
            ({"pass_rule_digest": ""}, MalformedDigestError),
            ({"frozen_look_candidate_digests": ()}, EvaluationContractError),
            ({"analysis_code_digest": ""}, MalformedDigestError),
            ({"declared_registered_at_utc": ""}, EvaluationContractError),
            (
                {"base_checkpoint_digests": (digest_of("b"),) * 2},
                EvaluationContractError,
            ),
        ],
    )
    def test_preregistration_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Every preregistration field is exact."""
        with pytest.raises(error):
            validate_preregistration(preregistration(**changes))

    def test_wrong_types_are_refused(self) -> None:
        """Each validator needs its own record type."""
        for validator in (
            validate_grader,
            validate_sealed_split_pool,
            validate_preregistration,
        ):
            with pytest.raises(EvaluationContractError):
                validator(licence())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it

    def test_a_preregistration_round_trips_through_json(self) -> None:
        """Nested records and pairs decode back exactly."""
        prereg = preregistration(base_checkpoint_digests=(digest_of("base"),))
        decoded = decode_record(
            PreregistrationRecord,
            canonical_json_bytes(prereg),
            validate_preregistration,
        )
        assert decoded == prereg and record_digest(decoded) == record_digest(prereg)


class TestPairedSampleSize:
    """J4-7: exact quantiles and the ceiling."""

    @pytest.mark.parametrize(
        ("rate", "margin", "expected"),
        [(0.05, 0.01, 3925), (0.10, 0.02, 1963), (0.20, 0.03, 1745), (0.05, 0.03, 437)],
    )
    def test_the_vectors(self, rate: float, margin: float, expected: int) -> None:
        """M-J4-5: neither truncation (3924) nor 1.96 + 0.84 (3920) passes."""
        assert required_paired_sample_size(rate, margin) == expected

    @pytest.mark.parametrize(
        "arguments",
        [
            {"discordance_rate": 0.0, "margin": 0.01},
            {"discordance_rate": 1.5, "margin": 0.01},
            {"discordance_rate": 0.1, "margin": 1.0},
            {"discordance_rate": 0.1, "margin": 0.01, "alpha": 0.5},
            {"discordance_rate": 0.1, "margin": 0.01, "power": 0.5},
            {"discordance_rate": 1, "margin": 0.01},
            {"discordance_rate": float("nan"), "margin": 0.01},
        ],
    )
    def test_out_of_range_inputs_are_refused(self, arguments: dict[str, float]) -> None:
        """Every input is a finite float in its range."""
        with pytest.raises(SampleSizeError):
            required_paired_sample_size(**arguments)

    def test_every_refusal_is_a_lifecycle_error(self) -> None:
        """Callers can catch the whole family."""
        with pytest.raises(ModelLifecycleError):
            required_paired_sample_size(0.0, 0.01)
