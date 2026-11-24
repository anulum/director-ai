# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Family Registry Tests

"""Family rules, companion contracts, family weight checks and model identity."""

from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path

import pytest

from director_ai.model_lifecycle import (
    FAMILY_CONTRACT_REGISTRY,
    HDF5_MAGIC,
    CustodyRecordError,
    DynamicsFamilyContract,
    EnvelopeError,
    FixedPointFormatDecl,
    InconsistentFixedPointDeclarationError,
    LanguageFamilyContract,
    MalformedDigestError,
    MalformedWeightsError,
    MissingFixedPointDeclarationError,
    MissingRequiredTensorPrefixError,
    MissingTensorSetDigestError,
    NegativeZeroTensorError,
    NonFiniteTensorError,
    SecondaryFamilyContractRef,
    TensorDescriptor,
    UnsupportedWeightFormatError,
    canonical_json_bytes,
    compute_archive_tensor_set_identity,
    compute_family_model_identity_digest,
    compute_family_tensor_set_identity,
    compute_tensor_set_identity,
    family_rule,
    from_canonical_json_bytes,
    record_digest,
    validate_dynamics_family_contract,
    validate_family_tensor_layout,
    validate_language_family_contract,
    validate_lineage_edge,
)
from tests.model_lifecycle_support import (
    TensorSpec,
    custody_record,
    digest_of,
    edge,
    envelope,
    safetensors_bytes,
    ustar_archive,
)

ONE = struct.pack("<f", 1.0)
NEGATIVE_ZERO = struct.pack("<f", -0.0)
NAN = struct.pack("<f", float("nan"))


def f32(name: str, payload: bytes = ONE) -> TensorSpec:
    """Return a one-element F32 tensor spec."""
    return (name, "F32", [len(payload) // 4], payload)


def integer(name: str, dtype: str) -> TensorSpec:
    """Return a one-element integer tensor spec of ``dtype``."""
    width = {"I16": 2, "I32": 4, "I64": 8}[dtype]
    return (name, dtype, [1], bytes(width))


SPIKING = [
    f32("pop.0.v_threshold"),
    f32("edge.0.weight"),
    integer("edge.0.delay_steps", "I32"),
    integer("qat.0.weight", "I16"),
    f32("readout.weight"),
]
Q8_8 = {"qat.0.weight": FixedPointFormatDecl(True, 16, 8)}
QUANTUM = [f32("ansatz.theta.0"), f32("observable.coeff.0")]


def write(tmp_path: Path, tensors: list[TensorSpec], name: str = "m") -> Path:
    """Write one safetensors file and return its path."""
    path = tmp_path / f"{name}.safetensors"
    path.write_bytes(safetensors_bytes(tensors))
    return path


def descriptor(name: str, dtype: str) -> TensorDescriptor:
    """Return a one-element descriptor for layout checks without a file."""
    return TensorDescriptor(name, dtype, (1,), digest_of(name))  # type: ignore[arg-type]  # the helper takes any string so refusal tests can pass unknown values


class TestRegistry:
    """The four registered families and the B2-1 negative-zero scope."""

    def test_four_families_with_their_companion_prefixes(self) -> None:
        """Each rule is filed under its own family with its schema prefix."""
        prefixes = {
            family: rule.companion_schema_prefix
            for family, rule in FAMILY_CONTRACT_REGISTRY.items()
        }
        assert prefixes == {
            "language": "language-family-contract.",
            "dynamics": "dynamics-family-contract.",
            "spiking": "sc-neurocore.spiking-artefact-contract.",
            "quantum_variational": "scpn-qc.variational-artefact-contract.",
        }
        assert all(r.family_id == f for f, r in FAMILY_CONTRACT_REGISTRY.items())

    def test_negative_zero_is_refused_only_for_spiking_and_quantum(self) -> None:
        """B2-1: trained language and dynamics weights keep negative zeros."""
        forbidding = {
            f
            for f, rule in FAMILY_CONTRACT_REGISTRY.items()
            if rule.forbid_negative_zero
        }
        assert forbidding == {"spiking", "quantum_variational"}

    def test_an_unknown_family_has_no_rule(self) -> None:
        """A family outside the registry is refused."""
        with pytest.raises(EnvelopeError):
            family_rule("vision")

    @pytest.mark.parametrize(
        "relation", ["pretrained_from_scratch", "continued_pretrained_from"]
    )
    def test_the_two_new_lineage_relations_validate(self, relation: str) -> None:
        """Lineage edges accept the two pretraining relations."""
        item = edge(digest_of("p"), digest_of("c"), relation=relation)
        assert validate_lineage_edge(item).relation == relation


class TestFamilyWeights:
    """Reading weights under each family's rule."""

    def test_a_spiking_model_reads_to_the_plain_identity(self, tmp_path: Path) -> None:
        """A valid spiking file has the same identity as without family rules."""
        path = write(tmp_path, SPIKING)
        identity = compute_family_tensor_set_identity(
            "spiking", "safetensors", [path], fixed_point_map=Q8_8
        )
        assert identity == compute_tensor_set_identity([path])

    def test_shards_and_archives_read_under_the_family_rule(
        self, tmp_path: Path
    ) -> None:
        """Shards and one archive give the identity of the single file."""
        whole = compute_tensor_set_identity([write(tmp_path, QUANTUM)])
        shards = [
            write(tmp_path, QUANTUM[:1], "a"),
            write(tmp_path, QUANTUM[1:], "b"),
        ]
        assert (
            compute_family_tensor_set_identity(
                "quantum_variational", "safetensors", shards
            )
            == whole
        )
        archive = tmp_path / "m.tar"
        archive.write_bytes(
            ustar_archive(
                [
                    (b"a.safetensors", safetensors_bytes(QUANTUM[:1])),
                    (b"b.safetensors", safetensors_bytes(QUANTUM[1:])),
                ]
            )
        )
        assert (
            compute_family_tensor_set_identity(
                "quantum_variational", "safetensors_tar", [archive]
            )
            == whole
            == compute_archive_tensor_set_identity(archive)
        )
        with pytest.raises(MalformedWeightsError, match="exactly one archive"):
            compute_family_tensor_set_identity(
                "quantum_variational", "safetensors_tar", [archive, archive]
            )

    @pytest.mark.parametrize("family", ["spiking", "quantum_variational"])
    def test_tqv_3_negative_zero_is_refused_in_its_families(
        self, tmp_path: Path, family: str
    ) -> None:
        """TQV-3: a negative zero refuses spiking and quantum weights."""
        tensors = SPIKING if family == "spiking" else QUANTUM
        bad = [f32(tensors[0][0], NEGATIVE_ZERO), *tensors[1:]]
        table = Q8_8 if family == "spiking" else None
        compute_family_tensor_set_identity(
            family,
            "safetensors",
            [write(tmp_path, tensors, "ok")],
            fixed_point_map=table,
        )
        with pytest.raises(NegativeZeroTensorError):
            compute_family_tensor_set_identity(
                family, "safetensors", [write(tmp_path, bad)], fixed_point_map=table
            )
        archive = tmp_path / "m.tar"
        archive.write_bytes(ustar_archive([(b"m.safetensors", safetensors_bytes(bad))]))
        with pytest.raises(NegativeZeroTensorError):
            compute_family_tensor_set_identity(
                family, "safetensors_tar", [archive], fixed_point_map=table
            )

    @pytest.mark.parametrize("family", ["language", "dynamics"])
    def test_b2_1_negative_zero_is_kept_in_trained_weights(
        self, tmp_path: Path, family: str
    ) -> None:
        """B2-1: language and dynamics weights may hold negative zeros."""
        path = write(tmp_path, [f32("w", NEGATIVE_ZERO + ONE)])
        identity = compute_family_tensor_set_identity(family, "safetensors", [path])
        assert identity == compute_tensor_set_identity([path])

    @pytest.mark.parametrize("family", sorted(FAMILY_CONTRACT_REGISTRY))
    def test_tqv_3_nan_is_refused_in_every_family(
        self, tmp_path: Path, family: str
    ) -> None:
        """NaN is refused whatever the family."""
        tensors = {"spiking": SPIKING, "quantum_variational": QUANTUM}.get(
            family, [f32("w")]
        )
        bad = [f32(tensors[0][0], NAN), *tensors[1:]]
        table = Q8_8 if family == "spiking" else None
        with pytest.raises(NonFiniteTensorError):
            compute_family_tensor_set_identity(
                family, "safetensors", [write(tmp_path, bad)], fixed_point_map=table
            )

    def test_tsn_1_hdf5_weights_are_refused(self, tmp_path: Path) -> None:
        """TSN-1: a NIR HDF5 file is refused even when declared safetensors."""
        path = tmp_path / "graph.nir"
        path.write_bytes(HDF5_MAGIC + bytes(64))
        with pytest.raises(UnsupportedWeightFormatError, match="HDF5"):
            compute_family_tensor_set_identity("spiking", "safetensors", [path])

    @pytest.mark.parametrize("weight_format", ["none", "gguf", ""])
    def test_formats_outside_the_rule_are_refused(
        self, tmp_path: Path, weight_format: str
    ) -> None:
        """No family accepts weights without a safetensors format."""
        with pytest.raises(UnsupportedWeightFormatError, match="does not accept"):
            compute_family_tensor_set_identity(
                "language", weight_format, [write(tmp_path, [f32("w")])]
            )

    def test_no_file_is_refused(self) -> None:
        """A model needs at least one weights file."""
        with pytest.raises(MalformedWeightsError, match="at least one"):
            compute_family_tensor_set_identity("language", "safetensors", [])

    def test_layout_errors_surface_through_the_file_reader(
        self, tmp_path: Path
    ) -> None:
        """A file whose tensors break the family's layout is refused."""
        path = write(tmp_path, SPIKING)
        with pytest.raises(MissingFixedPointDeclarationError):
            compute_family_tensor_set_identity("spiking", "safetensors", [path])


def _layout(
    family: str,
    names: list[tuple[str, str]],
    fixed_point_map: dict[str, object] | None = None,
) -> None:
    """Run the layout check on descriptors built from ``(name, dtype)`` pairs."""
    validate_family_tensor_layout(
        family,
        [descriptor(name, dtype) for name, dtype in names],
        fixed_point_map=fixed_point_map or {},  # type: ignore[arg-type]  # the helper takes any mapping so refusal tests can pass malformed formats
    )


SPIKING_NAMES = [
    ("pop.0.v", "F32"),
    ("edge.0.weight", "F32"),
    ("edge.0.delay_steps", "I64"),
]


class TestLayout:
    """Prefixes, dtypes per prefix and the spiking fixed-point table."""

    @pytest.mark.parametrize(
        "names",
        [
            [("edge.0.weight", "F32")],
            [("pop.0.v", "F32")],
            [*SPIKING_NAMES, ("membrane", "F32")],
        ],
        ids=["no-pop", "no-edge", "unprefixed"],
    )
    def test_tsn_2_spiking_prefixes(self, names: list[tuple[str, str]]) -> None:
        """TSN-2: spiking needs ``pop.`` and ``edge.`` and nothing unprefixed."""
        with pytest.raises(MissingRequiredTensorPrefixError):
            _layout("spiking", names)

    @pytest.mark.parametrize(
        "names",
        [[("classical_head.w", "F32")], [("rbm.visible_bias", "F64")]],
        ids=["no-theta", "rbm"],
    )
    def test_tqv_2_quantum_needs_ansatz_angles(
        self, names: list[tuple[str, str]]
    ) -> None:
        """TQV-2: quantum weights without ``ansatz.theta.`` are refused, RBMs too."""
        with pytest.raises(MissingRequiredTensorPrefixError):
            _layout("quantum_variational", names)

    @pytest.mark.parametrize("dtype", ["BF16", "F16"])
    def test_tqv_1_half_precision_angles_are_refused(self, dtype: str) -> None:
        """TQV-1: BF16 and F16 are not quantum dtypes."""
        with pytest.raises(MalformedWeightsError) as caught:
            _layout("quantum_variational", [("ansatz.theta.0", dtype)])
        assert type(caught.value) is MalformedWeightsError

    @pytest.mark.parametrize("prefix", ["pop.0.v", "edge.0.weight", "readout.w"])
    def test_tsn_5_i16_only_under_qat(self, prefix: str) -> None:
        """TSN-5 (SP-1): I16 is refused under every prefix except ``qat.``."""
        names = [(n, "I16" if n == prefix else d) for n, d in SPIKING_NAMES]
        if prefix == "readout.w":
            names.append((prefix, "I16"))
        with pytest.raises(MalformedWeightsError) as caught:
            _layout("spiking", names)
        assert type(caught.value) is MalformedWeightsError

    @pytest.mark.parametrize("dtype", ["I16", "I32", "I64"])
    def test_integer_qat_tensors_with_matching_declarations_pass(
        self, dtype: str
    ) -> None:
        """SP-2: a signed declaration of the tensor's own width is accepted."""
        bits = {"I16": 16, "I32": 32, "I64": 64}[dtype]
        decl = FixedPointFormatDecl(True, bits, bits - 1)  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it
        _layout("spiking", [*SPIKING_NAMES, ("qat.w", dtype)], {"qat.w": decl})
        _layout(
            "spiking",
            [*SPIKING_NAMES, ("qat.w", dtype)],
            {"qat.w": FixedPointFormatDecl(True, bits, 0)},  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it
        )

    @pytest.mark.parametrize("dtype", ["I16", "I32", "I64"])
    def test_tsn_6_integer_qat_needs_a_declaration(self, dtype: str) -> None:
        """TSN-6 (SP-2): an integer ``qat.`` tensor without a declaration."""
        with pytest.raises(MissingFixedPointDeclarationError):
            _layout("spiking", [*SPIKING_NAMES, ("qat.w", dtype)])

    @pytest.mark.parametrize(
        ("dtype", "decl"),
        [
            ("I32", FixedPointFormatDecl(True, 16, 8)),
            ("I16", FixedPointFormatDecl(True, 16, 16)),
            ("I16", FixedPointFormatDecl(True, 16, -1)),
            ("I16", FixedPointFormatDecl(False, 16, 8)),  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it
            ("I16", FixedPointFormatDecl(True, 16, True)),
            ("I16", FixedPointFormatDecl(True, 16, 8.0)),  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it
            ("I16", (True, 16, 8)),
        ],
        ids=[
            "width",
            "fraction-full",
            "fraction-negative",
            "unsigned",
            "bool",
            "float",
            "tuple",
        ],
    )
    def test_tsn_7_inconsistent_declarations(self, dtype: str, decl: object) -> None:
        """TSN-7 (SP-2): a declaration must match its tensor exactly."""
        with pytest.raises(InconsistentFixedPointDeclarationError):
            _layout("spiking", [*SPIKING_NAMES, ("qat.w", dtype)], {"qat.w": decl})

    @pytest.mark.parametrize(
        ("family", "names", "name"),
        [
            ("spiking", [], "edge.0.delay_steps"),
            ("spiking", [("qat.f", "F32")], "qat.f"),
            ("spiking", [], "qat.missing"),
            ("language", [("qat.w", "I16")], "qat.w"),
        ],
        ids=["integer-edge", "float-qat", "unknown", "language"],
    )
    def test_tsn_7_declarations_on_other_tensors_are_refused(
        self, family: str, names: list[tuple[str, str]], name: str
    ) -> None:
        """TSN-7 (SP-3): only integer ``qat.`` spiking tensors carry declarations."""
        base = SPIKING_NAMES if family == "spiking" else []
        with pytest.raises(InconsistentFixedPointDeclarationError, match=name):
            _layout(family, [*base, *names], {name: FixedPointFormatDecl(True, 16, 8)})

    @pytest.mark.parametrize(
        ("family", "dtype"),
        [("dynamics", "F8_E4M3"), ("language", "BOOL"), ("language", "U16")],
    )
    def test_dtypes_outside_a_family_are_refused(self, family: str, dtype: str) -> None:
        """Each family accepts only its own dtypes."""
        with pytest.raises(MalformedWeightsError):
            _layout(family, [("w", dtype)])

    @pytest.mark.parametrize("dtype", ["I8", "U8", "F8_E5M2", "BF16"])
    def test_language_accepts_its_dtypes_under_any_name(self, dtype: str) -> None:
        """Language has no prefix rule and keeps its quantised dtypes."""
        _layout("language", [("any.name", dtype)])


def language_contract(**changes: object) -> LanguageFamilyContract:
    """Return a valid language companion contract with ``changes`` applied."""
    base = LanguageFamilyContract(
        "language-family-contract.v1",
        digest_of("tokeniser"),
        None,
        digest_of("embedding"),
        False,
        (SecondaryFamilyContractRef("dynamics-family-contract.v1", digest_of("dyn")),),
    )
    return replace(base, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


def dynamics_contract(**changes: object) -> DynamicsFamilyContract:
    """Return a valid dynamics companion contract with ``changes`` applied."""
    base = DynamicsFamilyContract(
        "dynamics-family-contract.v1",
        digest_of("normaliser"),
        digest_of("architecture"),
        1000,
        ("m", "m", "m/s"),
        (),
        "review_only_non_actuating",
        (),
    )
    return replace(base, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


REF = SecondaryFamilyContractRef("x.v1", digest_of("x"))


class TestCompanions:
    """The language and dynamics companion contracts."""

    @pytest.mark.parametrize(
        "contract",
        [language_contract(), dynamics_contract()],
        ids=["language", "dynamics"],
    )
    def test_companions_round_trip_through_the_codec(self, contract: object) -> None:
        """Both companion contracts decode and validate through the codec."""
        raw = canonical_json_bytes(contract)
        assert from_canonical_json_bytes(type(contract), raw) == contract

    def test_optional_fields_may_be_empty(self) -> None:
        """Absent digests, no timestep and no action units are valid."""
        validate_language_family_contract(
            language_contract(
                embedding_space_digest=None, secondary_family_contracts=()
            )
        )
        validate_dynamics_family_contract(
            dynamics_contract(architecture_spec_digest=None, timestep_microseconds=None)
        )

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"schema": "language-family-contract.v2"}, EnvelopeError),
            ({"tokeniser_envelope_digest": "x"}, MalformedDigestError),
            ({"architecture_spec_digest": "x"}, MalformedDigestError),
            ({"embedding_space_digest": ""}, MalformedDigestError),
            ({"tie_word_embeddings": 1}, EnvelopeError),
            ({"secondary_family_contracts": [REF]}, EnvelopeError),
            ({"secondary_family_contracts": ("x.v1",)}, EnvelopeError),
            ({"secondary_family_contracts": (REF, REF)}, EnvelopeError),
            (
                {"secondary_family_contracts": (replace(REF, schema=""),)},
                EnvelopeError,
            ),
            (
                {"secondary_family_contracts": (replace(REF, digest="x"),)},
                MalformedDigestError,
            ),
        ],
    )
    def test_language_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Each malformed language field is refused by name."""
        with pytest.raises(error):
            validate_language_family_contract(language_contract(**changes))

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"schema": "language-family-contract.v1"}, EnvelopeError),
            ({"normaliser_envelope_digest": "x"}, MalformedDigestError),
            ({"architecture_spec_digest": "x"}, MalformedDigestError),
            ({"timestep_microseconds": 0}, EnvelopeError),
            ({"timestep_microseconds": True}, EnvelopeError),
            ({"state_units_ucum": ()}, EnvelopeError),
            ({"state_units_ucum": ("m", "")}, EnvelopeError),
            ({"state_units_ucum": ["m"]}, EnvelopeError),
            ({"action_units_ucum": ("",)}, EnvelopeError),
            ({"actuation_mode": "closed_loop_hardware"}, EnvelopeError),
            ({"secondary_family_contracts": (REF, REF)}, EnvelopeError),
        ],
    )
    def test_dynamics_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Each malformed dynamics field is refused by name."""
        with pytest.raises(error):
            validate_dynamics_family_contract(dynamics_contract(**changes))

    def test_a_contract_of_the_other_type_is_refused(self) -> None:
        """Each validator checks its own record type."""
        with pytest.raises(EnvelopeError):
            validate_language_family_contract(dynamics_contract())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it
        with pytest.raises(EnvelopeError):
            validate_dynamics_family_contract(language_contract())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it


class TestModelIdentity:
    """Q-8: the family model identity digest."""

    def test_the_identity_binds_tensors_family_and_contract(self) -> None:
        """The digest is the canonical document of the five bound fields."""
        item = envelope()
        record = custody_record(item.content_digest)
        expected = record_digest(
            {
                "schema": "family-model-identity.v1",
                "family": "language",
                "tensor_set_digest": record.tensor_set_digest,
                "family_contract_schema": "language-family-contract.v1",
                "family_contract_digest": item.family_contract_digest,
            }
        )
        assert compute_family_model_identity_digest(item, record) == expected
        other = replace(item, family_contract_digest=digest_of("other contract"))
        assert compute_family_model_identity_digest(other, record) != expected

    def test_a_spiking_model_binds_its_domain_contract(self) -> None:
        """A spiking envelope names the spiking companion schema."""
        item = envelope(
            family="spiking",
            family_contract_schema="sc-neurocore.spiking-artefact-contract.v1",
        )
        record = custody_record(item.content_digest)
        assert len(compute_family_model_identity_digest(item, record)) == 64

    def test_a_record_of_another_artefact_is_refused(self) -> None:
        """The ingest record must carry the envelope's content digest."""
        with pytest.raises(EnvelopeError, match="another artefact"):
            compute_family_model_identity_digest(
                envelope(), custody_record(digest_of("other"))
            )

    def test_weights_without_a_tensor_set_are_refused(self) -> None:
        """Weight format ``none`` has no tensor set to bind."""
        item = envelope()
        record = custody_record(
            item.content_digest,
            tensor_set_digest=None,
            weight_format="none",
            weight_format_reason="exported graph without safetensors",
        )
        with pytest.raises(MissingTensorSetDigestError):
            compute_family_model_identity_digest(item, record)

    def test_a_schema_of_another_family_is_refused(self) -> None:
        """The companion schema must carry the envelope family's prefix."""
        item = envelope(family_contract_schema="dynamics-family-contract.v1")
        with pytest.raises(EnvelopeError, match="must start with"):
            compute_family_model_identity_digest(
                item, custody_record(item.content_digest)
            )

    def test_malformed_inputs_are_refused_first(self) -> None:
        """A malformed envelope or record never yields an identity."""
        item = envelope()
        with pytest.raises(MalformedDigestError):
            compute_family_model_identity_digest(
                replace(item, family_contract_digest="x"),
                custody_record(item.content_digest),
            )
        with pytest.raises(CustodyRecordError):
            compute_family_model_identity_digest(
                item, custody_record(item.content_digest, submitting_role="")
            )
