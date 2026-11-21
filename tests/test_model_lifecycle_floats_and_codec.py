# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — model lifecycle float rules and record codec tests

"""Negative zero in canonical JSON, the tensor float scanner and the codec."""

from __future__ import annotations

import io
import json
import random
import struct
from pathlib import Path

import pytest

from director_ai.model_lifecycle import (
    FLOAT_DTYPES,
    EvaluationContractError,
    EvaluatorOutput,
    LicenceRecordError,
    MalformedWeightsError,
    NegativeZeroTensorError,
    NonCanonicalFloatError,
    NonCanonicalJsonError,
    NonFiniteTensorError,
    PreregistrationApproval,
    SealedLookResult,
    SliceAggregateEstimate,
    TensorDescriptor,
    TensorSetIdentity,
    any_lane_matches,
    canonical_json_bytes,
    compute_archive_tensor_set_identity,
    compute_tensor_set_digest,
    compute_tensor_set_identity,
    from_canonical_json_bytes,
    read_safetensors_descriptors,
    record_digest,
    scan_float_chunk,
    strict_json_loads,
)
from tests.model_lifecycle_support import (
    FIXTURE_DIGEST,
    TENSOR_A,
    TENSOR_B,
    UTC,
    LookScenario,
    base_checkpoint,
    budget,
    custody_record,
    digest_of,
    edge,
    envelope,
    grader,
    journal_entry,
    licence,
    pool,
    preregistration,
    safetensors_bytes,
    ustar_archive,
)

_WIDTH = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E5M2": 1, "F8_E4M3": 1}
_EXPONENT = {
    "F64": 0x7FF0000000000000,
    "F32": 0x7F800000,
    "F16": 0x7C00,
    "BF16": 0x7F80,
    "F8_E5M2": 0x7C,
    "F8_E4M3": 0x7F,
}
_ONE = {"F64": struct.pack("<d", 1.0), "F32": struct.pack("<f", 1.0), "F16": struct.pack("<e", 1.0),
        "BF16": struct.pack("<f", 1.0)[2:], "F8_E5M2": b"\x3c", "F8_E4M3": b"\x38"}  # fmt: skip


class TestCanonicalNegativeZero:
    """Negative zero has no canonical JSON form."""

    def test_the_encoder_refuses_negative_zero(self) -> None:
        """``-0.0`` is refused; ``0.0`` and other floats encode."""
        with pytest.raises(NonCanonicalFloatError, match="negative zero"):
            canonical_json_bytes({"a": -0.0})
        with pytest.raises(NonCanonicalFloatError, match="finite"):
            canonical_json_bytes({"a": float("inf")})
        assert canonical_json_bytes({"a": 0.0, "b": -1.5}) == b'{"a":0.0,"b":-1.5}'

    @pytest.mark.parametrize("token", ["-0", "-0.0", "-0e0", "-0E+5", "-0.000"])
    def test_the_decoder_refuses_negative_zero_tokens(self, token: str) -> None:
        """Every spelling of a negative zero is refused."""
        with pytest.raises(NonCanonicalFloatError, match="negative zero"):
            strict_json_loads(f'{{"a": {token}}}'.encode())

    def test_the_decoder_keeps_ordinary_numbers(self) -> None:
        """Zero, negative numbers and exponents decode."""
        assert strict_json_loads(b'{"a": 0, "b": 0.0, "c": -1, "d": -2.5e1}') == {
            "a": 0,
            "b": 0.0,
            "c": -1,
            "d": -25.0,
        }


class TestLaneScan:
    """The lane test agrees with an element-by-element scan."""

    @pytest.mark.parametrize("dtype", sorted(FLOAT_DTYPES))
    def test_the_lane_test_matches_a_naive_scan(self, dtype: str) -> None:
        """Random chunks biased towards special values agree with a direct scan."""
        width, exponent = _WIDTH[dtype], _EXPONENT[dtype]
        rng = random.Random(7)
        full, sign = (1 << (8 * width)) - 1, 1 << (8 * width - 1)
        for _ in range(300):
            lanes = [rng.getrandbits(8 * width) for _ in range(rng.randint(1, 24))]
            for index in range(len(lanes)):
                pick = rng.random()
                lanes[index] = (
                    lanes[index] | exponent
                    if pick < 0.1
                    else sign
                    if pick < 0.2
                    else lanes[index]
                )
            chunk = b"".join(value.to_bytes(width, "little") for value in lanes)
            for mask, value in ((exponent, exponent), (full, sign)):
                expected = any((lane & mask) == value for lane in lanes)
                assert any_lane_matches(chunk, width, mask, value) is expected

    def test_an_empty_chunk_matches_nothing(self) -> None:
        """No lanes, no match."""
        assert not any_lane_matches(b"", 4, 0x7F800000, 0x7F800000)


def _special(dtype: str, kind: str) -> bytes:
    """Return one element of ``dtype`` that is a NaN, an infinity or a negative zero."""
    width = _WIDTH[dtype]
    if kind == "negative_zero":
        return (1 << (8 * width - 1)).to_bytes(width, "little")
    pattern = _EXPONENT[dtype] | (1 if kind == "nan" else 0)
    return pattern.to_bytes(width, "little")


class TestScanner:
    """NaN and infinities are always refused; negative zero only when asked."""

    @pytest.mark.parametrize("dtype", sorted(FLOAT_DTYPES))
    @pytest.mark.parametrize("kind", ["nan", "infinity"])
    def test_non_finite_values_are_refused(self, dtype: str, kind: str) -> None:
        """Every float dtype refuses its non-finite patterns."""
        chunk = _ONE[dtype] * 3 + _special(dtype, kind)
        with pytest.raises(NonFiniteTensorError):
            scan_float_chunk(dtype, chunk, forbid_negative_zero=False, source="t")

    @pytest.mark.parametrize("dtype", sorted(FLOAT_DTYPES))
    def test_negative_zero_is_refused_only_on_request(self, dtype: str) -> None:
        """Negative zero passes by default and is refused when forbidden."""
        chunk = _ONE[dtype] + _special(dtype, "negative_zero")
        scan_float_chunk(dtype, chunk, forbid_negative_zero=False, source="t")
        with pytest.raises(NegativeZeroTensorError):
            scan_float_chunk(dtype, chunk, forbid_negative_zero=True, source="t")

    @pytest.mark.parametrize(
        ("dtype", "largest"),
        [
            ("F64", 0x7FEFFFFFFFFFFFFF),
            ("F32", 0x7F7FFFFF),
            ("F16", 0x7BFF),
            ("BF16", 0x7F7F),
            ("F8_E5M2", 0x7B),
            ("F8_E4M3", 0x7E),
        ],
    )
    def test_the_largest_finite_values_pass(self, dtype: str, largest: int) -> None:
        """The largest finite value of each dtype, and its negative, are not refused."""
        width = _WIDTH[dtype]
        negative = largest | (1 << (8 * width - 1))
        chunk = largest.to_bytes(width, "little") + negative.to_bytes(width, "little")
        scan_float_chunk(dtype, chunk, forbid_negative_zero=True, source="t")

    def test_integer_tensors_are_not_scanned(self) -> None:
        """A NaN bit pattern in an integer tensor is just an integer."""
        scan_float_chunk(
            "I32", _special("F32", "nan"), forbid_negative_zero=True, source="t"
        )


def _write(path: Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` and return it."""
    path.write_bytes(data)
    return path


class TestScannedDigests:
    """The scanner runs inside every digest path, chunk by chunk."""

    def test_the_vectors_still_hold(self, tmp_path: Path) -> None:
        """The reference tensors hold no forbidden value."""
        path = _write(
            tmp_path / "m.safetensors", safetensors_bytes([TENSOR_A, TENSOR_B])
        )
        assert (
            compute_tensor_set_digest([path], forbid_negative_zero=True)
            == FIXTURE_DIGEST
        )

    def test_a_nan_after_the_first_chunk_is_found(self, tmp_path: Path) -> None:
        """A NaN in the last element of a tensor larger than one chunk is refused."""
        payload = struct.pack("<f", 1.0) * 300_000 + struct.pack("<f", float("nan"))
        path = _write(
            tmp_path / "big.safetensors",
            safetensors_bytes([("w", "F32", [300_001], payload)]),
        )
        with pytest.raises(NonFiniteTensorError):
            compute_tensor_set_identity([path])

    def test_negative_zero_follows_the_flag(self, tmp_path: Path) -> None:
        """The same file passes by default and is refused when forbidden."""
        payload = struct.pack("<f", 1.0) + struct.pack("<f", -0.0)
        path = _write(
            tmp_path / "z.safetensors", safetensors_bytes([("w", "F32", [2], payload)])
        )
        assert compute_tensor_set_identity([path]).tensor_count == 1
        with pytest.raises(NegativeZeroTensorError):
            compute_tensor_set_identity([path], forbid_negative_zero=True)
        with pytest.raises(MalformedWeightsError):
            read_safetensors_descriptors(
                io.BytesIO(path.read_bytes()),
                offset=0,
                length=path.stat().st_size,
                source="z",
                forbid_negative_zero=True,
            )

    def test_archives_pass_the_flag_to_each_shard(self, tmp_path: Path) -> None:
        """A shard with a negative zero is refused when the archive forbids it."""
        payload = struct.pack("<f", -0.0)
        shard = safetensors_bytes([("w", "F32", [1], payload)])
        path = _write(tmp_path / "w.tar", ustar_archive([(b"a.safetensors", shard)]))
        assert compute_archive_tensor_set_identity(path).tensor_count == 1
        with pytest.raises(NegativeZeroTensorError):
            compute_archive_tensor_set_identity(path, forbid_negative_zero=True)


def _records() -> list[object]:
    """Return one valid instance of every registered record type."""
    scenario = LookScenario()
    entry = journal_entry(digest_of("lineage"), (digest_of("A1"),))
    estimate = SliceAggregateEstimate(
        digest_of("A1"), "accuracy", "all", 0.8, 0.7, 0.9, 30
    )
    output = EvaluatorOutput(
        "evaluator-output.v1", "slice_estimates", (estimate,), None, None
    )
    result = SealedLookResult(
        "sealed-look-result.v1",
        record_digest(entry),
        entry.derived_proposer_lineage_id,
        entry.split_digest,
        output,
    )
    from tests.test_model_lifecycle_looks import _decision

    return [
        licence(),
        budget(),
        edge(digest_of("p"), digest_of("c")),
        envelope(),
        base_checkpoint(),
        custody_record(digest_of("b"), "base_checkpoint"),
        PreregistrationApproval(
            "preregistration-approval.v1", digest_of("prereg"), "pool-1", UTC
        ),
        grader(),
        pool(),
        preregistration(),
        scenario.request(),
        entry,
        estimate,
        output,
        result,
        _decision(),
        TensorDescriptor("w", "F32", (2,), digest_of("data")),
        TensorSetIdentity("tensor-set-identity.v1", 1, 8, FIXTURE_DIGEST),
    ]


class TestRecordCodec:
    """Every registered record round-trips, and its validator always runs."""

    @pytest.mark.parametrize(
        "record", _records(), ids=lambda record: type(record).__name__
    )
    def test_every_registered_record_round_trips(self, record: object) -> None:
        """Decoding the canonical encoding returns an equal record."""
        assert (
            from_canonical_json_bytes(type(record), canonical_json_bytes(record))
            == record
        )

    @pytest.mark.parametrize(
        ("record", "error"),
        [
            (licence(data_class="not_applicable"), LicenceRecordError),
            (
                TensorSetIdentity("tensor-set-identity.v1", 0, 8, FIXTURE_DIGEST),
                MalformedWeightsError,
            ),
            (
                SealedLookResult(
                    "sealed-look-result.v1",
                    digest_of("entry"),
                    digest_of("lineage"),
                    digest_of("split"),
                    EvaluatorOutput(
                        "evaluator-output.v1", "slice_estimates", (), 1, None
                    ),
                ),
                EvaluationContractError,
            ),
        ],
        ids=["licence", "identity", "result"],
    )
    def test_the_validator_runs_after_decoding(
        self, record: object, error: type[Exception]
    ) -> None:
        """A well-formed but invalid record is refused by name."""
        with pytest.raises(error):
            from_canonical_json_bytes(type(record), canonical_json_bytes(record))

    def test_an_unregistered_type_is_refused(self) -> None:
        """Only registered types decode."""
        with pytest.raises(NonCanonicalJsonError, match="no registered validator"):
            from_canonical_json_bytes(dict, b"{}")

    def test_only_canonical_bytes_decode(self) -> None:
        """Other spellings of the same valid record are refused by name."""
        record = SliceAggregateEstimate(
            digest_of("A1"), "accuracy", "all", 0.5, 0.0, 0.75, 30
        )
        raw = canonical_json_bytes(record)
        assert from_canonical_json_bytes(SliceAggregateEstimate, raw) == record
        value = json.loads(raw)
        spellings = [
            raw.replace(b",", b", "),
            json.dumps(
                dict(reversed(sorted(value.items()))), separators=(",", ":")
            ).encode(),
            raw.replace(b":0.5,", b":0.50,"),
            raw.replace(b":0.0,", b":1e-400,"),
        ]
        for spelling in spellings:
            assert spelling != raw and json.loads(spelling) == value
            with pytest.raises(NonCanonicalJsonError, match="canonical encoding"):
                from_canonical_json_bytes(SliceAggregateEstimate, spelling)
