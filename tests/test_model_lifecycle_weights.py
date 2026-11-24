# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — model lifecycle weight identity tests

"""Tensor-set identity, the hardened safetensors reader and weight archives.

The digests are the peer-confirmed vectors of the kernel's design note.
"""

from __future__ import annotations

import hashlib
import io
import struct
import tarfile
from pathlib import Path
from typing import cast

import pytest

from director_ai.model_lifecycle import (
    MalformedDigestError,
    MalformedWeightsError,
    TensorDescriptor,
    TensorDtype,
    UnsupportedSafetensorsDtypeError,
    compute_archive_tensor_set_identity,
    compute_tensor_set_digest,
    compute_tensor_set_identity,
    list_archive_members,
    read_safetensors_descriptors,
    tensor_set_identity,
)
from tests.model_lifecycle_support import (
    FIXTURE_DIGEST,
    FLIPPED_DIGEST,
    TENSOR_A,
    TENSOR_B,
    safetensors_bytes,
    ustar_archive,
    ustar_header,
)


def _case_id(value: object) -> str:
    """Name a parametrised case by its expected message, not its bytes."""
    return value if isinstance(value, str) else "case"


def _write(path: Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` and return the path."""
    path.write_bytes(data)
    return path


class TestVectors:
    """The design note's four tensor-set vectors."""

    def test_tensor_digests_match_the_vectors(self, tmp_path: Path) -> None:
        """Each tensor's data digest is the SHA-256 of its raw bytes."""
        path = _write(
            tmp_path / "m.safetensors", safetensors_bytes([TENSOR_A, TENSOR_B])
        )
        with path.open("rb") as handle:
            found = read_safetensors_descriptors(
                handle, offset=0, length=path.stat().st_size, source="m"
            )
        assert [item.data_sha256 for item in found] == [
            "fa0176bb696ba40b43e4d73cc5d388fcb71414bf459e87826b780ea937df741e",
            "b9c80b5adeca450753a16950c3cc655d271f7bef7a485bc83f112b72fef21d37",
        ]
        assert found[0] == TensorDescriptor(
            "embed.weight", "BF16", (2, 2), found[0].data_sha256
        )

    def test_one_file_and_reversed_shards_give_one_digest(self, tmp_path: Path) -> None:
        """Shard order and differing metadata do not change the identity."""
        single = _write(
            tmp_path / "fixture_1.safetensors",
            safetensors_bytes([TENSOR_A, TENSOR_B], {"format": "pt"}),
        )
        shard_1 = _write(
            tmp_path / "shard_1.safetensors",
            safetensors_bytes([TENSOR_A], {"shard": "1"}),
        )
        shard_2 = _write(
            tmp_path / "shard_2.safetensors", safetensors_bytes([TENSOR_B])
        )
        identity = compute_tensor_set_identity([single])
        assert identity.tensor_set_digest == FIXTURE_DIGEST
        assert (identity.tensor_count, identity.total_byte_length) == (2, 16)
        assert compute_tensor_set_digest([shard_2, shard_1]) == FIXTURE_DIGEST

    def test_clearing_one_data_byte_changes_the_digest(self, tmp_path: Path) -> None:
        """256.0 becomes 128.0 in BF16; the header is identical."""
        flipped = (TENSOR_A[0], TENSOR_A[1], TENSOR_A[2], b"\x00" + TENSOR_A[3][1:])
        path = _write(
            tmp_path / "f.safetensors", safetensors_bytes([flipped, TENSOR_B])
        )
        assert compute_tensor_set_digest([path]) == FLIPPED_DIGEST

    def test_an_empty_path_list_is_refused(self) -> None:
        """A tensor set needs a file."""
        with pytest.raises(
            MalformedWeightsError, match="at least one safetensors file"
        ):
            compute_tensor_set_identity([])


class TestTensorSetIdentity:
    """Identity rules independent of the file format."""

    def test_empty_and_duplicate_sets_are_refused(self) -> None:
        """No tensors, or one name twice across shards, is malformed."""
        item = TensorDescriptor("w", "F32", (1,), "0" * 64)
        with pytest.raises(MalformedWeightsError, match="at least one tensor"):
            tensor_set_identity([])
        with pytest.raises(MalformedWeightsError, match="unique across shards"):
            tensor_set_identity([item, item])

    @pytest.mark.parametrize(
        ("item", "error", "message"),
        [
            (
                TensorDescriptor("w", cast(TensorDtype, "F4"), (1,), "0" * 64),
                UnsupportedSafetensorsDtypeError,
                "dtype",
            ),
            (
                TensorDescriptor("w", "F32", (-1,), "0" * 64),
                MalformedWeightsError,
                "shape",
            ),
            (
                TensorDescriptor("w", "F32", cast(tuple[int, ...], [1]), "0" * 64),
                MalformedWeightsError,
                "shape",
            ),
            (
                TensorDescriptor("", "F32", (1,), "0" * 64),
                MalformedWeightsError,
                "non-empty",
            ),
            (
                TensorDescriptor("\ud800", "F32", (1,), "0" * 64),
                MalformedWeightsError,
                "encode as UTF-8",
            ),
            ("w", MalformedWeightsError, "TensorDescriptor"),
        ],
    )
    def test_malformed_descriptors_are_refused(
        self, item: object, error: type[Exception], message: str
    ) -> None:
        """Each descriptor's name, dtype and shape are validated first."""
        with pytest.raises(error, match=message):
            tensor_set_identity([item])  # type: ignore[list-item]  # a malformed descriptor on purpose: it must be refused

    def test_a_malformed_data_digest_is_refused(self) -> None:
        """Descriptors must carry a real digest."""
        with pytest.raises(MalformedDigestError):
            tensor_set_identity([TensorDescriptor("w", "F32", (1,), "")])


def _parse(data: bytes, *, length: int | None = None) -> tuple[TensorDescriptor, ...]:
    """Parse ``data`` as one safetensors file."""
    return read_safetensors_descriptors(
        io.BytesIO(data),
        offset=0,
        length=len(data) if length is None else length,
        source="t",
    )


class TestHardenedParser:
    """Every refusal of the hardened safetensors reader."""

    @pytest.mark.parametrize(
        ("data", "message"),
        [
            (b"\x01\x00", "truncated"),
            (struct.pack("<Q", 0), "invalid header length"),
            (struct.pack("<Q", 100 * 1024 * 1024 + 1) + b"{}", "invalid header length"),
            (struct.pack("<Q", 50) + b"{}", "invalid header length"),
            (struct.pack("<Q", 2) + b"\xff\xfe", "header: JSON is not valid UTF-8"),
            (struct.pack("<Q", 2) + b"[]", "header must be a JSON object"),
            (struct.pack("<Q", 2) + b"{}", "holds no tensors"),
        ],
        ids=_case_id,
    )
    def test_framing_refusals(self, data: bytes, message: str) -> None:
        """Bad lengths, encodings and shapes of the header are refused."""
        with pytest.raises(MalformedWeightsError, match=message):
            _parse(data)

    def test_a_lone_surrogate_tensor_name_is_refused(self) -> None:
        """R-1: a header name that cannot encode as UTF-8 is malformed."""
        data = safetensors_bytes(
            [],
            entries={"\ud800": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}},
        )
        with pytest.raises(MalformedWeightsError, match="not valid Unicode"):
            _parse(data + b"\x00")

    def test_duplicate_tensor_keys_are_refused(self) -> None:
        """M-J4-11: a repeated member name in the header is refused."""
        entry = '{"dtype":"U8","shape":[1],"data_offsets":[0,1]}'
        raw = ('{"w":' + entry + ',"w":' + entry + "}").encode()
        with pytest.raises(MalformedWeightsError, match="appears twice"):
            _parse(struct.pack("<Q", len(raw)) + raw + b"\x01")

    def test_sixteen_bytes_of_slack_are_refused(self) -> None:
        """M-J4-11: bytes after the last tensor break the exact partition."""
        with pytest.raises(MalformedWeightsError, match="bytes after its last tensor"):
            _parse(safetensors_bytes([TENSOR_A, TENSOR_B], trailing=bytes(16)))

    @pytest.mark.parametrize(
        ("metadata", "entries", "message"),
        [
            ({"n": 1}, {}, "__metadata__"),
            (
                None,
                {"": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]}},
                "empty name",
            ),
            (None, {"w": {"dtype": "U8", "shape": [0]}}, "needs exactly"),
            (None, {"w": "U8"}, "needs exactly"),
            (
                None,
                {"w": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 0]}},
                "non-negative",
            ),
            (
                None,
                {"w": {"dtype": "F32", "shape": [True], "data_offsets": [0, 4]}},
                "non-negative",
            ),
            (
                None,
                {"w": {"dtype": "F32", "shape": "1", "data_offsets": [0, 4]}},
                "non-negative",
            ),
            (
                None,
                {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0]}},
                "invalid data_offsets",
            ),
            (
                None,
                {"w": {"dtype": "F32", "shape": [1], "data_offsets": "0-4"}},
                "invalid data_offsets",
            ),
            (
                None,
                {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0.0, 4]}},
                "invalid data_offsets",
            ),
            (
                None,
                {"w": {"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}},
                "invalid data_offsets",
            ),
            (
                None,
                {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 99]}},
                "invalid data_offsets",
            ),
            (
                None,
                {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
                "does not match its shape",
            ),
        ],
    )
    def test_entry_refusals(
        self,
        metadata: dict[str, object] | None,
        entries: dict[str, object],
        message: str,
    ) -> None:
        """Each malformed header entry is refused by name."""
        data = safetensors_bytes([], metadata, entries=entries) + bytes(4)  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it
        with pytest.raises(MalformedWeightsError, match=message):
            _parse(data)

    @pytest.mark.parametrize("dtype", ["F4", "f32", 7])
    def test_unlisted_dtypes_are_refused_by_name(self, dtype: object) -> None:
        """A dtype outside the closed set has its own error."""
        data = (
            safetensors_bytes(
                [],
                entries={"w": {"dtype": dtype, "shape": [1], "data_offsets": [0, 1]}},
            )
            + b"\x00"
        )
        with pytest.raises(UnsupportedSafetensorsDtypeError):
            _parse(data)

    def test_overlapping_and_gapped_slices_are_refused(self) -> None:
        """Slices must tile the data section from offset zero."""
        overlap: dict[str, object] = {
            "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
            "b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]},
        }
        gap: dict[str, object] = {
            "a": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}
        }
        for entries, size in ((overlap, 3), (gap, 2)):
            with pytest.raises(MalformedWeightsError, match="overlap or leave a gap"):
                _parse(safetensors_bytes([], entries=entries) + bytes(size))

    def test_zero_length_tensors_tile_at_one_offset(self) -> None:
        """Empty tensors share an offset without breaking the partition."""
        entries: dict[str, object] = {
            "empty": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]},
            "u": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
        }
        found = _parse(safetensors_bytes([], entries=entries) + b"\x07")
        assert sorted(item.name for item in found) == ["empty", "u"]

    def test_a_file_shorter_than_its_stated_length_is_refused(self) -> None:
        """Tensor bytes missing at read time give a truncation refusal."""
        data = safetensors_bytes([TENSOR_A, TENSOR_B])
        with pytest.raises(MalformedWeightsError, match="truncated"):
            _parse(data[:-8], length=len(data))

    def test_large_tensors_are_streamed_in_chunks(self, tmp_path: Path) -> None:
        """A tensor above one chunk hashes the same as in memory."""
        payload = bytes(range(256)) * 8193
        data = safetensors_bytes([("big", "U8", [len(payload)], payload)])
        (found,) = _parse(data)
        assert found.data_sha256 == hashlib.sha256(payload).hexdigest()
        path = _write(tmp_path / "big.safetensors", data)
        assert compute_tensor_set_identity([path]).total_byte_length == len(payload)


def _archive(tmp_path: Path, data: bytes) -> Path:
    """Write an archive and return its path."""
    return _write(tmp_path / "weights.tar", data)


class TestWeightArchives:
    """The ``safetensors_tar`` framing rules and shard invariance."""

    def test_two_shard_archive_matches_the_single_file(self, tmp_path: Path) -> None:
        """An archive of reversed shards gives the single file's digest."""
        data = ustar_archive(
            [
                (b"shard_2.safetensors", safetensors_bytes([TENSOR_B])),
                (b"config.json", b"{}"),
                (b"shard_1.safetensors", safetensors_bytes([TENSOR_A], {"x": "y"})),
                (b"tokenizer.json", b"{}"),
            ]
        )
        identity = compute_archive_tensor_set_identity(_archive(tmp_path, data))
        assert identity.tensor_set_digest == FIXTURE_DIGEST

    def test_an_archive_written_by_tarfile_is_accepted(self, tmp_path: Path) -> None:
        """The standard library's POSIX ustar writer meets the rules."""
        path = tmp_path / "std.tar"
        with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
            payload = safetensors_bytes([TENSOR_A, TENSOR_B])
            info = tarfile.TarInfo("model.safetensors")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        assert (
            compute_archive_tensor_set_identity(path).tensor_set_digest
            == FIXTURE_DIGEST
        )

    def test_member_listing_reports_offsets(self) -> None:
        """Members are listed in order with their data offsets."""
        data = ustar_archive([(b"a.safetensors", b"x" * 600)])
        (member,) = list_archive_members(io.BytesIO(data), len(data))
        assert (member.name, member.data_offset, member.size) == (
            "a.safetensors",
            512,
            600,
        )

    @pytest.mark.parametrize(
        ("header", "message"),
        [
            (ustar_header(b"a.safetensors", 0, magic=b"ustar "), "not POSIX ustar"),
            (ustar_header(b"a.safetensors", 0, version=b" \x00"), "not POSIX ustar"),
            (
                ustar_header(b"a.safetensors", 0, checksum_delta=1),
                "checksum does not match",
            ),
            (ustar_header(b"a.safetensors", 0, typeflag=b"x"), "not a regular file"),
            (ustar_header(b"a.safetensors", 0, typeflag=b"g"), "not a regular file"),
            (ustar_header(b"a.safetensors", 0, typeflag=b"L"), "not a regular file"),
            (ustar_header(b"a.safetensors", 0, typeflag=b"5"), "not a regular file"),
            (ustar_header(b"a.safetensors", 0, prefix=b"dir"), "prefix field"),
            (
                ustar_header(b"a.safetensors", 0, size_field=b"12x\x00"),
                "size field is not octal",
            ),
            (ustar_header(b"dir/a.safetensors", 0), "is not allowed"),
            (ustar_header(b"..", 0), "is not allowed"),
            (ustar_header(b".a.safetensors", 0), "is not allowed"),
            (ustar_header(b"", 0), "is not allowed"),
            (ustar_header(b"\xff.safetensors", 0), "not UTF-8"),
            (ustar_header(b"readme.txt", 0), "not a weight shard"),
            (ustar_header(b"a.safetensors", 4096), "is truncated"),
        ],
        ids=_case_id,
    )
    def test_header_refusals(self, header: bytes, message: str) -> None:
        """Each header rule refuses its violation."""
        data = header + bytes(1024)
        with pytest.raises(MalformedWeightsError, match=message):
            list_archive_members(io.BytesIO(data), len(data))

    def test_a_bad_checksum_field_is_refused(self) -> None:
        """A checksum field that is not octal is malformed."""
        header = bytearray(ustar_header(b"a.safetensors", 0))
        header[148:156] = b"zzzzzz\x00 "
        data = bytes(header) + bytes(1024)
        with pytest.raises(MalformedWeightsError, match="checksum field is not octal"):
            list_archive_members(io.BytesIO(data), len(data))

    @pytest.mark.parametrize(
        ("data", "message"),
        [
            (ustar_archive([(b"a.safetensors", b"")])[:-1], "multiple of 512"),
            (
                ustar_archive([(b"a.safetensors", b"")], end_region=0),
                "no end-of-archive",
            ),
            (ustar_archive([(b"a.safetensors", b"")], end_region=512), "wrong length"),
            (
                ustar_archive([(b"a.safetensors", b"")], end_region=10_752),
                "wrong length",
            ),
            (
                ustar_archive([(b"a.safetensors", b"")]) + b"\x01" + bytes(511),
                "data after its end",
            ),
            (
                ustar_archive([(b"a.safetensors", b""), (b"a.safetensors", b"")]),
                "repeats member",
            ),
            (ustar_archive([(b"config.json", b"{}")]), "no safetensors shard"),
        ],
        ids=_case_id,
    )
    def test_archive_refusals(self, data: bytes, message: str) -> None:
        """Length, end blocks, repeats and the shard requirement are enforced."""
        with pytest.raises(MalformedWeightsError, match=message):
            list_archive_members(io.BytesIO(data), len(data))

    def test_non_zero_member_padding_is_refused(self) -> None:
        """R-2: the bytes between a member's data and its boundary are zero."""
        data = (
            ustar_header(b"a.safetensors", 1)
            + b"x"
            + b"\x01"
            + bytes(510)
            + bytes(1024)
        )
        with pytest.raises(MalformedWeightsError, match="non-zero padding"):
            list_archive_members(io.BytesIO(data), len(data))

    def test_a_tensor_name_in_two_shards_is_refused(self, tmp_path: Path) -> None:
        """Tensor names are unique across the archive's shards."""
        shard = safetensors_bytes([TENSOR_A])
        data = ustar_archive([(b"1.safetensors", shard), (b"2.safetensors", shard)])
        with pytest.raises(MalformedWeightsError, match="unique across shards"):
            compute_archive_tensor_set_identity(_archive(tmp_path, data))

    def test_a_malformed_shard_inside_the_archive_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Shards are parsed with the hardened reader."""
        data = ustar_archive(
            [(b"1.safetensors", safetensors_bytes([TENSOR_A], trailing=b"\x00"))]
        )
        with pytest.raises(MalformedWeightsError, match="bytes after its last tensor"):
            compute_archive_tensor_set_identity(_archive(tmp_path, data))
