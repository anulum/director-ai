# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Dataset Fingerprint Tests

"""Multi-angle tests for content-addressed dataset fingerprints."""

from __future__ import annotations

import hashlib

import pytest

from director_ai.core.training.dataset_fingerprint import (
    DatasetFingerprint,
    fingerprint_dataset,
)


class TestLocalFileFingerprints:
    def test_file_digest_matches_reference_sha256(self, tmp_path):
        payload = b'{"premise": "a", "hypothesis": "b", "label": 1}\n'
        dataset = tmp_path / "train.jsonl"
        dataset.write_bytes(payload)

        fingerprint = fingerprint_dataset(str(dataset))

        assert fingerprint.hash_source == "content"
        assert fingerprint.digest == hashlib.sha256(payload).hexdigest()
        assert fingerprint.byte_size == len(payload)
        assert fingerprint.file_count == 1
        assert fingerprint.reason == ""
        assert fingerprint.is_content_addressed is True

    def test_file_uri_scheme_resolves_to_local_path(self, tmp_path):
        dataset = tmp_path / "train.jsonl"
        dataset.write_bytes(b"row\n")

        fingerprint = fingerprint_dataset(f"file://{dataset}")

        assert fingerprint.hash_source == "content"
        assert fingerprint.digest == hashlib.sha256(b"row\n").hexdigest()

    def test_content_change_changes_digest_same_uri(self, tmp_path):
        dataset = tmp_path / "train.jsonl"
        dataset.write_bytes(b"first\n")
        before = fingerprint_dataset(str(dataset))
        dataset.write_bytes(b"second\n")
        after = fingerprint_dataset(str(dataset))

        assert before.uri == after.uri
        assert before.digest != after.digest

    def test_multi_chunk_file_hashes_all_bytes(self, tmp_path):
        payload = b"x" * ((1 << 20) + 7)
        dataset = tmp_path / "big.bin"
        dataset.write_bytes(payload)

        fingerprint = fingerprint_dataset(str(dataset))

        assert fingerprint.digest == hashlib.sha256(payload).hexdigest()
        assert fingerprint.byte_size == len(payload)

    def test_empty_uri_is_rejected(self):
        with pytest.raises(ValueError, match="dataset uri is required"):
            fingerprint_dataset("")


class TestDirectoryFingerprints:
    def test_directory_covers_nested_files_deterministically(self, tmp_path):
        root = tmp_path / "dataset"
        (root / "nested").mkdir(parents=True)
        (root / "a.jsonl").write_bytes(b"alpha\n")
        (root / "nested" / "b.jsonl").write_bytes(b"beta\n")

        first = fingerprint_dataset(str(root))
        second = fingerprint_dataset(str(root))

        assert first.hash_source == "content"
        assert first.digest == second.digest
        assert first.file_count == 2
        assert first.byte_size == len(b"alpha\n") + len(b"beta\n")

    def test_directory_digest_depends_on_relative_paths(self, tmp_path):
        left = tmp_path / "left"
        right = tmp_path / "right"
        left.mkdir()
        right.mkdir()
        (left / "a.jsonl").write_bytes(b"same\n")
        (right / "b.jsonl").write_bytes(b"same\n")

        assert (
            fingerprint_dataset(str(left)).digest
            != fingerprint_dataset(str(right)).digest
        )

    def test_directory_digest_depends_on_file_content(self, tmp_path):
        root = tmp_path / "dataset"
        root.mkdir()
        member = root / "a.jsonl"
        member.write_bytes(b"one\n")
        before = fingerprint_dataset(str(root))
        member.write_bytes(b"two\n")
        after = fingerprint_dataset(str(root))

        assert before.digest != after.digest

    def test_empty_directory_is_rejected(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        with pytest.raises(ValueError, match="dataset directory is empty"):
            fingerprint_dataset(str(root))


class TestFallbacksAreLabelled:
    def test_remote_uri_without_reader_is_uri_only(self):
        uri = "gs://director-data/train.jsonl"
        fingerprint = fingerprint_dataset(uri)

        assert fingerprint.hash_source == "uri-only"
        assert fingerprint.reason == "remote-uri-without-reader"
        assert fingerprint.digest == hashlib.sha256(uri.encode("utf-8")).hexdigest()
        assert fingerprint.byte_size == 0
        assert fingerprint.file_count == 0
        assert fingerprint.is_content_addressed is False

    def test_remote_uri_with_reader_is_content_addressed(self):
        payload = b'{"premise": "a"}\n'
        seen: list[str] = []

        def reader(uri: str) -> bytes:
            seen.append(uri)
            return payload

        fingerprint = fingerprint_dataset(
            "s3://bucket/train.jsonl",
            remote_reader=reader,
        )

        assert seen == ["s3://bucket/train.jsonl"]
        assert fingerprint.hash_source == "content"
        assert fingerprint.digest == hashlib.sha256(payload).hexdigest()
        assert fingerprint.byte_size == len(payload)
        assert fingerprint.file_count == 1

    def test_missing_local_path_raises_by_default(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="dataset path not found"):
            fingerprint_dataset(str(tmp_path / "absent.jsonl"))

    def test_missing_local_path_degrades_when_not_strict(self, tmp_path):
        uri = str(tmp_path / "absent.jsonl")
        fingerprint = fingerprint_dataset(uri, strict=False)

        assert fingerprint.hash_source == "uri-only"
        assert fingerprint.reason == "missing-local-path"
        assert fingerprint.digest == hashlib.sha256(uri.encode("utf-8")).hexdigest()

    def test_scheme_without_netloc_is_treated_as_local(self, tmp_path):
        fingerprint = fingerprint_dataset("gs://", strict=False)

        assert fingerprint.hash_source == "uri-only"
        assert fingerprint.reason == "missing-local-path"


class TestFingerprintSerialisation:
    def test_round_trip_preserves_every_field(self, tmp_path):
        dataset = tmp_path / "train.jsonl"
        dataset.write_bytes(b"row\n")
        fingerprint = fingerprint_dataset(str(dataset))

        rebuilt = DatasetFingerprint.from_dict(fingerprint.to_dict())

        assert rebuilt == fingerprint

    def test_from_dict_defaults_optional_fields(self):
        rebuilt = DatasetFingerprint.from_dict(
            {
                "uri": "gs://bucket/x",
                "digest": "ab" * 32,
                "hash_source": "uri-only",
            }
        )

        assert rebuilt.reason == ""
        assert rebuilt.byte_size == 0
        assert rebuilt.file_count == 0
        assert rebuilt.algorithm == "sha256"

    def test_short_digest_is_sixteen_hex_chars(self, tmp_path):
        dataset = tmp_path / "train.jsonl"
        dataset.write_bytes(b"row\n")
        fingerprint = fingerprint_dataset(str(dataset))

        assert fingerprint.short_digest == fingerprint.digest[:16]
        assert len(fingerprint.short_digest) == 16

    def test_to_dict_is_json_safe_and_complete(self):
        fingerprint = fingerprint_dataset("gs://bucket/train.jsonl")
        payload = fingerprint.to_dict()

        assert sorted(payload) == [
            "algorithm",
            "byte_size",
            "digest",
            "file_count",
            "hash_source",
            "reason",
            "uri",
        ]
        assert payload["algorithm"] == "sha256"
