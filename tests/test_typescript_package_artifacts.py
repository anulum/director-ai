# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — actual npm package members, framing and custody tests
"""Compare genuine native npm bytes and exercise real archive corruption/refusal."""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import stat
import tarfile
import time
from pathlib import Path

import pytest

from tests.test_typescript_generated_recipe import SDKCase
from tests.test_typescript_generated_recipe import sdk_case as sdk_case
from tools.typescript_package_artifacts import inspect_npm_package


@pytest.fixture
def expected_members(sdk_case: SDKCase) -> dict[str, tuple[str, int]]:
    """Derive the expected public payload from real source and fresh native output.

    Parameters
    ----------
    sdk_case : SDKCase
        Actual SDK source, compiler/renderer outputs and original npm pack result.

    Returns
    -------
    dict[str, tuple[str, int]]
        Six public npm member digests and their actual npm permission projection.
    """
    paths = (
        "package.json",
        "README.md",
        "LICENSE",
        "LICENSE.license",
        "dist/index.js",
        "dist/index.d.ts",
    )
    return {
        "package/" + name: (
            hashlib.sha256((sdk_case.sdk / name).read_bytes()).hexdigest(),
            stat.S_IMODE((sdk_case.sdk / name).stat().st_mode) & 0o755,
        )
        for name in paths
    }


def test_actual_npm_prepack_payload_matches_source_and_native_compiler(
    sdk_case: SDKCase, expected_members: dict[str, tuple[str, int]]
) -> None:
    """Accept original npm bytes only when every public source/native member agrees.

    Parameters
    ----------
    sdk_case : SDKCase
        Actual producer fixture with unmodified prepack/build hooks.
    expected_members : dict[str, tuple[str, int]]
        Expectations independently derived from source and native compiler bytes.
    """
    result = inspect_npm_package(
        sdk_case.archive, expected_members, deadline=time.monotonic() + 10
    )
    assert (
        result.archive_sha256
        == hashlib.sha256(sdk_case.archive.read_bytes()).hexdigest()
    )
    assert result.members == expected_members
    assert len(result.members) == 6
    assert {mode for _, mode in result.members.values()} <= {0o600, 0o644, 0o755}


@pytest.mark.parametrize(
    "fault", ["bytes", "mode", "missing", "duplicate", "extra", "symlink"]
)
def test_actual_native_member_mutations_refuse(
    tmp_path: Path,
    sdk_case: SDKCase,
    expected_members: dict[str, tuple[str, int]],
    fault: str,
) -> None:
    """Reject real changed, missing, duplicate, additional and nonregular members.

    Parameters
    ----------
    tmp_path : Path
        Actual private corrupt-archive output directory.
    sdk_case : SDKCase
        Original native package, never modified by a negative control.
    expected_members : dict[str, tuple[str, int]]
        Original source/compiled payload identities.
    fault : str
        Real member-level corruption applied to a derived test archive.
    """
    output = tmp_path / "member-fault.tgz"
    with (
        tarfile.open(sdk_case.archive, "r:gz") as source,
        tarfile.open(output, "w:gz") as target,
    ):
        for index, member in enumerate(source):
            content = source.extractfile(member)
            assert content is not None
            with content:
                data = content.read()
            if index == 0:
                if fault == "missing":
                    continue
                if fault == "bytes":
                    data += b"changed original bytes"
                    member.size = len(data)
                elif fault == "mode":
                    member.mode ^= 0o111
                elif fault == "symlink":
                    member.type = tarfile.SYMTYPE
                    member.linkname = "/outside-package"
                    member.size = 0
                target.addfile(member, io.BytesIO(data))
                if fault == "duplicate":
                    target.addfile(member, io.BytesIO(data))
                elif fault == "extra":
                    extra = tarfile.TarInfo("package/unowned.txt")
                    extra.size = 6
                    target.addfile(extra, io.BytesIO(b"unowned"[:6]))
            else:
                target.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError, match="member|inventory"):
        inspect_npm_package(output, expected_members, deadline=time.monotonic() + 10)


@pytest.mark.parametrize(
    "fault",
    ["not-gzip", "gzip-footer", "tar-checksum", "end-marker", "trailing-payload"],
)
def test_real_compressed_and_tar_framing_refusals(
    tmp_path: Path,
    sdk_case: SDKCase,
    expected_members: dict[str, tuple[str, int]],
    fault: str,
) -> None:
    """Read through actual gzip/tar framing and refuse truncation or hidden payload.

    Parameters
    ----------
    tmp_path : Path
        Runner-owned derived archive location.
    sdk_case : SDKCase
        Genuine original compressed archive.
    expected_members : dict[str, tuple[str, int]]
        Independently derived real native payload contract.
    fault : str
        Actual gzip or tar framing fault, without mocked decoder output.
    """
    original = sdk_case.archive.read_bytes()
    raw = gzip.decompress(original)
    if fault == "not-gzip":
        changed = b"not an archive"
    elif fault == "gzip-footer":
        changed = original[:-4]
    elif fault == "tar-checksum":
        changed = gzip.compress(b"X" + raw[1:])
    elif fault == "end-marker":
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as package:
            last = package.getmembers()[-1]
        end = last.offset_data + (last.size + 511) // 512 * 512
        changed = gzip.compress(raw[: end + 512])
    else:
        changed = gzip.compress(raw + b"unparsed actual bytes")
    archive = tmp_path / "framing.tgz"
    archive.write_bytes(changed)
    with pytest.raises(ValueError, match="complete|end markers|trailing"):
        inspect_npm_package(archive, expected_members, deadline=time.monotonic() + 10)


@pytest.mark.parametrize("deadline", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_archive_deadlines_refuse(
    sdk_case: SDKCase, expected_members: dict[str, tuple[str, int]], deadline: float
) -> None:
    """Refuse invalid shared budgets before opening an original archive.

    Parameters
    ----------
    sdk_case : SDKCase
        Genuine archive input.
    expected_members : dict[str, tuple[str, int]]
        Source/compiled byte contract.
    deadline : float
        Invalid nonpositive or nonfinite monotonic deadline.
    """
    with pytest.raises(ValueError, match="positive and finite"):
        inspect_npm_package(sdk_case.archive, expected_members, deadline=deadline)


@pytest.mark.parametrize(
    "fault",
    [
        "expired",
        "missing-manifest",
        "outside-package",
        "traversal",
        "digest",
        "negative-mode",
        "large-mode",
        "boolean-mode",
        "symlink",
        "fifo",
    ],
)
def test_public_input_contract_refusals(
    tmp_path: Path,
    sdk_case: SDKCase,
    expected_members: dict[str, tuple[str, int]],
    fault: str,
) -> None:
    """Reject actual invalid input paths, complete manifests and exhausted budgets.

    Parameters
    ----------
    tmp_path : Path
        Real local link/FIFO fixtures; no external file is followed.
    sdk_case : SDKCase
        Original native archive.
    expected_members : dict[str, tuple[str, int]]
        Mutable copy of independently derived source/native expectations.
    fault : str
        Input-boundary mutation to the real public API.
    """
    archive = sdk_case.archive
    deadline = time.monotonic() + 10
    if fault == "expired":
        deadline = time.monotonic() - 1
    elif fault == "missing-manifest":
        expected_members.pop("package/README.md")
    elif fault == "outside-package":
        expected_members["outside/file"] = ("a" * 64, 0o600)
    elif fault == "traversal":
        expected_members["package/../outside"] = ("a" * 64, 0o600)
    elif fault == "digest":
        expected_members["package/README.md"] = ("not-sha256", 0o600)
    elif fault == "negative-mode":
        expected_members["package/README.md"] = ("a" * 64, -1)
    elif fault == "large-mode":
        expected_members["package/README.md"] = ("a" * 64, 0o1000)
    elif fault == "boolean-mode":
        expected_members["package/README.md"] = ("a" * 64, True)
    elif fault == "symlink":
        archive = tmp_path / "alias.tgz"
        archive.symlink_to(sdk_case.archive)
    else:
        archive = tmp_path / "archive-pipe"
        os.mkfifo(archive)
    with pytest.raises(ValueError):
        inspect_npm_package(archive, expected_members, deadline=deadline)


@pytest.mark.parametrize(
    "phase", ["before-open", "after-read-bytes", "after-read-inode"]
)
def test_real_archive_custody_interleavings_refuse(
    tmp_path: Path,
    sdk_case: SDKCase,
    expected_members: dict[str, tuple[str, int]],
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    """Refuse real inode replacement or byte changes at capture/read boundaries.

    Parameters
    ----------
    tmp_path : Path
        Runner-owned archive copy for genuine filesystem races.
    sdk_case : SDKCase
        Unmodified native original supplying starting bytes.
    expected_members : dict[str, tuple[str, int]]
        Source/compiler payload identities.
    monkeypatch : pytest.MonkeyPatch
        Negative-only boundary interleaving; real open/read/close always execute.
    phase : str
        Real mutation before held capture or after successful actual decoding.
    """
    archive = tmp_path / "custody.tgz"
    original = sdk_case.archive.read_bytes()
    archive.write_bytes(original)
    changed = False
    open_original = os.open
    close_original = gzip.GzipFile.close

    def replace() -> None:
        """Replace the real named inode while retaining original bytes elsewhere."""
        nonlocal changed
        archive.rename(tmp_path / "held-original.tgz")
        archive.write_bytes(original)
        changed = True

    def open_interleaving(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        """Perform a genuine replacement immediately before the real native open."""
        assert os.fspath(path) == str(archive)
        monkeypatch.setattr("tools.typescript_package_artifacts.os.open", open_original)
        replace()
        return open_original(path, flags, mode, dir_fd=dir_fd)

    def close_interleaving(decoded: gzip.GzipFile) -> None:
        """Close the actual decoder, then perform the requested real mutation."""
        nonlocal changed
        monkeypatch.setattr(
            "tools.typescript_package_artifacts.gzip.GzipFile.close", close_original
        )
        close_original(decoded)
        if phase == "after-read-bytes":
            with archive.open("ab") as stream:
                stream.write(b"changed after real decode")
            changed = True
        else:
            replace()

    if phase == "before-open":
        monkeypatch.setattr(
            "tools.typescript_package_artifacts.os.open", open_interleaving
        )
        message = "before capture"
    else:
        monkeypatch.setattr(
            "tools.typescript_package_artifacts.gzip.GzipFile.close", close_interleaving
        )
        message = "bytes changed|custody changed"
    with pytest.raises(ValueError, match=message):
        inspect_npm_package(archive, expected_members, deadline=time.monotonic() + 10)
    assert changed
    assert sdk_case.archive.read_bytes() == original
