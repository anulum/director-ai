# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Julia manifest audit inventory tests

"""Exercise the real Julia lock and native OSV-Scanner input contract."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tools.export_julia_lock_for_osv import julia_lock_packages

_ROOT = Path(__file__).resolve().parents[2]
_LOCK = _ROOT / "tools/julia_tuner/Manifest.toml"


def test_export_projects_every_version_from_the_native_lock() -> None:
    """Send every locked package version through the maintained CLI."""
    manifest = tomllib.loads(_LOCK.read_text())
    expected = {
        (name, package["version"], "Julia")
        for name, packages in manifest["deps"].items()
        for package in packages
        if "version" in package
    }
    native = subprocess.run(
        [sys.executable, "-m", "tools.export_julia_lock_for_osv", str(_LOCK)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    entries = json.loads(native.stdout)["results"][0]["packages"]
    packages = [entry["package"] for entry in entries]
    assert {
        (package["name"], package["version"], package["ecosystem"])
        for package in packages
    } == expected
    assert len(packages) == len(expected)


@pytest.mark.parametrize(
    ("pattern", "replacement", "message"),
    [
        (r'^julia_version = "[^"\n]+"$', 'julia_version = "unknown"', "for Julia"),
        (r"(?ms)^\[\[deps\..*\Z", "[deps]\n", "no dependencies"),
        (r"^\[\[deps\.ADTypes\]\]$", "[deps.ADTypes]", "records for ADTypes"),
        (
            r'(?m)^uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"$',
            'uuid = "00000000-0000-0000-0000-000000000000"',
            "Unexpected unversioned SuiteSparse identity",
        ),
        (
            r'(?m)^(uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9")$',
            r'\1\npath = "local"',
            "wrapper must be bundled",
        ),
        (
            r"(?ms)^\[\[deps\.SuiteSparse_jll\]\]\n.*?(?=^\[\[deps\.)",
            "",
            "wrapper has no locked binary",
        ),
        (
            r'(?ms)(^\[\[deps\.ADTypes\]\]\n.*?)^version = "[^"\n]+"\n',
            r"\1",
            "lock version for ADTypes",
        ),
    ],
)
def test_export_refuses_incomplete_native_lock(
    tmp_path: Path, pattern: str, replacement: str, message: str
) -> None:
    """Reject exact corruptions of the real manifest before scanning packages."""
    amended, changes = re.subn(
        pattern, replacement, _LOCK.read_text(), count=1, flags=re.MULTILINE
    )
    assert changes == 1
    invalid = tmp_path / "Manifest.toml"
    invalid.write_text(amended)
    with pytest.raises(ValueError, match=message):
        julia_lock_packages(invalid)
