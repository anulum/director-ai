# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Julia lock inventory for OSV-Scanner

"""Export native Julia manifest identities in OSV-Scanner's lockfile format."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping

_SUITESPARSE_WRAPPER_UUID = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.+-]+)?")


def _version(value: object, name: str) -> str:
    """Require a concrete native version for an OSV Julia package query."""
    if not isinstance(value, str) or not _VERSION.fullmatch(value):
        raise ValueError(f"Missing or invalid Julia lock version for {name}")
    return value


def julia_lock_packages(path: Path) -> list[dict[str, dict[str, str]]]:
    """Read the locked Julia packages for an OSV source scan.

    Parameters
    ----------
    path : pathlib.Path
        Complete Julia ``Manifest.toml`` written by Pkg.

    Returns
    -------
    list of dict
        OSV-Scanner package identities. The native unversioned SuiteSparse
        wrapper is represented by the locked SuiteSparse_jll binary package.

    Raises
    ------
    ValueError
        If the manifest omits dependency records or a concrete audit version.
    """
    with path.open("rb") as stream:
        manifest = tomllib.load(stream)
    locked = object_mapping(manifest.get("deps"))
    if not locked:
        raise ValueError("Julia lock has no dependencies")
    _version(manifest.get("julia_version"), "Julia")
    packages: set[tuple[str, str]] = set()
    for name, records in locked.items():
        if not isinstance(records, list) or not records:
            raise ValueError(f"Missing Julia lock records for {name}")
        for raw in records:
            package = object_mapping(raw)
            if name == "SuiteSparse" and "version" not in package:
                if package.get("uuid") != _SUITESPARSE_WRAPPER_UUID:
                    raise ValueError("Unexpected unversioned SuiteSparse identity")
                if {"git-tree-sha1", "path", "repo-url", "repo-rev"} & package.keys():
                    raise ValueError("SuiteSparse wrapper must be bundled with Julia")
                if "SuiteSparse_jll" not in locked:
                    raise ValueError("SuiteSparse wrapper has no locked binary package")
                continue
            packages.add((name, _version(package.get("version"), name)))
    return [
        {"package": {"name": name, "version": version, "ecosystem": "Julia"}}
        for name, version in sorted(packages)
    ]


def main() -> None:
    """Write the selected manifest's custom OSV lockfile to standard output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps({"results": [{"packages": julia_lock_packages(args.manifest)}]}))


if __name__ == "__main__":
    main()
