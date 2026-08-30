#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Offline Licence Ceremony Bundle Verifier
"""Verify the corruption-detection manifest in an offline ceremony bundle."""

from __future__ import annotations

import hashlib
from pathlib import Path


def main() -> int:
    """Verify every manifest entry before the ceremony installs or runs code."""
    root = Path(__file__).resolve().parent
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise SystemExit("ERROR: MANIFEST.sha256 is missing.")

    checked = 0
    expected_paths: set[Path] = set()
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line:
            continue
        try:
            expected, relative = raw_line.split("  ", 1)
        except ValueError as exc:
            raise SystemExit(f"ERROR: malformed manifest line {line_number}.") from exc
        candidate = (root / relative).resolve()
        if root not in candidate.parents:
            raise SystemExit(f"ERROR: unsafe manifest path: {relative}")
        if not candidate.is_file():
            raise SystemExit(f"ERROR: bundle file is missing: {relative}")
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"ERROR: checksum mismatch: {relative}")
        expected_paths.add(candidate)
        checked += 1

    if checked == 0:
        raise SystemExit("ERROR: manifest contains no files.")
    actual_paths = {
        path.resolve(): path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    extras = sorted(
        str(actual_paths[path]) for path in actual_paths.keys() - expected_paths
    )
    if extras:
        raise SystemExit(f"ERROR: unmanifested bundle file: {extras[0]}")
    print(f"Bundle integrity check passed ({checked} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
