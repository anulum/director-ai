#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Offline Licence Key Ceremony Controller
"""Collect safe output paths and invoke the platform-neutral key generator."""

from __future__ import annotations

from pathlib import Path

from generate_license_keypair import main as keygen_main


def _existing_absolute_directory(prompt: str) -> Path:
    raw = input(prompt).strip().strip('"')
    path = Path(raw).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("enter an existing absolute directory")
    return path.resolve(strict=True)


def main() -> int:
    """Keep path parsing out of shell launchers and enforce media separation."""
    try:
        private_dir = _existing_absolute_directory(
            "Absolute folder on the PRIVATE vault medium: "
        )
        public_dir = _existing_absolute_directory(
            "Absolute folder on a separate PUBLIC transfer medium: "
        )
    except (EOFError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    bundle_root = Path(__file__).resolve().parent
    if private_dir == public_dir:
        raise SystemExit("ERROR: private and public outputs require separate media.")
    if private_dir.stat().st_dev == public_dir.stat().st_dev:
        raise SystemExit("ERROR: private and public folders are on the same device.")
    for output_dir in (private_dir, public_dir):
        if output_dir == bundle_root or bundle_root in output_dir.parents:
            raise SystemExit("ERROR: output directories must be outside the bundle.")

    return keygen_main(
        [
            "--private-out",
            str(private_dir / "director_license_private_key.hex"),
            "--public-out",
            str(public_dir / "PUBLIC_KEY_ONLY.txt"),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
