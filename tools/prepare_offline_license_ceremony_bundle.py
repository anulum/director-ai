#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Offline Licence Ceremony Bundle Builder
"""Build a platform-specific, secret-free SEC-1 ceremony ZIP."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PLATFORMS = {
    "windows-x64": "win_amd64",
    "linux-x64": "manylinux_2_17_x86_64",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_exclusive(source: Path, destination: Path) -> None:
    """Copy an archive into place without an overwrite race."""
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with source.open("rb") as input_stream, os.fdopen(fd, "wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        destination.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    """Download target wheels, generate a manifest, and emit one ZIP archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=sorted(PLATFORMS))
    parser.add_argument(
        "--python-version",
        required=True,
        help="Target CPython major.minor, for example 3.11.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        major, minor = (int(part) for part in args.python_version.split(".", 1))
    except ValueError as exc:
        parser.error("--python-version must look like 3.11")
        raise AssertionError from exc
    if major != 3 or minor < 11:
        parser.error("Director-AI requires Python 3.11 or newer")
    if not args.output_dir.is_dir():
        parser.error("--output-dir must already exist")

    tools_root = Path(__file__).resolve().parent
    template_root = tools_root / "offline_license_ceremony"
    bundle_name = f"director-ai-sec1-{args.target}-py{major}{minor}"
    archive = args.output_dir.resolve() / f"{bundle_name}.zip"
    with tempfile.TemporaryDirectory(prefix="director-sec1-") as temporary:
        stage = Path(temporary) / bundle_name
        shutil.copytree(template_root, stage)
        shutil.copy2(tools_root / "generate_license_keypair.py", stage)
        wheelhouse = stage / "wheelhouse"
        wheelhouse.mkdir()

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                "--require-hashes",
                "--dest",
                str(wheelhouse),
                "--platform",
                PLATFORMS[args.target],
                "--python-version",
                f"{major}{minor}",
                "--implementation",
                "cp",
                "--abi",
                f"cp{major}{minor}",
                "--abi",
                "abi3",
                "-r",
                str(stage / "requirements-offline.txt"),
            ],
            check=True,
        )

        manifest_lines = []
        for path in sorted(stage.rglob("*")):
            if path.is_file() and path.name != "MANIFEST.sha256":
                relative = path.relative_to(stage).as_posix()
                manifest_lines.append(f"{_digest(path)}  {relative}\n")
        (stage / "MANIFEST.sha256").write_text(
            "".join(manifest_lines), encoding="utf-8", newline="\n"
        )

        temporary_archive = Path(
            shutil.make_archive(str(Path(temporary) / bundle_name), "zip", stage)
        )
        try:
            _copy_exclusive(temporary_archive, archive)
        except FileExistsError:
            parser.error(f"refusing to overwrite existing archive: {archive}")

    print(f"Created secret-free ceremony bundle: {archive}")
    print(f"SHA-256: {_digest(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
