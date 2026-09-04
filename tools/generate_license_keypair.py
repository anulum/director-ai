#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Generate an Ed25519 license-signing keypair — run OFFLINE, ANULUM only.

The private key mints licences (set it as ``DIRECTOR_LICENSE_PRIVATE_KEY`` when
calling ``generate_license``); it must never leave the signing machine. The
public key is embedded in ``core/license.py`` as ``_LICENSE_ED25519_PUBLIC_KEY_HEX``
so every install can verify signatures without being able to forge them (SEC-1).

    python tools/generate_license_keypair.py \
        --private-out /media/KEY_VAULT/director_license_private_key.hex \
        --public-out /media/PUBLIC_TRANSFER/PUBLIC_KEY_ONLY.txt

Prints the public key to paste into the source and can write a separate public
transfer file. The private output is created exclusively with mode 0600 and is
never overwritten. Rotating the keypair invalidates every previously issued
signed licence, so treat it as a deliberate, audited operation.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path


def generate_keypair() -> tuple[str, str]:
    """Return ``(private_hex, public_hex)`` for a fresh Ed25519 keypair."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    private_hex = private_key.private_bytes_raw().hex()
    public_hex = private_key.public_key().public_bytes_raw().hex()
    return private_hex, public_hex


def _absolute_output(value: str) -> Path:
    """Return a normalized absolute output path or reject unsafe ambiguity."""
    output = Path(value).expanduser()
    if not output.is_absolute():
        raise ValueError("output paths must be absolute")
    if not output.parent.is_dir():
        raise ValueError(f"output directory does not exist: {output.parent}")
    return output.parent.resolve(strict=True) / output.name


def _open_exclusive(path: Path, mode: int) -> int:
    """Create *path* once without following an existing final-component link."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, mode)
    try:
        if os.name != "nt":
            os.fchmod(fd, mode)
    except BaseException:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    return fd


def _write_fd(fd: int, content: str) -> None:
    """Write and durably flush UTF-8 text to an already-created descriptor."""
    remaining = memoryview(content.encode("utf-8"))
    while remaining:
        written = os.write(fd, remaining)
        if written == 0:
            raise OSError("zero-byte write while saving key material")
        remaining = remaining[written:]
    os.fsync(fd)
    os.close(fd)


def _close_quietly(fd: int | None) -> None:
    """Close a descriptor during rollback without masking the original error."""
    if fd is None or fd < 0:
        return
    with contextlib.suppress(OSError):
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    """CLI: write the private key to a 0600 file and print the public key."""
    parser = argparse.ArgumentParser(
        description="Generate an Ed25519 license-signing keypair (offline use).",
    )
    parser.add_argument(
        "--private-out",
        required=True,
        help="Absolute private-key path (created once with 0600 permissions).",
    )
    parser.add_argument(
        "--public-out",
        help="Optional absolute path for a public-key-only transfer file.",
    )
    args = parser.parse_args(argv)

    try:
        private_out = _absolute_output(args.private_out)
        public_out = _absolute_output(args.public_out) if args.public_out else None
    except ValueError as exc:
        parser.error(str(exc))

    if public_out is not None and private_out.parent == public_out.parent:
        parser.error("private and public outputs must use separate directories/media")

    private_fd = _open_exclusive(private_out, 0o600)
    public_fd: int | None = None
    public_created = False
    try:
        if public_out is not None:
            public_fd = _open_exclusive(public_out, 0o644)
            public_created = True
        private_hex, public_hex = generate_keypair()
        _write_fd(private_fd, private_hex + "\n")
        private_fd = -1
        if public_fd is not None:
            _write_fd(public_fd, public_hex + "\n")
            public_fd = None
    except BaseException:
        _close_quietly(private_fd)
        _close_quietly(public_fd)
        private_out.unlink(missing_ok=True)
        if public_out is not None and public_created:
            public_out.unlink(missing_ok=True)
        raise

    print("Ed25519 license keypair generated.\n")
    print("Public key — paste into core/license.py _LICENSE_ED25519_PUBLIC_KEY_HEX:")
    print(f"  {public_hex}\n")
    if public_out is not None:
        print(f"Public-key-only transfer file written to {public_out}.")
    print(f"Private key written to {private_out} (permissions 0600 on POSIX).")
    print(
        "Keep it OFFLINE. To sign licences, export it as DIRECTOR_LICENSE_PRIVATE_KEY "
        "before calling generate_license."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
