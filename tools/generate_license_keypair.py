#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Generate an Ed25519 license-signing keypair — run OFFLINE, ANULUM only.

The private key mints licences (set it as ``DIRECTOR_LICENSE_PRIVATE_KEY`` when
calling ``generate_license``); it must never leave the signing machine. The
public key is embedded in ``core/license.py`` as ``_LICENSE_ED25519_PUBLIC_KEY_HEX``
so every install can verify signatures without being able to forge them (SEC-1).

    python tools/generate_license_keypair.py --private-out license_private_key.hex

Prints the public key to paste into the source; writes the private key to a
0600 file. Rotating the keypair invalidates every previously issued signed
licence, so treat it as a deliberate, audited operation.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def generate_keypair() -> tuple[str, str]:
    """Return ``(private_hex, public_hex)`` for a fresh Ed25519 keypair."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    private_hex = private_key.private_bytes_raw().hex()
    public_hex = private_key.public_key().public_bytes_raw().hex()
    return private_hex, public_hex


def main(argv: list[str] | None = None) -> int:
    """CLI: write the private key to a 0600 file and print the public key."""
    parser = argparse.ArgumentParser(
        description="Generate an Ed25519 license-signing keypair (offline use).",
    )
    parser.add_argument(
        "--private-out",
        default="director_license_private_key.hex",
        help="File to write the private key to (created with 0600 permissions).",
    )
    args = parser.parse_args(argv)

    private_hex, public_hex = generate_keypair()

    out = Path(args.private_out)
    out.write_text(private_hex + "\n", encoding="utf-8")
    out.chmod(0o600)

    print("Ed25519 license keypair generated.\n")
    print("Public key — paste into core/license.py _LICENSE_ED25519_PUBLIC_KEY_HEX:")
    print(f"  {public_hex}\n")
    print(f"Private key written to {out} (permissions 0600).")
    print(
        "Keep it OFFLINE. To sign licences, export it as DIRECTOR_LICENSE_PRIVATE_KEY "
        "before calling generate_license."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
