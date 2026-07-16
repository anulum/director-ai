# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — paid-wheel namespace-safety and contents check

"""Verify a built paid-tier wheel layers cleanly onto the free core.

A paid wheel (``director-ai-pro`` / ``director-ai-full``) ships the BUSL
advanced packages AND the paid single modules into the ``director_ai``
namespace. It must NOT ship the parent ``director_ai/__init__.py`` or
``director_ai/core/__init__.py`` — those come from the free Apache core
wheel, and shipping them again would collide on install. It must contain
every paid single module from ``packages/paid_tier_manifest.json``, the
gRPC stub type files, and — for the ``full`` tier only — the labs packages.

Usage: ``python scripts/check_paid_wheel.py <dist-dir> <pro|full>``
"""

from __future__ import annotations

import glob
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "packages" / "paid_tier_manifest.json"

_PARENT_INIT = ("director_ai/__init__.py", "director_ai/core/__init__.py")


def _labs_prefixes() -> list[str]:
    """Return the labs package globs as wheel path prefixes."""
    labs = json.loads(MANIFEST.read_text(encoding="utf-8"))["labs_packages"]
    return [g.rstrip("*").replace(".", "/") + "/" for g in labs]


def main(dist_dir: str, tier: str) -> int:
    """Check the single *tier* wheel in *dist_dir*; return an exit code."""
    if tier not in ("pro", "full"):
        print(f"FAIL: unknown tier {tier!r} (expected 'pro' or 'full')")
        return 1
    matches = glob.glob(f"{dist_dir}/*.whl")
    if len(matches) != 1:
        print(f"FAIL: expected exactly one wheel in {dist_dir}, found {matches}")
        return 1
    whl = matches[0]
    names = set(zipfile.ZipFile(whl).namelist())

    failures: list[str] = []
    conflict = [n for n in names if n in _PARENT_INIT]
    if conflict:
        failures.append(f"ships parent __init__ that collides with core: {conflict}")
    if not any(n.startswith("director_ai/core/streaming_repair/") for n in names):
        failures.append("missing the streaming_repair advanced package")

    paid_modules = json.loads(MANIFEST.read_text(encoding="utf-8"))["paid_modules"]
    missing = [m for m in paid_modules if m not in names]
    if missing:
        failures.append(f"missing {len(missing)} paid single modules: {missing[:5]} …")

    if "director_ai/proto/director/v1/director_pb2.pyi" not in names:
        failures.append("missing the gRPC stub type file (director_pb2.pyi)")

    labs_hits = [n for n in names if any(n.startswith(p) for p in _labs_prefixes())]
    if tier == "pro" and labs_hits:
        failures.append(f"pro wheel ships labs files: {labs_hits[:5]} …")
    if tier == "full" and not labs_hits:
        failures.append("full wheel is missing the labs packages")

    if failures:
        for f in failures:
            print(f"FAIL: {whl}: {f}")
        return 1
    print(f"OK: {whl.split('/')[-1]} ({tier}) — {len(names)} files, no conflicts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
