# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — paid-wheel namespace-safety check

"""Verify a built paid-tier wheel layers cleanly onto the free core.

A paid wheel (``director-ai-pro`` / ``director-ai-full``) ships the BUSL
advanced/labs packages into the ``director_ai`` namespace. It must NOT ship the
parent ``director_ai/__init__.py`` or ``director_ai/core/__init__.py`` — those
come from the free Apache core wheel, and shipping them again would collide on
install. It must contain the advanced modules it claims to.

Usage: ``python scripts/check_paid_wheel.py <dist-dir>``
"""

from __future__ import annotations

import glob
import sys
import zipfile

_PARENT_INIT = ("director_ai/__init__.py", "director_ai/core/__init__.py")


def main(dist_dir: str) -> int:
    """Check the single wheel in *dist_dir*; return a process exit code."""
    matches = glob.glob(f"{dist_dir}/*.whl")
    if not matches:
        print(f"FAIL: no wheel found in {dist_dir}")
        return 1
    whl = matches[0]
    names = zipfile.ZipFile(whl).namelist()
    conflict = [n for n in names if n in _PARENT_INIT]
    if conflict:
        print(f"FAIL: {whl} ships parent __init__ that collides with core: {conflict}")
        return 1
    if not any(n.startswith("director_ai/core/streaming_repair/") for n in names):
        print(f"FAIL: {whl} is missing the streaming_repair advanced module")
        return 1
    print(f"OK: {whl.split('/')[-1]} — {len(names)} files, no namespace conflict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
