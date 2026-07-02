#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — STUDIO federation manifest emitter

"""Emit (or check) the Director-AI STUDIO federation manifest artifact.

This is the federation-gate artifact the SCPN-STUDIO keeper and the Director-AI
Tier-B portal consume: an envelope with the original schema-A capability manifest
under ``schema_a`` plus the authored ``architecture-map.v2`` extension under
``architecture_map``. It is distinct from
``docs/_generated/capability_manifest.json`` (the repo-inventory manifest); this
one is the canonical product of
:func:`director_ai.federation.build_federation_document`.

``--check`` fails if the committed artifact has drifted from the producer, so a
verb, evidence-schema, or architecture-map change cannot silently leave a stale
federation manifest behind. ``schema_a.studio_version`` is excluded from the
check (an environment-dependent stamp); ``schema_a.content_digest`` covers the
verb/evidence/ui contract.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import cast

from director_ai.federation import build_federation_document

_ARTIFACT = (
    Path(__file__).resolve().parents[1] / "docs" / "_generated" / "studio_manifest.json"
)


def render() -> str:
    """Return the deterministic STUDIO federation JSON with a trailing newline."""
    payload = build_federation_document()
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _without_env_dependent_fields(payload: dict[str, object]) -> dict[str, object]:
    """Return a comparison copy with environment-dependent fields removed."""
    normalised = copy.deepcopy(payload)
    schema_a = normalised.get("schema_a")
    if isinstance(schema_a, dict):
        schema_a.pop("studio_version", None)
    return normalised


def main(argv: list[str] | None = None) -> int:
    """Emit the artifact, or check the committed copy against the producer.

    Returns ``0`` on success, ``1`` when ``--check`` finds a missing or stale
    artifact (ignoring the environment-dependent ``schema_a.studio_version``
    stamp).
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed artifact differs from the producer (no write).",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=_ARTIFACT,
        help="Manifest artifact path to emit or check.",
    )
    args = parser.parse_args(argv)
    check = cast("bool", args.check)
    artifact = cast("Path", args.artifact)

    rendered = render()
    if check:
        if not artifact.exists():
            print(f"{artifact} is missing; run `python tools/emit_studio_manifest.py`.")
            return 1
        committed = cast(
            "dict[str, object]", json.loads(artifact.read_text(encoding="utf-8"))
        )
        produced = cast("dict[str, object]", json.loads(rendered))
        if _without_env_dependent_fields(committed) != _without_env_dependent_fields(
            produced
        ):
            print(f"{artifact} is stale; run `python tools/emit_studio_manifest.py`.")
            return 1
        return 0

    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(rendered, encoding="utf-8")
    print(f"wrote {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
