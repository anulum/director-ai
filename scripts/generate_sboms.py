# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — per-extra SBOM generator
"""Generate a CycloneDX SBOM per installable extra from ``pyproject.toml``.

CI already emits one resolved-environment SBOM for the whole install. This
script emits a *declared-dependency* SBOM for each shippable surface — the core
install and each extra a customer can select — so a procurement or security
review can see exactly which third-party components a given extra pulls in,
without resolving a full environment. Output is deterministic (sorted, no
network), one ``sbom/director-ai-<target>.cdx.json`` per target.

Usage::

    python -m scripts.generate_sboms                 # audited target set
    python -m scripts.generate_sboms server nli      # specific extras
    python -m scripts.generate_sboms --all-extras    # one per declared extra
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

# The shippable surfaces the security audit tracks. "core" is the base install;
# "training" maps to the "train" extra; "all-extras" is the full union.
_AUDITED_TARGETS = (
    "core",
    "server",
    "nli",
    "enterprise",
    "ingestion",
    "voice",
    "physical",
    "training",
    "all-extras",
)
_TARGET_TO_EXTRA = {"training": "train"}

_ROOT = Path(__file__).resolve().parent.parent
_SBOM_SCHEMA = "http://cyclonedx.org/schema/bom-1.5.schema.json"


def _load_pyproject() -> dict:
    with open(_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


def _component(requirement: str) -> dict:
    """Map a PEP 508 requirement string to a CycloneDX library component."""
    req = Requirement(requirement)
    version = str(req.specifier) if req.specifier else ""
    component: dict[str, object] = {
        "type": "library",
        "name": req.name,
        "purl": f"pkg:pypi/{req.name}",
        "bom-ref": f"pkg:pypi/{req.name}",
    }
    if version:
        component["version"] = version
    if req.extras:
        component["properties"] = [
            {"name": "pypi:extras", "value": ",".join(sorted(req.extras))}
        ]
    return component


def _requirements_for(target: str, project: dict) -> list[str]:
    base = list(project.get("dependencies", []))
    extras = project.get("optional-dependencies", {})
    if target == "core":
        return base
    if target == "all-extras":
        merged = list(base)
        for deps in extras.values():
            merged.extend(deps)
        return merged
    extra = _TARGET_TO_EXTRA.get(target, target)
    if extra not in extras:
        raise SystemExit(f"unknown extra {extra!r}; choose from {sorted(extras)}")
    return base + list(extras[extra])


def build_sbom(target: str, project: dict) -> dict:
    """Build a deterministic CycloneDX 1.5 SBOM dict for one target."""
    name = project["name"]
    version = project["version"]
    requirements = _requirements_for(target, project)
    # Deduplicate by requirement string, then sort by component name for stable
    # output across runs.
    components = [_component(r) for r in sorted(set(requirements))]
    components.sort(key=lambda c: (str(c["name"]).lower(), str(c.get("version", ""))))
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "$schema": _SBOM_SCHEMA,
        "metadata": {
            "component": {
                "type": "application",
                "name": name,
                "version": version,
                "bom-ref": f"pkg:pypi/{name}@{version}",
                "purl": f"pkg:pypi/{name}@{version}",
            },
            "properties": [
                {"name": "director:install_target", "value": target},
                {"name": "director:sbom_kind", "value": "declared-dependencies"},
            ],
        },
        "components": components,
    }


def write_sboms(targets: list[str], out_dir: Path) -> list[Path]:
    project = _load_pyproject()["project"]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for target in targets:
        sbom = build_sbom(target, project)
        path = out_dir / f"director-ai-{target}.cdx.json"
        path.write_text(json.dumps(sbom, indent=2) + "\n", encoding="utf-8")
        written.append(path)
        print(f"wrote {path} ({len(sbom['components'])} components)")
    return written


def main(argv: list[str]) -> None:
    out_dir = _ROOT / "sbom"
    if argv == ["--all-extras"]:
        project = _load_pyproject()["project"]
        targets = [
            "core",
            *sorted(project.get("optional-dependencies", {})),
            "all-extras",
        ]
    elif argv:
        targets = argv
    else:
        targets = list(_AUDITED_TARGETS)
    write_sboms(targets, out_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
