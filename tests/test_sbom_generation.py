# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the per-extra SBOM generator.

Covers component mapping, per-target dependency resolution (core / extra /
training alias / all-extras), determinism, the CycloneDX envelope, and the
unknown-extra error.
"""

from __future__ import annotations

import pytest

from scripts.generate_sboms import build_sbom

_PROJECT = {
    "name": "director-ai",
    "version": "9.9.9",
    "dependencies": ["pydantic>=2", "numpy", "httpx>=0.27,<1"],
    "optional-dependencies": {
        "server": ["fastapi>=0.110", "uvicorn[standard]>=0.29"],
        "nli": ["transformers>=4.40"],
        "train": ["torch>=2.8,<3", "datasets"],
    },
}


class TestBuildSbom:
    def test_core_has_only_base_deps(self):
        sbom = build_sbom("core", _PROJECT)
        names = [c["name"] for c in sbom["components"]]
        assert names == sorted(names)
        assert set(names) == {"pydantic", "numpy", "httpx"}

    def test_cyclonedx_envelope(self):
        sbom = build_sbom("core", _PROJECT)
        assert sbom["bomFormat"] == "CycloneDX"
        assert sbom["specVersion"] == "1.5"
        assert sbom["metadata"]["component"]["name"] == "director-ai"
        assert sbom["metadata"]["component"]["version"] == "9.9.9"
        props = {p["name"]: p["value"] for p in sbom["metadata"]["properties"]}
        assert props["director:install_target"] == "core"
        assert props["director:sbom_kind"] == "declared-dependencies"

    def test_component_purl_and_version(self):
        sbom = build_sbom("server", _PROJECT)
        fastapi = next(c for c in sbom["components"] if c["name"] == "fastapi")
        assert fastapi["type"] == "library"
        assert fastapi["purl"] == "pkg:pypi/fastapi"
        assert fastapi["version"] == ">=0.110"

    def test_extra_includes_base_plus_extra(self):
        sbom = build_sbom("server", _PROJECT)
        names = {c["name"] for c in sbom["components"]}
        assert {"pydantic", "numpy", "httpx", "fastapi", "uvicorn"} <= names

    def test_requirement_extras_recorded(self):
        sbom = build_sbom("server", _PROJECT)
        uvicorn = next(c for c in sbom["components"] if c["name"] == "uvicorn")
        props = {p["name"]: p["value"] for p in uvicorn["properties"]}
        assert props["pypi:extras"] == "standard"

    def test_training_alias_maps_to_train(self):
        sbom = build_sbom("training", _PROJECT)
        names = {c["name"] for c in sbom["components"]}
        assert "torch" in names
        assert "datasets" in names

    def test_all_extras_is_union(self):
        sbom = build_sbom("all-extras", _PROJECT)
        names = {c["name"] for c in sbom["components"]}
        assert names == {
            "pydantic",
            "numpy",
            "httpx",
            "fastapi",
            "uvicorn",
            "transformers",
            "torch",
            "datasets",
        }

    def test_unknown_extra_raises(self):
        with pytest.raises(SystemExit, match="unknown extra"):
            build_sbom("does-not-exist", _PROJECT)

    def test_deterministic(self):
        assert build_sbom("server", _PROJECT) == build_sbom("server", _PROJECT)


class TestCommittedSboms:
    def test_audited_sboms_exist_and_are_valid(self):
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        for target in ("core", "server", "nli", "enterprise", "all-extras"):
            path = root / "sbom" / f"director-ai-{target}.cdx.json"
            assert path.exists(), f"missing committed SBOM: {path}"
            sbom = json.loads(path.read_text(encoding="utf-8"))
            assert sbom["bomFormat"] == "CycloneDX"
            props = {p["name"]: p["value"] for p in sbom["metadata"]["properties"]}
            assert props["director:install_target"] == target
