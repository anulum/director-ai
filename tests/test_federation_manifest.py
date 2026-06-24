# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the schema-A studio capability manifest producer.

Covers the built manifest's verb set and honest safety tiers, the full schema-A
key surface, deterministic content-digest semantics (stable, sha256-prefixed,
excludes the environment-dependent version, moves when the contract changes),
sorted/de-duplicated evidence types, the verb and ui_module renderings, and the
not-installed version fallback.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from director_ai.federation import StudioManifest, Verb, build_manifest
from director_ai.federation import manifest as manifest_mod

_SCHEMA_A_KEYS = {
    "contract_era",
    "protocol_version",
    "transport_profile",
    "studio",
    "studio_version",
    "platform_sdk",
    "enumeration",
    "evidence_types",
    "verbs",
    "ui_module",
    "content_digest",
}


def test_build_manifest_verbs_and_panel() -> None:
    manifest = build_manifest()
    verbs = {v.verb for v in manifest.verbs}
    assert verbs == {
        "score",
        "validate",
        "halt",
        "calibrate",
        "detect-injection",
        "benchmark",
        "replay",
        "redact",
    }
    assert manifest.ui_module.exposes == ("./DirectorAIStudioPanel",)
    assert manifest.ui_module.federation == "module-federation-2"


def test_honest_safety_tiers() -> None:
    """Response scoring/verification are production; the opt-in halt is research."""
    tiers = {v.verb: v.safety_tier for v in build_manifest().verbs}
    assert tiers["score"] == "production"
    assert tiers["validate"] == "production"
    assert tiers["halt"] == "research"


def test_to_dict_schema_a_surface() -> None:
    payload = build_manifest().to_dict()
    assert set(payload) == _SCHEMA_A_KEYS
    assert payload["contract_era"] == "v1"
    assert payload["protocol_version"] == "1"
    assert payload["transport_profile"] == "local-first"
    assert payload["studio"] == "director-ai"
    assert payload["platform_sdk"] == ">=0.1,<0.2"
    assert payload["enumeration"] == "language-agnostic"
    assert isinstance(payload["verbs"], list)
    assert payload["ui_module"]["remote_entry"] == "/studio/remoteEntry.js"


def test_evidence_types_sorted_and_deduplicated() -> None:
    manifest = build_manifest()
    et = manifest.evidence_types
    assert list(et) == sorted(et)
    assert len(et) == len(set(et))
    # Every evidence type is produced by some verb, and vice versa.
    produced = {schema for v in manifest.verbs for schema in v.produces}
    assert set(et) == produced


def test_content_digest_is_deterministic_and_prefixed() -> None:
    manifest = build_manifest()
    digest = manifest.content_digest()
    assert digest == build_manifest().content_digest()
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)


def test_content_digest_excludes_studio_version() -> None:
    """A version bump alone must not move the contract digest."""
    base = build_manifest()
    bumped = dataclasses.replace(base, studio_version="99.99.99")
    assert bumped.studio_version == "99.99.99"
    assert bumped.content_digest() == base.content_digest()
    # ...but the rendered version field does change.
    assert bumped.to_dict()["studio_version"] == "99.99.99"


def test_content_digest_moves_when_contract_changes() -> None:
    """Dropping a verb changes the contract, so the digest must move."""
    base = build_manifest()
    fewer = dataclasses.replace(base, verbs=base.verbs[:-1])
    assert fewer.content_digest() != base.content_digest()


def test_verb_to_dict_with_and_without_fidelity() -> None:
    with_fid = Verb(
        verb="score",
        safety_tier="production",
        side_effect="read-only",
        timing_class="interactive",
        produces=("studio.response-score.v1",),
        backends=("python", "rust"),
        fidelity="ml-surrogate",
    ).to_dict()
    assert with_fid["fidelity"] == "ml-surrogate"
    assert with_fid["timing"] == {"class": "interactive"}
    assert with_fid["produces"] == ["studio.response-score.v1"]

    without_fid = Verb(
        verb="halt",
        safety_tier="research",
        side_effect="read-only",
        timing_class="realtime",
        produces=("studio.streaming-halt.v1",),
        backends=("python", "rust"),
    ).to_dict()
    assert "fidelity" not in without_fid


def test_ui_module_render() -> None:
    payload = build_manifest().ui_module.to_dict()
    assert payload == {
        "remote_entry": "/studio/remoteEntry.js",
        "exposes": ["./DirectorAIStudioPanel"],
        "federation": "module-federation-2",
    }


def test_studio_version_from_installed_distribution() -> None:
    """The default version comes from the installed distribution metadata."""
    assert isinstance(build_manifest().studio_version, str)
    assert build_manifest().studio_version != ""


def test_studio_version_fallback_when_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source tree with no installed dist falls back to the sentinel."""

    def _raise(_name: str) -> str:
        raise manifest_mod.PackageNotFoundError

    monkeypatch.setattr(manifest_mod, "version", _raise)
    assert manifest_mod._studio_version() == "0+unknown"


def test_manifest_is_frozen() -> None:
    manifest = build_manifest()
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.studio_version = "x"  # type: ignore[misc]


def test_public_surface_reexports() -> None:
    from director_ai.federation import UiModule

    assert isinstance(build_manifest(), StudioManifest)
    assert UiModule is manifest_mod.UiModule
