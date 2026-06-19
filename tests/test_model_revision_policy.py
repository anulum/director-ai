# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import pytest


def test_known_remote_model_resolves_pinned_revision() -> None:
    from director_ai.core.model_revisions import resolve_model_revision

    assert (
        resolve_model_revision("distilbert-base-uncased")
        == "12040accade4e8a0f71eabdb258fecc2e7e948be"
    )


def test_training_baseline_resolves_pinned_revision() -> None:
    from director_ai.core.model_revisions import resolve_model_revision

    assert (
        resolve_model_revision("microsoft/deberta-v3-base")
        == "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
    )


def test_prompt_guard_model_resolves_pinned_revision() -> None:
    from director_ai.core.config import DirectorConfig
    from director_ai.core.model_revisions import resolve_model_revision

    # The default model-backed prompt guard must resolve to an immutable pin so a
    # moving upstream branch cannot silently swap the security classifier.
    assert (
        resolve_model_revision(DirectorConfig().prompt_guard_model_id)
        == "e6535ca4ce3ba852083e75ec585d7c8aeb4be4c5"
    )


def test_explicit_revision_overrides_registry_pin() -> None:
    from director_ai.core.model_revisions import resolve_model_revision

    assert (
        resolve_model_revision(
            "distilbert-base-uncased",
            revision="verified-local-policy",
        )
        == "verified-local-policy"
    )


def test_local_existing_path_does_not_require_revision(tmp_path) -> None:
    from director_ai.core.model_revisions import resolve_model_revision

    model_dir = tmp_path / "exported-model"
    model_dir.mkdir()

    assert resolve_model_revision(str(model_dir)) is None


def test_blank_model_name_is_not_a_remote_reference() -> None:
    from director_ai.core.model_revisions import resolve_model_revision

    assert resolve_model_revision(" ") is None


def test_unpinned_remote_model_requires_explicit_revision() -> None:
    from director_ai.core.model_revisions import resolve_model_revision

    with pytest.raises(ValueError, match="requires an explicit immutable revision"):
        resolve_model_revision("unverified-org/unverified-model")


def test_model_revision_health_reports_unpinned_remote_failure() -> None:
    from director_ai.core.model_revisions import model_revision_health

    report = model_revision_health(
        {
            "nli": ("unverified-org/unverified-model", None),
            "embedding": ("BAAI/bge-large-en-v1.5", "d4aa6901"),
        }
    )

    assert report["ok"] is False
    assert report["checks"]["nli"]["status"] == "error"
    assert "immutable revision" in report["checks"]["nli"]["detail"]
    assert report["checks"]["embedding"]["status"] == "pinned"


def test_model_revision_health_preserves_local_model_paths(tmp_path) -> None:
    from director_ai.core.model_revisions import model_revision_health

    model_dir = tmp_path / "local-model"
    model_dir.mkdir()

    report = model_revision_health({"local_judge": (str(model_dir), None)})

    assert report["ok"] is True
    assert report["checks"]["local_judge"]["status"] == "local"


def test_config_health_reports_contradiction_model_when_enabled() -> None:
    from director_ai.core.config import DirectorConfig

    enabled = DirectorConfig(streaming_contradiction_halt=True)
    checks = enabled.model_revision_health()["checks"]
    assert checks["contradiction"]["status"] == "pinned"
    assert checks["contradiction"]["revision"]

    disabled = DirectorConfig(streaming_contradiction_halt=False)
    off_checks = disabled.model_revision_health()["checks"]
    assert off_checks["contradiction"]["status"] == "skipped"
