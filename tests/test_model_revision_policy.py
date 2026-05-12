# SPDX-License-Identifier: AGPL-3.0-or-later
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


def test_unpinned_remote_model_requires_explicit_revision() -> None:
    from director_ai.core.model_revisions import resolve_model_revision

    with pytest.raises(ValueError, match="requires an explicit immutable revision"):
        resolve_model_revision("unverified-org/unverified-model")
