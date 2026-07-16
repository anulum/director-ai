# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — core facade tests: moved-import guidance + PEP 562 lazy pro tier

from __future__ import annotations

import pytest


def test_moved_to_enterprise_attribute_raises_helpful_import_error() -> None:
    import director_ai.core as core

    with pytest.raises(ImportError, match="moved to director_ai.enterprise"):
        _ = core.TenantRouter


def test_unknown_attribute_raises_attribute_error() -> None:
    import director_ai.core as core

    with pytest.raises(AttributeError, match="has no attribute 'NoSuchThing'"):
        _ = core.NoSuchThing


def test_lazy_paid_attribute_resolves_and_caches() -> None:
    import director_ai.core as core

    vars(core).pop("FeedbackStore", None)  # drop any earlier cache
    assert "FeedbackStore" not in vars(core)
    resolved = core.FeedbackStore
    from director_ai.core.calibration.feedback_store import FeedbackStore

    assert resolved is FeedbackStore
    # second access hits the module namespace, not __getattr__
    assert vars(core)["FeedbackStore"] is FeedbackStore


def test_lazy_paid_covers_every_all_name() -> None:
    import director_ai.core as core

    for name in core.__all__:
        assert getattr(core, name) is not None


def test_dir_lists_lazy_names_without_importing_them() -> None:
    import director_ai.core as core

    vars(core).pop("OnlineCalibrator", None)
    listing = dir(core)
    assert "OnlineCalibrator" in listing
    assert "FeedbackStore" in listing
    assert "TenantRouter" not in listing  # moved names are not advertised


def test_missing_paid_submodule_raises_advanced_tier_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import director_ai.core as core

    vars(core).pop("tune", None)  # ensure the lazy path runs

    def _boom(name: str, package: str | None = None) -> None:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(core, "import_module", _boom)
    with pytest.raises(ImportError, match="requires the advanced tier"):
        _ = core.tune
