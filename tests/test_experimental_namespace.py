# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Experimental Namespace Tests

from __future__ import annotations

import pytest

from director_ai import experimental


@pytest.fixture(autouse=True)
def _reset_experimental_gate(monkeypatch):
    experimental.disable_experimental_hooks()
    monkeypatch.delenv("DIRECTOR_AI_ENABLE_EXPERIMENTAL_HOOKS", raising=False)
    for name in experimental.available_hooks():
        experimental.__dict__.pop(name, None)
    yield
    experimental.disable_experimental_hooks()
    for name in experimental.available_hooks():
        experimental.__dict__.pop(name, None)


def test_available_hooks_lists_research_surface():
    hooks = experimental.available_hooks()

    assert "trajectory" in hooks
    assert "meta_guard" in hooks
    assert "zk_attestation" in hooks
    assert hooks == tuple(sorted(hooks))


def test_load_hook_requires_explicit_gate():
    with pytest.raises(experimental.ExperimentalFeatureError, match="experimental"):
        experimental.load_hook("trajectory")


def test_load_hook_accepts_process_gate():
    experimental.enable_experimental_hooks()

    module = experimental.load_hook("trajectory")

    assert module.__name__ == "director_ai.core.trajectory"
    assert hasattr(module, "TrajectorySimulator")


def test_load_hook_accepts_env_gate(monkeypatch):
    monkeypatch.setenv("DIRECTOR_AI_ENABLE_EXPERIMENTAL_HOOKS", "true")

    module = experimental.load_hook("zk_attestation")

    assert module.__name__ == "director_ai.core.zk_attestation"
    assert hasattr(module, "PassportVerifier")


def test_attribute_access_uses_same_gate():
    with pytest.raises(experimental.ExperimentalFeatureError):
        _ = experimental.trace_safe

    experimental.enable_experimental_hooks()

    assert experimental.trace_safe.__name__ == "director_ai.core.trace_safe"


def test_unknown_hook_is_not_silent():
    with pytest.raises(KeyError):
        experimental.load_hook("missing_hook")

    with pytest.raises(AttributeError):
        _ = experimental.missing_hook
