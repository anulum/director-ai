# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Experimental Namespace Tests

from __future__ import annotations

import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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


@settings(max_examples=80, deadline=None)
@given(name=st.sampled_from(experimental.available_hooks()))
def test_experimental_hook_gate_rejects_every_registered_surface(name):
    with pytest.raises(experimental.ExperimentalFeatureError, match=name):
        experimental.load_hook(name)


@settings(max_examples=80, deadline=None)
@given(
    env_value=st.sampled_from(["1", "true", "TRUE", " yes ", "ON", "off", "0", ""]),
    name=st.sampled_from(experimental.available_hooks()),
)
def test_environment_gate_truth_table_is_explicit(env_value, name):
    old_value = os.environ.get("DIRECTOR_AI_ENABLE_EXPERIMENTAL_HOOKS")
    os.environ["DIRECTOR_AI_ENABLE_EXPERIMENTAL_HOOKS"] = env_value
    enabled = env_value.strip().lower() in {"1", "true", "yes", "on"}
    try:
        assert experimental.experimental_hooks_enabled() is enabled
        if enabled:
            module = experimental.load_hook(name)
            assert module.__name__ == experimental.EXPERIMENTAL_HOOKS[name]
        else:
            with pytest.raises(experimental.ExperimentalFeatureError):
                experimental.load_hook(name)
    finally:
        if old_value is None:
            os.environ.pop("DIRECTOR_AI_ENABLE_EXPERIMENTAL_HOOKS", None)
        else:
            os.environ["DIRECTOR_AI_ENABLE_EXPERIMENTAL_HOOKS"] = old_value
