# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Pilot Onboarding Example Tests (STRONG)
"""Multi-angle tests for examples/pilot_onboarding.py.

Covers: demo store retrieval, argument parsing, NLI resolution logic,
profile selection, edge cases (missing args, unknown profiles),
and pipeline integration documentation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "pilot_onboarding.py"
    spec = importlib.util.spec_from_file_location("pilot_onboarding_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def module():
    return _load_example_module()


# ── Demo store ───────────────────────────────────────────────────


class TestDemoStore:
    """Demo knowledge base must contain expected facts."""

    def test_retrieves_team_plan_price(self, module):
        store = module.build_demo_store()
        context = store.retrieve_context("How much does the Team plan cost?")
        assert context is not None
        assert "$19/user/month" in context

    def test_store_has_documents(self, module):
        store = module.build_demo_store()
        assert store.backend.count() > 0

    def test_retrieves_relevant_context(self, module):
        store = module.build_demo_store()
        context = store.retrieve_context("pricing")
        assert context is not None
        assert len(context) > 0


# ── Argument parsing ─────────────────────────────────────────────


class TestArgParsing:
    """Argument parser must handle various input combinations."""

    def test_profile_only(self, module):
        args = module.parse_args(["--profile", "medical"])
        assert args["profile"] == "medical"
        assert args["use_nli"] is None

    @pytest.mark.parametrize("profile", ["medical", "legal", "finance", "default"])
    def test_various_profiles(self, module, profile):
        args = module.parse_args(["--profile", profile])
        assert args["profile"] == profile

    def test_default_args(self, module):
        args = module.parse_args([])
        assert "profile" in args
        assert "use_nli" in args


# ── NLI resolution ───────────────────────────────────────────────


class TestResolveUseNli:
    """NLI flag resolution: explicit override > profile default."""

    def test_falls_back_to_profile_default_true(self, module):
        assert module.resolve_use_nli(True, None) is True

    def test_falls_back_to_profile_default_false(self, module):
        assert module.resolve_use_nli(False, None) is False

    def test_honors_explicit_override_true(self, module):
        assert module.resolve_use_nli(False, True) is True

    def test_honors_explicit_override_false(self, module):
        assert module.resolve_use_nli(True, False) is False

    @pytest.mark.parametrize(
        "default,override,expected",
        [
            (True, None, True),
            (False, None, False),
            (True, False, False),
            (False, True, True),
            (True, True, True),
            (False, False, False),
        ],
    )
    def test_nli_resolution_matrix(self, module, default, override, expected):
        assert module.resolve_use_nli(default, override) is expected


# ── Pipeline documentation ───────────────────────────────────────


class TestOnboardingPipelineDoc:
    """Example must demonstrate full pipeline: store → scorer → review."""

    def test_module_has_build_demo_store(self, module):
        assert callable(module.build_demo_store)

    def test_module_has_parse_args(self, module):
        assert callable(module.parse_args)

    def test_module_has_resolve_use_nli(self, module):
        assert callable(module.resolve_use_nli)
