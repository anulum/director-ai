# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Semantic Kernel Integration Tests (STRONG)
"""Multi-angle tests for Semantic Kernel integration pipeline (STRONG)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.exceptions import HallucinationError
from director_ai.core.types import CoherenceScore
from director_ai.integrations.semantic_kernel import DirectorAIFilter


def _fake_review_pass(self, prompt, action, session=None, tenant_id=""):
    cs = CoherenceScore(score=0.95, approved=True, h_logical=0.02, h_factual=0.03)
    return True, cs


def _fake_review_fail(self, prompt, action, session=None, tenant_id=""):
    cs = CoherenceScore(score=0.15, approved=False, h_logical=0.8, h_factual=0.7)
    return False, cs


@pytest.fixture()
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def _run(coro, loop):
    return loop.run_until_complete(coro)


class TestDirectorAIFilterInit:
    def test_creates_with_facts(self):
        f = DirectorAIFilter(facts={"k": "v"})
        assert isinstance(f.scorer, CoherenceScorer)

    def test_creates_with_store(self):
        store = GroundTruthStore()
        store.add("k", "v")
        f = DirectorAIFilter(store=store)
        assert f.scorer is not None

    def test_store_overrides_facts(self):
        store = GroundTruthStore()
        store.add("data", "Store value.")
        f = DirectorAIFilter(
            facts={"data": "Facts value — should be ignored."},
            store=store,
        )
        assert f.scorer is not None

    def test_custom_threshold(self):
        f = DirectorAIFilter(threshold=0.8)
        assert isinstance(f.scorer, CoherenceScorer)

    def test_raise_on_fail_default(self):
        f = DirectorAIFilter()
        assert f._raise is True

    def test_raise_on_fail_false(self):
        f = DirectorAIFilter(raise_on_fail=False)
        assert f._raise is False

    def test_scorer_property(self):
        f = DirectorAIFilter()
        assert isinstance(f.scorer, CoherenceScorer)


class TestDirectorAIFilterCall:
    @patch.object(CoherenceScorer, "review", _fake_review_pass)
    def test_passes_coherent_response(self, event_loop):
        f = DirectorAIFilter(
            facts={"pricing": "Team plan costs $19/user/month."},
            threshold=0.3,
        )
        context = SimpleNamespace(
            result="Team plan costs $19/user/month.",
            arguments={"input": "What is the pricing?"},
        )
        next_fn = AsyncMock()
        _run(f(context, next_fn), event_loop)
        next_fn.assert_awaited_once_with(context)

    @patch.object(CoherenceScorer, "review", _fake_review_fail)
    def test_raises_on_hallucination(self, event_loop):
        f = DirectorAIFilter(
            facts={"pricing": "Team plan costs $19/user/month."},
            threshold=0.5,
            raise_on_fail=True,
        )
        context = SimpleNamespace(
            result="Quantum teleportation is free.",
            arguments={"input": "Pricing?"},
        )
        next_fn = AsyncMock()
        with pytest.raises(HallucinationError):
            _run(f(context, next_fn), event_loop)

    @patch.object(CoherenceScorer, "review", _fake_review_fail)
    def test_no_raise_returns_dict(self, event_loop):
        f = DirectorAIFilter(
            facts={"pricing": "Team plan costs $19/user/month."},
            threshold=0.5,
            raise_on_fail=False,
        )
        context = SimpleNamespace(
            result="Completely wrong answer.",
            arguments={"input": "Pricing?"},
        )
        next_fn = AsyncMock()
        _run(f(context, next_fn), event_loop)
        assert isinstance(context.result, dict)
        assert context.result["approved"] is False
        assert "score" in context.result
        assert "original" in context.result

    def test_empty_result_skips_scoring(self, event_loop):
        f = DirectorAIFilter(
            facts={"k": "v"},
            threshold=0.99,
            raise_on_fail=True,
        )
        context = SimpleNamespace(result="", arguments={})
        next_fn = AsyncMock()
        _run(f(context, next_fn), event_loop)
        next_fn.assert_awaited_once()

    def test_none_result_skips_scoring(self, event_loop):
        f = DirectorAIFilter(
            facts={"k": "v"},
            threshold=0.99,
            raise_on_fail=True,
        )
        context = SimpleNamespace(result=None, arguments={})
        next_fn = AsyncMock()
        _run(f(context, next_fn), event_loop)
        next_fn.assert_awaited_once()

    @patch.object(CoherenceScorer, "review", _fake_review_pass)
    def test_no_arguments_attribute(self, event_loop):
        f = DirectorAIFilter(
            facts={"pricing": "Team plan costs $19/user/month."},
            threshold=0.3,
        )
        context = SimpleNamespace(result="Team plan costs $19/user/month.")
        next_fn = AsyncMock()
        _run(f(context, next_fn), event_loop)
        next_fn.assert_awaited_once()

    @patch.object(CoherenceScorer, "review", _fake_review_pass)
    def test_empty_arguments(self, event_loop):
        f = DirectorAIFilter(
            facts={"pricing": "Team plan costs $19/user/month."},
            threshold=0.3,
        )
        context = SimpleNamespace(
            result="Team plan costs $19/user/month.",
            arguments={},
        )
        next_fn = AsyncMock()
        _run(f(context, next_fn), event_loop)
        next_fn.assert_awaited_once()

    @patch.object(CoherenceScorer, "review", _fake_review_fail)
    def test_next_fn_always_called_first(self, event_loop):
        f = DirectorAIFilter(
            facts={"k": "v"},
            threshold=0.5,
            raise_on_fail=True,
        )
        call_order = []

        async def tracked_next(ctx):
            call_order.append("next")
            ctx.result = "Fabricated nonsense."

        context = SimpleNamespace(result=None, arguments={"input": "q"})
        with pytest.raises(HallucinationError):
            _run(f(context, tracked_next), event_loop)
        assert call_order == ["next"]
