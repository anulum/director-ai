# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Adaptive Retrieval Router Tests
"""Tests for ``director_ai.core.retrieval.adaptive_router``.

Covers factual queries, creative queries, ambiguous cases, confidence
scoring, threshold behaviour, and pattern matching.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType

import pytest

import director_ai.core.retrieval.adaptive_router as router_mod
from director_ai.core.retrieval.adaptive_router import (
    AdaptiveRouter,
    RoutingDecision,
)


def _assert_mandatory_failure(
    exc_type: type[Exception], action: Callable[[], object]
) -> None:
    """Require the action to propagate the selected FFI failure type.

    Parameters
    ----------
    exc_type : type of Exception
        Error class that the routing action must raise.
    action : callable
        Zero-argument routing action exercised inside the error assertion.
    """
    with pytest.raises(exc_type, match="ffi"):
        action()


# ── Factual queries → should retrieve ──────────────────────────────────


class TestFactualQueries:
    """Check retrieval decisions for factual questions, verification and source references."""

    def test_what_is_question(self) -> None:
        """Retrieve for a refund-policy question with at least half-scale confidence."""
        r = AdaptiveRouter()
        d = r.should_retrieve("What is the refund policy?")
        assert d.retrieve is True
        assert d.confidence >= 0.5

    def test_who_is_question(self) -> None:
        """Retrieve for a question identifying an organization executive."""
        d = AdaptiveRouter().should_retrieve("Who is the CEO of ANULUM?")
        assert d.retrieve is True

    def test_how_many_question(self) -> None:
        """Retrieve for a question asking for a system layer count."""
        d = AdaptiveRouter().should_retrieve("How many layers does SCPN have?")
        assert d.retrieve is True

    def test_verify_request(self) -> None:
        """Retrieve for an explicit request to verify a physical claim."""
        d = AdaptiveRouter().should_retrieve("Verify that water boils at 100°C")
        assert d.retrieve is True

    def test_policy_question(self) -> None:
        """Retrieve for a regulatory data-retention question."""
        d = AdaptiveRouter().should_retrieve(
            "What does the regulation say about data retention?"
        )
        assert d.retrieve is True

    def test_according_to(self) -> None:
        """Retrieve for a question grounded in a referenced specification."""
        d = AdaptiveRouter().should_retrieve(
            "According to the specification, what is the maximum load?"
        )
        assert d.retrieve is True


# ── Creative / conversational → should skip ─────────────────────────────


class TestCreativeQueries:
    """Check retrieval bypass for creative, conversational and transformation requests."""

    def test_write_poem(self) -> None:
        """Bypass retrieval for a request to write a seasonal poem."""
        d = AdaptiveRouter().should_retrieve("Write me a poem about spring")
        assert d.retrieve is False

    def test_compose_email(self) -> None:
        """Bypass retrieval for a request to compose an email."""
        d = AdaptiveRouter().should_retrieve("Compose an email to the team")
        assert d.retrieve is False

    def test_greeting(self) -> None:
        """Bypass retrieval for a conversational greeting."""
        d = AdaptiveRouter().should_retrieve("Hello, how are you?")
        assert d.retrieve is False

    def test_brainstorm(self) -> None:
        """Bypass retrieval for a marketing brainstorming request."""
        d = AdaptiveRouter().should_retrieve(
            "Brainstorm ideas for a marketing campaign"
        )
        assert d.retrieve is False

    def test_translate(self) -> None:
        """Bypass retrieval for a translation request."""
        d = AdaptiveRouter().should_retrieve("Translate this to German")
        assert d.retrieve is False

    def test_tell_joke(self) -> None:
        """Bypass retrieval for a joke-writing request."""
        d = AdaptiveRouter().should_retrieve("Tell me a joke about programmers")
        assert d.retrieve is False

    def test_code_request(self) -> None:
        """Bypass retrieval for a request to write a sorting function."""
        d = AdaptiveRouter().should_retrieve("Code a function that sorts a list")
        assert d.retrieve is False


# ── Confidence scoring ──────────────────────────────────────────────────


class TestConfidence:
    """Check unit-interval confidence and minimum confidence for explicit query types."""

    def test_confidence_range(self) -> None:
        """Keep confidence within zero and one for factual, creative and greeting queries."""
        router = AdaptiveRouter()
        for query in [
            "What is X?",
            "Write me a poem",
            "Hello",
            "Explain the policy on returns",
        ]:
            d = router.should_retrieve(query)
            assert 0.0 <= d.confidence <= 1.0

    def test_factual_has_high_confidence(self) -> None:
        """Give an explicit capital-city question confidence of at least 0.7."""
        d = AdaptiveRouter().should_retrieve("What is the capital of France?")
        assert d.confidence >= 0.7

    def test_creative_has_reasonable_confidence(self) -> None:
        """Give an explicit haiku request confidence of at least 0.5."""
        d = AdaptiveRouter().should_retrieve("Write a haiku")
        assert d.confidence >= 0.5


# ── Threshold behaviour ────────────────────────────────────────────────


class TestThreshold:
    """Check ambiguous decision shape and retrieval of strong factual signals."""

    def test_high_threshold_skips_ambiguous(self) -> None:
        # With very high threshold, ambiguous queries should not retrieve
        """Return a Boolean decision for an ambiguous query under a high factual threshold."""
        router = AdaptiveRouter(factual_threshold=0.95, default_retrieve=False)
        d = router.should_retrieve("Tell me about science")
        # This is ambiguous — should respect default_retrieve=False
        # when confidence is below threshold
        assert isinstance(d.retrieve, bool)

    def test_low_threshold_retrieves_more(self) -> None:
        """Retrieve for a factual query under a low factual threshold."""
        router = AdaptiveRouter(factual_threshold=0.1)
        d = router.should_retrieve("What is photosynthesis?")
        assert d.retrieve is True

    def test_default_retrieve_true(self) -> None:
        """Return a Boolean decision for an ambiguous query with retrieval enabled by default."""
        router = AdaptiveRouter(default_retrieve=True)
        d = router.should_retrieve("Something ambiguous happening here")
        # Ambiguous should default to True
        assert isinstance(d.retrieve, bool)

    def test_default_retrieve_false(self) -> None:
        """Retrieve for a specification query even when the default bypasses retrieval."""
        router = AdaptiveRouter(default_retrieve=False)
        # Even with default=False, strong factual signals should still retrieve
        d = router.should_retrieve("What is the specification for this component?")
        assert d.retrieve is True


# ── Task type detection ─────────────────────────────────────────────────


class TestTaskType:
    """Check decision fields and mandatory native versus optional Python detector behavior."""

    def test_task_type_returned(self) -> None:
        """Return a nonempty task-type string for a factual policy question."""
        d = AdaptiveRouter().should_retrieve("What is the refund policy?")
        assert isinstance(d.task_type, str)
        assert len(d.task_type) > 0

    def test_routing_decision_is_frozen(self) -> None:
        """Retain the configured retrieve, task-type and confidence decision fields."""
        d = RoutingDecision(retrieve=True, task_type="qa", confidence=0.9)
        assert d.retrieve is True
        assert d.task_type == "qa"
        assert d.confidence == 0.9

    def test_rust_task_type_exception_is_mandatory_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Propagate a native task-detector RuntimeError from the retrieval decision.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Enable native detection and replace its binding with a RuntimeError.
        """
        monkeypatch.setattr(router_mod, "_RUST_AVAILABLE", True)
        monkeypatch.setattr(
            router_mod,
            "_rust_task_type",
            lambda _query, _response: (_ for _ in ()).throw(RuntimeError("ffi fail")),
            raising=True,
        )
        _assert_mandatory_failure(
            RuntimeError,
            lambda: AdaptiveRouter().should_retrieve("What is the refund policy?"),
        )

    def test_rust_task_type_non_runtime_exception_is_mandatory_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Propagate a native task-detector ValueError from the retrieval decision.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Enable native detection and replace its binding with a ValueError.
        """
        monkeypatch.setattr(router_mod, "_RUST_AVAILABLE", True)
        monkeypatch.setattr(
            router_mod,
            "_rust_task_type",
            lambda _query, _response: (_ for _ in ()).throw(ValueError("ffi fail")),
            raising=True,
        )
        _assert_mandatory_failure(
            ValueError,
            lambda: AdaptiveRouter().should_retrieve("What is the refund policy?"),
        )

    def test_rust_task_type_type_error_is_mandatory_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Propagate a native task-detector TypeError from the retrieval decision.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Enable native detection and replace its binding with a TypeError.
        """
        monkeypatch.setattr(router_mod, "_RUST_AVAILABLE", True)
        monkeypatch.setattr(
            router_mod,
            "_rust_task_type",
            lambda _query, _response: (_ for _ in ()).throw(TypeError("ffi fail")),
            raising=True,
        )
        _assert_mandatory_failure(
            TypeError,
            lambda: AdaptiveRouter().should_retrieve("What is the refund policy?"),
        )

    def test_python_detector_import_error_defaults_safely(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Use the default task type and disabled-retrieval default if Python detector import fails.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Disable native detection and intercept the optional Python detector import.
        """
        original_import = builtins.__import__

        def blocked_import(
            name: str,
            globals: Mapping[str, object] | None = None,
            locals: Mapping[str, object] | None = None,
            fromlist: Sequence[str] | None = (),
            level: int = 0,
        ) -> ModuleType:
            """Reject only the optional Python task-detector import and delegate every other import.

            Parameters
            ----------
            name : str
                Import target whose exact task-detector name is intercepted.
            globals, locals : mapping of str to object or None, optional
                Import namespaces forwarded unchanged to the original builtin.
            fromlist : sequence of str or None, optional
                Requested imported names forwarded to the original builtin.
            level : int, optional
                Relative import level forwarded to the original builtin.

            Returns
            -------
            ModuleType
                Module returned by the original import for other targets.

            Raises
            ------
            ImportError
                If the target is the optional task-detector module.
            """
            if name == "director_ai.core.scoring._task_scoring":
                raise ImportError("task detector unavailable")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(router_mod, "_RUST_AVAILABLE", False)
        monkeypatch.setattr(router_mod, "_rust_task_type", None)
        monkeypatch.setattr(builtins, "__import__", blocked_import)

        d = AdaptiveRouter(default_retrieve=False).should_retrieve(
            "Something ambiguous"
        )

        assert d.task_type == "default"
        assert d.retrieve is False

    def test_python_detector_success_path_when_rust_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Return a task type and retrieve the factual query through the Python detector.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Disable native detection while preserving the actual Python task detector.
        """
        monkeypatch.setattr(router_mod, "_RUST_AVAILABLE", False)
        monkeypatch.setattr(router_mod, "_rust_task_type", None)

        d = AdaptiveRouter().should_retrieve("What is the refund policy?")

        assert d.task_type
        assert d.retrieve is True


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """Check decision shape and confidence for empty, long, mixed and Unicode inputs."""

    def test_empty_query(self) -> None:
        """Return a Boolean decision for an empty query."""
        d = AdaptiveRouter().should_retrieve("")
        assert isinstance(d.retrieve, bool)

    def test_very_long_query(self) -> None:
        """Retrieve for a long query repeating an explicit factual question prefix."""
        d = AdaptiveRouter().should_retrieve("what is " * 1000)
        assert d.retrieve is True

    def test_mixed_signals(self) -> None:
        """Return a Boolean decision and bounded confidence for mixed creative and factual cues."""
        d = AdaptiveRouter().should_retrieve(
            "Write me a summary based on the policy document"
        )
        # Should not crash; confidence should be lower
        assert isinstance(d.retrieve, bool)
        assert 0.0 <= d.confidence <= 1.0

    def test_unicode_query(self) -> None:
        """Return a Boolean decision for a Slovak question containing accented characters."""
        d = AdaptiveRouter().should_retrieve("Čo je to fotosyntéza?")
        assert isinstance(d.retrieve, bool)

    def test_with_response(self) -> None:
        """Return a Boolean decision when a candidate response accompanies the query."""
        d = AdaptiveRouter().should_retrieve(
            "Tell me about water",
            response="Water is H2O, a molecule consisting of two hydrogen atoms.",
        )
        assert isinstance(d.retrieve, bool)
