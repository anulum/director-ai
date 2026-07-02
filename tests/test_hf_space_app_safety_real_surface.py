# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Hugging Face Space app real-surface tests
"""Real Gradio app coverage for the checked-in Hugging Face Space demo."""

from __future__ import annotations

import asyncio
import importlib
import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol, TypedDict, cast

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


class HfSpaceApp(Protocol):
    """Typed subset of the public Space callbacks exercised by these tests."""

    COMPARISON_SCENARIOS: dict[str, dict[str, str]]
    STREAMING_SCENARIOS: dict[str, StreamingScenario]
    score_response: Callable[[str, str, str], tuple[str, str, str, str]]
    run_comparison: Callable[[str], tuple[str, str, str]]
    run_streaming_demo: Callable[[str], str]
    build_app: Callable[[], GradioBlocks]


class GradioBlocks(Protocol):
    """Typed subset of Gradio Blocks used by the real-surface tests."""

    def call_function(
        self,
        block_fn: int,
        processed_input: Sequence[object],
    ) -> Awaitable[Mapping[str, object]]:
        """Call a registered Gradio callback by function index."""


class StreamingScenario(TypedDict):
    """Streaming scenario shape consumed by the Space demo callback."""

    tokens: list[str]
    scores: list[float]


def _load_app() -> HfSpaceApp:
    """Import the checked-in Space app with the installed runtime dependencies."""
    return cast(HfSpaceApp, importlib.import_module("demo.app"))


async def _call_profile_callback(app: GradioBlocks) -> Mapping[str, object]:
    """Call the registered profile callback through the Gradio app registry."""
    return await app.call_function(4, ["medical"])


def test_hf_space_app_safety_unit_guard_has_real_surface_companion() -> None:
    """The app-safety guard should be backed by real Space app coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_hf_space_app_safety.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_hf_space_app_safety_real_surface.py" in category


def test_hf_space_app_builds_checked_in_gradio_blocks() -> None:
    """The checked-in Space app should build a real Gradio Blocks object."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        app = _load_app().build_app()

    assert type(app).__name__ == "Blocks"
    assert callable(getattr(app, "launch", None))
    app_warnings = [
        str(warning.message)
        for warning in caught
        if not issubclass(warning.category, ResourceWarning)
    ]
    assert app_warnings == []


def test_hf_space_score_callback_escapes_user_markdown() -> None:
    """The public score callback should escape user-supplied fact text."""
    malicious_fact = '<script>alert("owned")</script>'

    badge, details, bar, context = _load_app().score_response(
        f"payload: {malicious_fact}",
        "payload",
        malicious_fact,
    )

    assert malicious_fact not in badge
    assert malicious_fact not in details
    assert malicious_fact not in bar
    assert malicious_fact not in context
    assert "&lt;script&gt;" in context


def test_hf_space_score_callback_handles_supported_and_missing_context() -> None:
    """The score callback should render evidence and no-context paths."""
    app = _load_app()

    _badge, supported_details, supported_bar, supported_context = app.score_response(
        "capital: Paris is the capital of France.",
        "What is the capital of France?",
        "Paris is the capital of France.",
    )
    _yellow_badge, _yellow_details, yellow_bar, _yellow_context = app.score_response(
        "boiling point: Water boils at 100 degrees Celsius at sea level.",
        "At what temperature does water boil?",
        "Water boils at 100 degrees Celsius at sea level.",
    )
    _missing_badge, missing_details, _missing_bar, missing_context = app.score_response(
        "\nignored line without separator\n",
        "What is the capital of France?",
        "Paris is the capital of France.",
    )

    assert "**Evidence:**" in supported_details
    assert "background:#22c55e" in supported_bar
    assert "background:#f59e0b" in yellow_bar
    assert "Paris is the capital of France." in supported_context
    assert "**Evidence:**" not in missing_details
    assert missing_context == "No matching facts found."


def test_hf_space_public_demo_callbacks_execute_real_paths() -> None:
    """The comparison and streaming tabs should execute their public callbacks."""
    app = _load_app()

    raw_html, guarded_html, explanation_html = app.run_comparison(
        "Capital of France (hallucination)"
    )
    streaming_markdown = app.run_streaming_demo(
        "Blatant hallucination (HARD LIMIT halt)"
    )

    assert "Berlin" in raw_html
    assert "HALTED" in guarded_html
    assert "BLOCKED" in explanation_html
    assert "**Result: HALTED**" in streaming_markdown


def test_hf_space_public_callbacks_cover_success_and_warning_paths() -> None:
    """The public comparison and streaming callbacks should cover UI states."""
    app = _load_app()
    scenario_name = "Temporary approved mixed-score scenario"
    streaming_name = "Temporary warning-level streaming scenario"
    original = app.COMPARISON_SCENARIOS.get(scenario_name)
    original_streaming = app.STREAMING_SCENARIOS.get(streaming_name)
    app.COMPARISON_SCENARIOS[scenario_name] = {
        "facts": "\nignored line without separator\ncapital: Paris is the capital of France.",
        "query": "What is the capital of France?",
        "raw_response": "Paris is the capital of France.",
        "guarded_response": "Paris is the capital of France.",
    }
    app.STREAMING_SCENARIOS[streaming_name] = {
        "tokens": ["Grounded", " warning", " recovers"],
        "scores": [0.9, 0.5, 0.7],
    }

    try:
        _raw_html, approved_html, approved_explanation = app.run_comparison(
            scenario_name
        )
        _subtle_raw, subtle_html, _subtle_explanation = app.run_comparison(
            "Sky color (subtle hallucination)"
        )
        approved_streaming = app.run_streaming_demo("Truthful response (APPROVED)")
        warning_streaming = app.run_streaming_demo(streaming_name)
    finally:
        if original is None:
            del app.COMPARISON_SCENARIOS[scenario_name]
        else:
            app.COMPARISON_SCENARIOS[scenario_name] = original
        if original_streaming is None:
            del app.STREAMING_SCENARIOS[streaming_name]
        else:
            app.STREAMING_SCENARIOS[streaming_name] = original_streaming

    assert "APPROVED" in approved_explanation
    assert "score=0.400" in approved_html
    assert "background:#fef9c3" in approved_html
    assert "background:#fecaca" in approved_html
    assert "score=0." in subtle_html
    assert "**Result: APPROVED**" in approved_streaming
    assert "warn" in warning_streaming


def test_hf_space_profile_tab_callback_runs_through_gradio_registry() -> None:
    """The real Gradio app should execute the registered profile callback."""
    app = _load_app().build_app()
    result = asyncio.run(_call_profile_callback(app))
    prediction = result["prediction"]

    assert isinstance(prediction, str)
    assert "| `profile` | `medical` |" in prediction
    assert "| `use_nli` | `True` |" in prediction
