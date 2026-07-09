# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface tests for the live streaming halt demo runtime."""

from __future__ import annotations

from collections import Counter

import director_ai.ui.streaming_halt_live as runtime
from director_ai.ui.streaming_halt_live import (
    SCENARIOS,
    render_banner,
    render_token_span,
    run_live_demo,
)
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def test_live_streaming_halt_unit_guard_has_real_surface_companion() -> None:
    """The adapter unit guard should be backed by the reusable runtime tests."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_demo_streaming_halt_live.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_demo_streaming_halt_live_real_surface.py" in category


def test_live_streaming_halt_runtime_escapes_html_payloads() -> None:
    """The reusable demo runtime should not render raw HTML from stream data."""
    token = '<script>alert("token")</script>'
    reason = '<img src=x onerror="alert(1)">'

    span = render_token_span(token, coherence=0.9, halted=False)
    banner = render_banner(
        True,
        reason,
        {"token_count": 1, "avg_coherence": 0.2, "warning_count": 0},
    )

    assert token not in span
    assert reason not in banner
    assert "&lt;script&gt;" in span
    assert "&lt;img" in banner


def test_live_streaming_halt_runtime_handles_defensive_rendering_edges() -> None:
    """Rendering should tolerate NaN scores and mixed summary value types."""
    span = render_token_span("nan-score", coherence=float("nan"), halted=False)
    coerced = render_banner(
        False,
        "",
        {"token_count": "2", "avg_coherence": "0.75", "warning_count": True},
    )
    invalid = render_banner(False, "", {"avg_coherence": "not-a-number"})
    non_numeric = render_banner(False, "", {"avg_coherence": object()})

    assert "#fecaca" in span
    assert "tokens: 2" in coerced
    assert "avg coherence: 0.750" in coerced
    assert "warnings: 0" in coerced
    assert "avg coherence: 0.000" in invalid
    assert "avg coherence: 0.000" in non_numeric
    assert runtime.token_colour(float("nan")) == ("#fecaca", "#991b1b")


def test_live_streaming_halt_runtime_uses_real_kernel_events() -> None:
    """The runtime should stream all shipped scenarios through StreamingKernel."""
    outcomes: Counter[str] = Counter()

    for scenario_key in SCENARIOS:
        frames = list(run_live_demo(scenario_key, speed_s=0.0))
        assert frames
        final_banner = frames[-1][2]
        outcomes["halted" if "HALTED" in final_banner else "approved"] += 1

    assert outcomes == {"halted": 2, "approved": 1}


def test_score_callback_repeats_last_score_once_curve_is_exhausted() -> None:
    """A drained score curve should keep answering with its final value."""
    scenario = next(iter(SCENARIOS.values()))
    callback = runtime.score_callback_for(scenario)

    served = [callback(f"token-{i}") for i in range(len(scenario.scores))]
    assert served == scenario.scores

    assert callback("beyond-the-curve") == scenario.scores[-1]
    assert callback("still-beyond") == scenario.scores[-1]
