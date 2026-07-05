# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Structural tests for the live streaming halt Gradio adapter."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from typing import Protocol, cast

import pytest

gradio = pytest.importorskip("gradio")
if not hasattr(gradio, "Blocks"):  # pragma: no cover - environment-dependent
    pytest.skip(
        "gradio is present but exposes no Blocks API (partial install)",
        allow_module_level=True,
    )


class DemoScenario(Protocol):
    """Typed subset of a shipped demo scenario."""

    name: str
    tokens: list[str]
    scores: list[float]
    description: str


class DemoModule(Protocol):
    """Typed subset of the launchable demo adapter under test."""

    SCENARIOS: dict[str, DemoScenario]

    def _token_span(self, token: str, coherence: float, halted: bool) -> str:
        """Render one token span."""

    def _gauge(self, coherence: float) -> str:
        """Render the coherence gauge."""

    def _banner(
        self,
        halted: bool,
        halt_reason: str,
        summary: dict[str, float],
    ) -> str:
        """Render the stream verdict banner."""

    def run_live_demo(self, scenario_key: str, speed_s: float) -> Iterator[Frame]:
        """Run one demo scenario and yield HTML frames."""

    def build_app(self) -> object:
        """Build the Gradio Blocks application."""


Frame = tuple[str, str, str]


def _load_demo() -> DemoModule:
    """Import the checked-in launchable demo module."""
    return cast(DemoModule, importlib.import_module("demo.streaming_halt_live"))


demo = _load_demo()


class TestScenarios:
    """Scenario catalogue invariants."""

    def test_three_scenarios_shipped(self) -> None:
        """The launchable demo should expose the three documented scenarios."""
        assert set(demo.SCENARIOS.keys()) == {"truthful", "hard_halt", "drift"}

    def test_tokens_and_scores_aligned(self) -> None:
        """Every scenario should pair each token with exactly one score."""
        for scenario in demo.SCENARIOS.values():
            assert len(scenario.tokens) == len(scenario.scores), (
                f"{scenario.name}: token/score length mismatch"
            )

    def test_descriptions_present(self) -> None:
        """Every scenario should include operator-facing explanatory text."""
        for scenario in demo.SCENARIOS.values():
            assert scenario.description.strip()


class TestHelpers:
    """HTML helper coverage for the demo adapter aliases."""

    def test_token_span_applies_halted_strikethrough(self) -> None:
        """Halted tokens should render with a strike-through decoration."""
        span = demo._token_span("tok", coherence=0.1, halted=True)

        assert "line-through" in span

    def test_token_span_preserves_nbsp_for_leading_space(self) -> None:
        """Leading token whitespace should survive HTML rendering."""
        span = demo._token_span(" tok", coherence=0.9, halted=False)

        assert "&nbsp;" in span

    def test_gauge_clamps_values(self) -> None:
        """Gauge width should clamp outside the normalized score range."""
        low = demo._gauge(-1.0)
        high = demo._gauge(2.0)

        assert "width:0%" in low
        assert "width:100%" in high

    def test_banner_halted_shows_reason(self) -> None:
        """Halt banners should expose the kernel halt reason."""
        html = demo._banner(
            True,
            "hard_limit",
            {"token_count": 5.0, "avg_coherence": 0.4},
        )

        assert "HALTED" in html
        assert "hard_limit" in html

    def test_banner_approved_shows_approved(self) -> None:
        """Completed streams should render an approved verdict."""
        html = demo._banner(
            False,
            "",
            {"token_count": 10.0, "avg_coherence": 0.9, "warning_count": 0.0},
        )

        assert "APPROVED" in html


class TestLiveGenerator:
    """Generator-level coverage for the launchable demo adapter."""

    def _drain(self, key: str) -> list[Frame]:
        """Collect all emitted frames for one demo scenario."""
        frames = list(demo.run_live_demo(key, speed_s=0.0))
        assert frames, "generator must yield at least one frame"
        return frames

    def test_truthful_scenario_completes_with_approved_banner(self) -> None:
        """The truthful scenario should complete without a halt."""
        frames = self._drain("truthful")
        last_banner = frames[-1][2]

        assert "APPROVED" in last_banner

    def test_hard_halt_scenario_banner_flags_halt(self) -> None:
        """The hard-halt scenario should surface a halt banner."""
        frames = self._drain("hard_halt")
        banners = [frame[2] for frame in frames]

        assert any("HALTED" in banner for banner in banners), banners[-1]

    def test_drift_scenario_eventually_halts(self) -> None:
        """The drift scenario should halt by the final frame."""
        frames = self._drain("drift")
        banners = [frame[2] for frame in frames]

        assert "HALTED" in banners[-1]

    def test_unknown_scenario_rejected(self) -> None:
        """Unknown scenario keys should fail before streaming begins."""
        with pytest.raises(ValueError, match="unknown scenario"):
            list(demo.run_live_demo("nope", speed_s=0.0))


class TestAppBuilder:
    """Gradio app construction coverage."""

    def test_build_app_returns_blocks(self) -> None:
        """The demo adapter should construct a real Gradio Blocks instance."""
        app = demo.build_app()

        assert type(app).__name__ == "Blocks"
