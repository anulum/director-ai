# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — tests for demo/streaming_halt_live.py

"""Structural tests for the live streaming halt demo. Exercises the
scenario catalogue, the Gradio generator, and the HTML rendering
helpers without actually launching the Gradio server."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

# Load the demo module from its source path so the repo's
# ``demo/`` directory does not need to be on PYTHONPATH.
_DEMO_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "demo" / "streaming_halt_live.py"
)

# Gradio might not be installed in the CI venv; skip cleanly.
gradio = pytest.importorskip("gradio")


def _load_demo():
    spec = importlib.util.spec_from_file_location(
        "director_ai_demo_streaming_halt_live", _DEMO_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo()


class TestScenarios:
    def test_three_scenarios_shipped(self):
        assert set(demo.SCENARIOS.keys()) == {"truthful", "hard_halt", "drift"}

    def test_tokens_and_scores_aligned(self):
        for scenario in demo.SCENARIOS.values():
            assert len(scenario.tokens) == len(scenario.scores), (
                f"{scenario.name}: token/score length mismatch"
            )

    def test_descriptions_present(self):
        for scenario in demo.SCENARIOS.values():
            assert scenario.description.strip()


class TestHelpers:
    def test_token_span_applies_halted_strikethrough(self):
        span = demo._token_span("tok", coherence=0.1, halted=True)
        assert "line-through" in span

    def test_token_span_preserves_nbsp_for_leading_space(self):
        span = demo._token_span(" tok", coherence=0.9, halted=False)
        assert "&nbsp;" in span

    def test_gauge_clamps_values(self):
        low = demo._gauge(-1.0)
        high = demo._gauge(2.0)
        assert "width:0%" in low
        assert "width:100%" in high

    def test_banner_halted_shows_reason(self):
        html = demo._banner(
            True, "hard_limit", {"token_count": 5, "avg_coherence": 0.4}
        )
        assert "HALTED" in html
        assert "hard_limit" in html

    def test_banner_approved_shows_approved(self):
        html = demo._banner(
            False, "", {"token_count": 10, "avg_coherence": 0.9, "warning_count": 0}
        )
        assert "APPROVED" in html


class TestLiveGenerator:
    def _drain(self, key: str):
        frames = list(demo.run_live_demo(key, speed_s=0.0))
        assert frames, "generator must yield at least one frame"
        return frames

    def test_truthful_scenario_completes_with_approved_banner(self):
        frames = self._drain("truthful")
        last_banner = frames[-1][2]
        assert "APPROVED" in last_banner

    def test_hard_halt_scenario_banner_flags_halt(self):
        frames = self._drain("hard_halt")
        banners = [f[2] for f in frames]
        assert any("HALTED" in b for b in banners), banners[-1]

    def test_drift_scenario_eventually_halts(self):
        frames = self._drain("drift")
        banners = [f[2] for f in frames]
        # Drift may resolve via trend or window halt; either way the
        # final banner announces a halt.
        assert "HALTED" in banners[-1]

    def test_unknown_scenario_rejected(self):
        with pytest.raises(ValueError, match="unknown scenario"):
            list(demo.run_live_demo("nope", speed_s=0.0))


class TestAppBuilder:
    def test_build_app_returns_blocks(self):
        app = demo.build_app()
        assert isinstance(app, gradio.Blocks)
