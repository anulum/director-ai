# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Gradio adapter for the live streaming halt demo.

The reusable runtime lives in :mod:`director_ai.ui.streaming_halt_live`; this
script keeps the launchable demo surface thin so the real kernel path and HTML
rendering can be tested without starting a Gradio server.
"""

from __future__ import annotations

import gradio as gr

from director_ai.ui.streaming_halt_live import (
    SCENARIOS,
    paced_tokens,
    render_banner,
    render_gauge,
    render_token_span,
    run_live_demo,
    score_callback_for,
    wrap_token_strip,
)
from director_ai.ui.streaming_halt_live import (
    Scenario as _Scenario,
)

_DEFAULT_SCENARIO = "hard_halt"

# Backward-compatible helper aliases for existing imports and tests.
Scenario = _Scenario
_banner = render_banner
_gauge = render_gauge
_paced_tokens = paced_tokens
_score_queue_for = score_callback_for
_token_span = render_token_span
_wrap_strip = wrap_token_strip

_DESCRIPTION = """
# Director-AI - Live Streaming Halt

Every token scored as it arrives. When coherence drops below the configured
floor, the sliding window average slips, or the rolling trend turns sharply
downward, the kernel halts the stream and the red banner fires on the same
frame the token would have been emitted.

Pick a scenario below and watch the strip fill in real time. The gauge tracks
the latest coherence score; the banner updates once the stream ends, either
APPROVED or HALTED.
""".strip()


def build_app() -> gr.Blocks:
    """Build the launchable Gradio Blocks app.

    Returns
    -------
    gr.Blocks
        Configured live streaming halt demo.
    """
    with gr.Blocks(title="Director-AI Live Halt") as app:
        gr.Markdown(_DESCRIPTION)

        scenario_dd = gr.Dropdown(
            label="Scenario",
            choices=[(scenario.name, key) for key, scenario in SCENARIOS.items()],
            value=_DEFAULT_SCENARIO,
        )
        scenario_description = gr.Markdown()
        speed = gr.Slider(
            label="Per-token delay (seconds)",
            minimum=0.02,
            maximum=0.6,
            value=0.12,
            step=0.02,
        )
        run_btn = gr.Button("Stream", variant="primary")

        with gr.Row():
            strip_html = gr.HTML(label="Token strip", value=wrap_token_strip(""))
        with gr.Row():
            gauge_html = gr.HTML(label="Coherence gauge", value=render_gauge(0.5))
        with gr.Row():
            banner_html = gr.HTML(label="Verdict", value=render_banner(False, "", {}))

        def _describe(key: str) -> str:
            scenario = SCENARIOS.get(key)
            return scenario.description if scenario else ""

        scenario_dd.change(
            _describe,
            inputs=[scenario_dd],
            outputs=[scenario_description],
        )
        run_btn.click(
            run_live_demo,
            inputs=[scenario_dd, speed],
            outputs=[strip_html, gauge_html, banner_html],
        )
        app.load(
            lambda: _describe(_DEFAULT_SCENARIO),
            inputs=None,
            outputs=[scenario_description],
        )

    return app


if __name__ == "__main__":
    build_app().launch(theme=gr.themes.Soft())
