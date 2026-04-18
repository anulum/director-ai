# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Live streaming halt demo (Gradio)

"""Gradio demo that streams tokens one at a time and surfaces each
coherence verdict as it happens. The existing ``demo/app.py`` shows
post-hoc scoring and a pre-baked streaming scenario; this module is
the *interactive* counterpart — UI updates with every emitted token
and the halt banner appears on the same frame the kernel stops the
stream.

Intended for local use and, after CEO approval, publication as a
Hugging Face Space. The file is deliberately standalone so the
Space can ship just this one script plus ``requirements.txt``.

Run locally::

    python demo/streaming_halt_live.py

The module can also be imported from ``demo/app.py`` to add a
``Live streaming halt`` tab to the full demo.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from queue import Empty, Queue

import gradio as gr

from director_ai.core.observability.callbacks import (
    TokenTraceCallback,
    TokenTraceEvent,
)
from director_ai.core.runtime.streaming import StreamingKernel

# ---------------------------------------------------------------
# Scenario catalogue
# ---------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    tokens: list[str]
    scores: list[float]
    description: str


SCENARIOS: dict[str, Scenario] = {
    "truthful": Scenario(
        name="Truthful — no halt",
        description=(
            "Every token lands well above the 0.5 hard limit. The "
            "stream runs to completion; no halt banner fires."
        ),
        tokens=[
            "Water",
            " boils",
            " at",
            " 100",
            " degrees",
            " Celsius",
            " at",
            " sea",
            " level",
            ".",
        ],
        scores=[0.92, 0.90, 0.88, 0.91, 0.93, 0.90, 0.88, 0.89, 0.90, 0.90],
    ),
    "hard_halt": Scenario(
        name="Blatant hallucination — hard-limit halt",
        description=(
            "Sentence starts on-topic, then introduces a fabricated "
            "claim whose coherence drops below the hard limit. The "
            "kernel halts the stream on that token."
        ),
        tokens=[
            "Water",
            " boils",
            " at",
            " 100",
            " C.",
            " In",
            " fact",
            ",",
            " negative",
            " forty",
            " C",
            " is",
            " also",
            " correct",
            ".",
        ],
        scores=[
            0.92,
            0.90,
            0.91,
            0.89,
            0.88,
            0.85,
            0.82,
            0.79,
            0.15,
            0.10,
            0.08,
            0.05,
            0.03,
            0.03,
            0.01,
        ],
    ),
    "drift": Scenario(
        name="Gradual drift — trend halt",
        description=(
            "Coherence decays smoothly as the response drifts off "
            "source. No single token crosses the hard limit but the "
            "downward trend check fires mid-stream."
        ),
        tokens=[
            "Paris",
            " is",
            " the",
            " capital",
            " of",
            " France",
            " and",
            " also",
            " the",
            " largest",
            " city",
            " in",
            " the",
            " European",
            " Union",
            " by",
            " far",
            ".",
        ],
        scores=[
            0.91,
            0.90,
            0.90,
            0.89,
            0.88,
            0.89,
            0.84,
            0.80,
            0.75,
            0.66,
            0.58,
            0.50,
            0.43,
            0.38,
            0.33,
            0.28,
            0.23,
            0.18,
        ],
    ),
}


# ---------------------------------------------------------------
# Live trace callback → queue → UI
# ---------------------------------------------------------------


@dataclass
class _LiveEvent:
    index: int
    token: str
    coherence: float
    halted: bool
    halt_reason: str = ""
    final: bool = False
    summary: dict[str, float] = field(default_factory=dict)


class _QueueCallback(TokenTraceCallback):
    """Push every on_token / on_stream_end record to a thread-safe
    queue so the Gradio generator can drain it in parallel."""

    def __init__(self, queue: Queue[_LiveEvent]) -> None:
        self._queue = queue

    def on_token(self, event: TokenTraceEvent) -> None:
        self._queue.put(
            _LiveEvent(
                index=event.index,
                token=event.token,
                coherence=event.coherence,
                halted=event.halted,
                halt_reason=event.halt_reason,
            )
        )

    def on_stream_end(self, *, tenant_id: str, request_id: str, summary: dict) -> None:
        self._queue.put(
            _LiveEvent(
                index=-1,
                token="",
                coherence=summary.get("avg_coherence", 0.0),
                halted=bool(summary.get("halted", False)),
                halt_reason=str(summary.get("halt_reason", "")),
                final=True,
                summary={
                    "token_count": float(summary.get("token_count", 0)),
                    "avg_coherence": float(summary.get("avg_coherence", 0.0)),
                    "warning_count": float(summary.get("warning_count", 0)),
                },
            )
        )


# ---------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------


_SCORE_COLOURS = (
    (0.70, "#dcfce7", "#166534"),  # green
    (0.55, "#fef9c3", "#854d0e"),  # amber
    (0.00, "#fecaca", "#991b1b"),  # red
)


def _token_colour(coherence: float) -> tuple[str, str]:
    """Pick (background, foreground) hex colours for ``coherence``."""
    for cutoff, bg, fg in _SCORE_COLOURS:
        if coherence >= cutoff:
            return bg, fg
    # Last tuple in _SCORE_COLOURS has cutoff 0.0, so the loop always
    # returns above. This is a belt-and-braces fallback.
    _, bg, fg = _SCORE_COLOURS[-1]
    return bg, fg


def _token_span(token: str, coherence: float, halted: bool) -> str:
    bg, fg = _token_colour(coherence)
    decoration = "line-through" if halted else "none"
    display = token.replace(" ", "&nbsp;")
    return (
        f"<span style='background:{bg};color:{fg};"
        f"text-decoration:{decoration};padding:2px 6px;"
        f"border-radius:4px;margin:2px;display:inline-block;"
        f"font-family:ui-monospace,monospace;font-size:14px' "
        f"title='score={coherence:.3f}'>{display}</span>"
    )


def _banner(halted: bool, halt_reason: str, summary: dict) -> str:
    if halted:
        return (
            "<div style='background:#ef4444;color:white;padding:12px;"
            "border-radius:8px;font-weight:bold;font-size:1.2em'>"
            f"HALTED — {halt_reason or 'coherence floor reached'}<br>"
            "<span style='font-weight:normal;font-size:0.85em'>"
            f"tokens emitted: {int(summary.get('token_count', 0))}, "
            f"avg coherence: {summary.get('avg_coherence', 0):.3f}, "
            f"warnings: {int(summary.get('warning_count', 0))}"
            "</span></div>"
        )
    return (
        "<div style='background:#22c55e;color:white;padding:12px;"
        "border-radius:8px;font-weight:bold;font-size:1.2em'>"
        "APPROVED — stream completed<br>"
        "<span style='font-weight:normal;font-size:0.85em'>"
        f"tokens: {int(summary.get('token_count', 0))}, "
        f"avg coherence: {summary.get('avg_coherence', 0):.3f}, "
        f"warnings: {int(summary.get('warning_count', 0))}"
        "</span></div>"
    )


def _wrap_strip(html: str) -> str:
    return (
        "<div style='line-height:2.2;padding:12px;"
        "background:#f9fafb;border-radius:8px;"
        "min-height:80px'>"
        f"{html}</div>"
    )


def _gauge(coherence: float) -> str:
    pct = int(max(0.0, min(1.0, coherence)) * 100)
    if coherence >= 0.7:
        colour = "#22c55e"
    elif coherence >= 0.55:
        colour = "#f59e0b"
    else:
        colour = "#ef4444"
    return (
        "<div style='background:#e5e7eb;border-radius:6px;"
        "height:18px;width:100%'>"
        f"<div style='background:{colour};height:100%;"
        f"border-radius:6px;width:{pct}%;transition:width 0.15s'>"
        "</div></div>"
        f"<div style='text-align:right;font-family:monospace;"
        f"font-size:12px;margin-top:4px'>"
        f"last score: {coherence:.3f}</div>"
    )


# ---------------------------------------------------------------
# Streaming engine
# ---------------------------------------------------------------


def _paced_tokens(scenario: Scenario, delay_s: float) -> Iterable[str]:
    """Emit tokens at a fixed cadence so the UI has time to breathe."""
    for token in scenario.tokens:
        yield token
        time.sleep(delay_s)


def _score_queue_for(scenario: Scenario) -> Iterator[float]:
    """Return a coherence-score callable closing over the fixed curve."""
    iterator = iter(scenario.scores)
    last = 0.5

    def _cb(_accumulated: str) -> float:
        nonlocal last
        with contextlib.suppress(StopIteration):
            last = next(iterator)
        return last

    return _cb  # type: ignore[return-value]


def run_live_demo(scenario_key: str, speed_s: float) -> Iterator[tuple[str, str, str]]:
    """Gradio generator — yields ``(strip_html, gauge_html, banner_html)``
    tuples as each token arrives. The kernel runs in a background
    thread so the UI generator can pace the Queue drain."""
    scenario = SCENARIOS.get(scenario_key)
    if scenario is None:
        raise ValueError(f"unknown scenario: {scenario_key}")

    queue: Queue[_LiveEvent] = Queue()
    callback = _QueueCallback(queue)
    kernel = StreamingKernel(
        hard_limit=0.4,
        window_size=4,
        window_threshold=0.5,
        trend_window=4,
        trend_threshold=0.25,
    )

    def _worker() -> None:
        kernel.stream_tokens(
            _paced_tokens(scenario, speed_s),
            coherence_callback=_score_queue_for(scenario),
            trace_callbacks=[callback],
            tenant_id="demo",
            request_id="live",
        )

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    spans: list[str] = []
    banner_html = _banner(False, "", {})
    gauge_html = _gauge(0.5)
    last_coherence = 0.5

    while True:
        try:
            event = queue.get(timeout=5.0)
        except Empty:
            break
        if event.final:
            banner_html = _banner(event.halted, event.halt_reason, event.summary)
            yield _wrap_strip("".join(spans)), gauge_html, banner_html
            break
        spans.append(_token_span(event.token, event.coherence, event.halted))
        last_coherence = event.coherence
        gauge_html = _gauge(last_coherence)
        if event.halted:
            banner_html = _banner(
                True,
                event.halt_reason,
                {
                    "token_count": event.index + 1,
                    "avg_coherence": last_coherence,
                    "warning_count": 0,
                },
            )
        yield _wrap_strip("".join(spans)), gauge_html, banner_html

    thread.join(timeout=1.0)


# ---------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------


_DESCRIPTION = """
# Director-AI — Live Streaming Halt

Every token scored as it arrives. When coherence drops below the
configured floor — or the sliding window average slips, or the
rolling trend turns sharply downward — the kernel halts the stream
and the red banner fires on the same frame the token would have been
emitted.

Pick a scenario below and watch the strip fill in real time. The
gauge tracks the latest coherence score; the banner updates once the
stream ends (either APPROVED or HALTED).
""".strip()


def build_app() -> gr.Blocks:
    # Gradio 6 moved theme from the Blocks constructor to ``launch()``.
    # Construct the Blocks object plainly; callers that want a themed
    # launch pass ``theme=gr.themes.Soft()`` to ``app.launch``.
    with gr.Blocks(title="Director-AI Live Halt") as app:
        gr.Markdown(_DESCRIPTION)

        scenario_dd = gr.Dropdown(
            label="Scenario",
            choices=[(s.name, key) for key, s in SCENARIOS.items()],
            value="hard_halt",
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
            strip_html = gr.HTML(label="Token strip", value=_wrap_strip(""))
        with gr.Row():
            gauge_html = gr.HTML(label="Coherence gauge", value=_gauge(0.5))
        with gr.Row():
            banner_html = gr.HTML(label="Verdict", value=_banner(False, "", {}))

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

        # Populate the description on first load.
        app.load(
            lambda: _describe(list(SCENARIOS)[0]),
            inputs=None,
            outputs=[scenario_description],
        )

    return app


if __name__ == "__main__":
    build_app().launch(theme=gr.themes.Soft())
