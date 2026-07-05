# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Reusable runtime for the live streaming halt demo.

The Gradio app in :mod:`demo.streaming_halt_live` delegates its scenario
catalogue, token rendering, and kernel execution to this module so those
production behaviors can be exercised without launching a browser server.
"""

from __future__ import annotations

import contextlib
import html
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any

from director_ai.core.observability.callbacks import (
    TokenTraceCallback,
    TokenTraceEvent,
)
from director_ai.core.runtime.streaming import StreamingKernel

CoherenceCallback = Callable[[str], float]
Frame = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class Scenario:
    """Predefined streaming halt demonstration scenario.

    Parameters
    ----------
    name:
        Human-readable label shown in the scenario picker.
    tokens:
        Ordered token fragments emitted by the simulated model stream.
    scores:
        Coherence scores consumed by :class:`StreamingKernel`.
    description:
        Operator-facing explanation of the expected scenario outcome.
    """

    name: str
    tokens: list[str]
    scores: list[float]
    description: str


SCENARIOS: dict[str, Scenario] = {
    "truthful": Scenario(
        name="Truthful - no halt",
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
        name="Blatant hallucination - hard-limit halt",
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
        name="Gradual drift - trend halt",
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


@dataclass(slots=True)
class _LiveEvent:
    """Internal event passed from the streaming worker to the UI generator."""

    index: int
    token: str
    coherence: float
    halted: bool
    halt_reason: str = ""
    final: bool = False
    summary: dict[str, float] = field(default_factory=dict)


class _QueueCallback(TokenTraceCallback):
    """Push stream trace events into a thread-safe queue."""

    def __init__(self, queue: Queue[_LiveEvent]) -> None:
        self._queue = queue

    def on_token(self, event: TokenTraceEvent) -> None:
        """Record one kernel token event for the UI generator."""
        self._queue.put(
            _LiveEvent(
                index=event.index,
                token=event.token,
                coherence=event.coherence,
                halted=event.halted,
                halt_reason=event.halt_reason,
            )
        )

    def on_stream_end(
        self, *, tenant_id: str, request_id: str, summary: dict[str, Any]
    ) -> None:
        """Record the terminal stream summary for the UI generator."""
        del tenant_id, request_id
        self._queue.put(
            _LiveEvent(
                index=-1,
                token=_FINAL_EVENT_TEXT,
                coherence=_summary_float(summary, "avg_coherence"),
                halted=bool(summary.get("halted", False)),
                halt_reason=str(summary.get("halt_reason", "")),
                final=True,
                summary={
                    "token_count": _summary_float(summary, "token_count"),
                    "avg_coherence": _summary_float(summary, "avg_coherence"),
                    "warning_count": _summary_float(summary, "warning_count"),
                },
            )
        )


_SCORE_COLOURS = (
    (0.70, "#dcfce7", "#166534"),
    (0.55, "#fef9c3", "#854d0e"),
    (0.00, "#fecaca", "#991b1b"),
)
_FINAL_EVENT_TEXT = ""


def token_colour(coherence: float) -> tuple[str, str]:
    """Return ``(background, foreground)`` hex colours for a score.

    Parameters
    ----------
    coherence:
        Coherence score from the streaming kernel.

    Returns
    -------
    tuple[str, str]
        Background and foreground colour values for the score band.
    """
    for cutoff, background, foreground in _SCORE_COLOURS:
        if coherence >= cutoff:
            return background, foreground
    _, background, foreground = _SCORE_COLOURS[-1]
    return background, foreground


def render_token_span(token: str, coherence: float, halted: bool) -> str:
    """Render one escaped token badge for the live token strip.

    Parameters
    ----------
    token:
        Raw token text emitted by the simulated model stream.
    coherence:
        Kernel coherence score for the token.
    halted:
        Whether this token tripped the halt condition.

    Returns
    -------
    str
        Safe HTML span ready for Gradio's HTML component.
    """
    background, foreground = token_colour(coherence)
    decoration = "line-through" if halted else "none"
    display = html.escape(token, quote=True).replace(" ", "&nbsp;")
    return (
        f"<span style='background:{background};color:{foreground};"
        f"text-decoration:{decoration};padding:2px 6px;"
        f"border-radius:4px;margin:2px;display:inline-block;"
        f"font-family:ui-monospace,monospace;font-size:14px' "
        f"title='score={coherence:.3f}'>{display}</span>"
    )


def render_banner(
    halted: bool,
    halt_reason: str,
    summary: Mapping[str, object],
) -> str:
    """Render the escaped terminal verdict banner.

    Parameters
    ----------
    halted:
        Whether the stream halted before normal completion.
    halt_reason:
        Raw halt reason reported by the kernel.
    summary:
        Numeric terminal stream metrics.

    Returns
    -------
    str
        Safe HTML banner for the verdict panel.
    """
    token_count = int(_summary_float(summary, "token_count"))
    average = _summary_float(summary, "avg_coherence")
    warning_count = int(_summary_float(summary, "warning_count"))
    if halted:
        safe_reason = html.escape(
            halt_reason or "coherence floor reached",
            quote=True,
        )
        return (
            "<div style='background:#ef4444;color:white;padding:12px;"
            "border-radius:8px;font-weight:bold;font-size:1.2em'>"
            f"HALTED - {safe_reason}<br>"
            "<span style='font-weight:normal;font-size:0.85em'>"
            f"tokens emitted: {token_count}, "
            f"avg coherence: {average:.3f}, "
            f"warnings: {warning_count}"
            "</span></div>"
        )
    return (
        "<div style='background:#22c55e;color:white;padding:12px;"
        "border-radius:8px;font-weight:bold;font-size:1.2em'>"
        "APPROVED - stream completed<br>"
        "<span style='font-weight:normal;font-size:0.85em'>"
        f"tokens: {token_count}, "
        f"avg coherence: {average:.3f}, "
        f"warnings: {warning_count}"
        "</span></div>"
    )


def wrap_token_strip(html_fragment: str) -> str:
    """Wrap escaped token HTML in the demo strip container.

    Parameters
    ----------
    html_fragment:
        Already escaped token span markup.

    Returns
    -------
    str
        HTML container used by the live demo.
    """
    return (
        "<div style='line-height:2.2;padding:12px;"
        "background:#f9fafb;border-radius:8px;"
        "min-height:80px'>"
        f"{html_fragment}</div>"
    )


def render_gauge(coherence: float) -> str:
    """Render the coherence gauge for the latest token score.

    Parameters
    ----------
    coherence:
        Latest coherence score. Values outside ``[0, 1]`` are clamped for the
        visual width while preserving the displayed score.

    Returns
    -------
    str
        Safe HTML gauge for Gradio's HTML component.
    """
    percent = int(max(0.0, min(1.0, coherence)) * 100)
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
        f"border-radius:6px;width:{percent}%;transition:width 0.15s'>"
        "</div></div>"
        f"<div style='text-align:right;font-family:monospace;"
        f"font-size:12px;margin-top:4px'>"
        f"last score: {coherence:.3f}</div>"
    )


def paced_tokens(scenario: Scenario, delay_s: float) -> Iterable[str]:
    """Yield scenario tokens at a fixed non-negative cadence.

    Parameters
    ----------
    scenario:
        Scenario whose token stream should be emitted.
    delay_s:
        Delay between emitted tokens. Negative values are treated as zero.

    Returns
    -------
    Iterable[str]
        Token stream consumed by :class:`StreamingKernel`.
    """
    delay = max(0.0, delay_s)
    for token in scenario.tokens:
        yield token
        time.sleep(delay)


def score_callback_for(scenario: Scenario) -> CoherenceCallback:
    """Build a deterministic score callback for one scenario.

    Parameters
    ----------
    scenario:
        Scenario whose score curve should drive the streaming kernel.

    Returns
    -------
    CoherenceCallback
        Callback matching ``StreamingKernel.stream_tokens``.
    """
    score_iter = iter(scenario.scores)
    last = 0.5

    def _score(_accumulated: str) -> float:
        nonlocal last
        with contextlib.suppress(StopIteration):
            last = next(score_iter)
        return last

    return _score


def run_live_demo(scenario_key: str, speed_s: float) -> Iterator[Frame]:
    """Stream one demo scenario through the real kernel and yield UI frames.

    Parameters
    ----------
    scenario_key:
        Key in :data:`SCENARIOS`.
    speed_s:
        Per-token delay in seconds.

    Yields
    ------
    tuple[str, str, str]
        Token strip HTML, coherence gauge HTML, and verdict banner HTML.

    Raises
    ------
    ValueError
        If ``scenario_key`` is not in :data:`SCENARIOS`.
    """
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
            paced_tokens(scenario, speed_s),
            coherence_callback=score_callback_for(scenario),
            trace_callbacks=[callback],
            tenant_id="demo",
            request_id="live",
        )

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    spans: list[str] = []
    banner_html = render_banner(False, "", {})
    gauge_html = render_gauge(0.5)
    last_coherence = 0.5

    while True:
        try:
            event = queue.get(timeout=5.0)
        except Empty:  # pragma: no cover - defensive worker-loss timeout
            break
        if event.final:
            banner_html = render_banner(event.halted, event.halt_reason, event.summary)
            yield wrap_token_strip("".join(spans)), gauge_html, banner_html
            break
        spans.append(render_token_span(event.token, event.coherence, event.halted))
        last_coherence = event.coherence
        gauge_html = render_gauge(last_coherence)
        if event.halted:
            banner_html = render_banner(
                True,
                event.halt_reason,
                {
                    "token_count": event.index + 1,
                    "avg_coherence": last_coherence,
                    "warning_count": 0,
                },
            )
        yield wrap_token_strip("".join(spans)), gauge_html, banner_html

    thread.join(timeout=1.0)


def _summary_float(
    summary: Mapping[str, object],
    key: str,
    default: float = 0.0,
) -> float:
    """Return one numeric stream summary field as ``float``."""
    value = summary.get(key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            return float(value)
    return default
