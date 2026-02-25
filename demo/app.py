#!/usr/bin/env python3
"""
Director-AI — Gradio demo for HuggingFace Spaces.

    pip install director-ai gradio
    python demo/app.py
"""

from __future__ import annotations

import gradio as gr

from director_ai.core import (
    CoherenceScorer,
    GroundTruthStore,
    StreamingKernel,
)


def score_response(
    facts_text: str,
    query: str,
    llm_response: str,
) -> tuple[str, str, str, str]:
    """Score an LLM response against user-provided facts."""
    store = GroundTruthStore()
    for line in facts_text.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        store.add(key.strip(), value.strip())

    scorer = CoherenceScorer(
        threshold=0.6,
        ground_truth_store=store,
        use_nli=False,
    )
    approved, score = scorer.review(query, llm_response)

    verdict = "PASS" if approved else "BLOCKED"
    colour = "#22c55e" if approved else "#ef4444"
    badge = (
        "<div style='text-align:center;padding:12px;"
        f"border-radius:8px;background:{colour};"
        "color:white;font-size:1.5em;font-weight:bold'>"
        f"{verdict}</div>"
    )

    details = (
        f"**Coherence score:** {score.score:.3f}\n\n"
        f"**H_logical:** {score.h_logical:.3f}\n\n"
        f"**H_factual:** {score.h_factual:.3f}\n\n"
        f"**Threshold:** 0.6"
    )

    bar_pct = int(score.score * 100)
    if score.score >= 0.6:
        bar_colour = "#22c55e"
    elif score.score >= 0.45:
        bar_colour = "#f59e0b"
    else:
        bar_colour = "#ef4444"
    bar = (
        "<div style='background:#e5e7eb;border-radius:6px;"
        "height:24px;width:100%'>"
        f"<div style='background:{bar_colour};height:100%;"
        f"border-radius:6px;width:{bar_pct}%;"
        "transition:width 0.4s'></div></div>"
    )

    ctx = store.retrieve_context(query) or "No matching facts found."
    return badge, details, bar, ctx


STREAMING_SCENARIOS = {
    "Truthful response (APPROVED)": {
        "tokens": [
            "Water",
            " boils",
            " at",
            " 100",
            " degrees",
            " Celsius",
            " (212",
            " F)",
            " at",
            " standard",
            " atmospheric",
            " pressure",
            ".",
        ],
        "scores": [
            0.92,
            0.90,
            0.88,
            0.91,
            0.93,
            0.90,
            0.88,
            0.87,
            0.89,
            0.91,
            0.90,
            0.89,
            0.90,
        ],
    },
    "Blatant hallucination (HARD LIMIT halt)": {
        "tokens": [
            "Water",
            " boils",
            " at",
            " 100",
            " degrees",
            " Celsius",
            ".",
            " But",
            " the",
            " real",
            " temperature",
            " is",
            " negative",
            " forty",
            " degrees",
            ".",
        ],
        "scores": [
            0.92,
            0.90,
            0.91,
            0.89,
            0.88,
            0.87,
            0.86,
            0.85,
            0.84,
            0.83,
            0.30,
            0.15,
            0.10,
            0.08,
            0.05,
            0.03,
        ],
    },
    "Gradual drift (TREND halt)": {
        "tokens": [
            "Water",
            " boils",
            " at",
            " 100",
            " C.",
            " However",
            " at",
            " high",
            " altitude",
            " it",
            " actually",
            " boils",
            " at",
            " only",
            " 50",
            " C,",
            " which",
            " means",
            " climbers",
            " can",
            " boil",
            " water",
            " with",
            " body",
            " heat",
            " alone",
            ".",
        ],
        "scores": [
            0.91,
            0.89,
            0.87,
            0.90,
            0.88,
            0.78,
            0.72,
            0.65,
            0.58,
            0.52,
            0.46,
            0.41,
            0.38,
            0.33,
            0.28,
            0.22,
            0.18,
            0.15,
            0.12,
            0.10,
            0.08,
            0.05,
            0.03,
            0.02,
            0.01,
            0.01,
            0.01,
        ],
    },
}


def run_streaming_demo(scenario_name: str) -> str:
    """Run a streaming halt scenario and return formatted output."""
    scenario = STREAMING_SCENARIOS[scenario_name]
    tokens = scenario["tokens"]
    scores = scenario["scores"]

    kernel = StreamingKernel(
        hard_limit=0.35,
        window_size=5,
        window_threshold=0.45,
        trend_window=4,
        trend_threshold=0.20,
    )

    idx = 0

    def coherence_cb(_tok: str) -> float:
        nonlocal idx
        s = scores[min(idx, len(scores) - 1)]
        idx += 1
        return s

    session = kernel.stream_tokens(iter(tokens), coherence_cb)

    lines = [
        "**Token-by-token trace:**\n\n| # | Token | Coherence | Status |",
        "|---|-------|-----------|--------|",
    ]
    for ev in session.events:
        if ev.coherence >= 0.6:
            status = "ok"
        elif ev.coherence >= 0.45:
            status = "warn"
        else:
            status = "**LOW**"
        if ev.halted:
            status = "**HALTED**"
        lines.append(f"| {ev.index} | `{ev.token}` | {ev.coherence:.3f} | {status} |")

    lines.append("")
    if session.halted:
        lines.append(f"**Result: HALTED** at token {session.halt_index}/{len(tokens)}")
        lines.append(f"\n**Reason:** `{session.halt_reason}`")
        lines.append(f"\n**Partial output:** {session.output}")
    else:
        lines.append("**Result: APPROVED**")
        lines.append(f"\n**Full output:** {session.output}")

    lines.append(
        f"\n**Stats:** avg={session.avg_coherence:.3f}, "
        f"min={session.min_coherence:.3f}, "
        f"tokens={session.token_count}/{len(tokens)}"
    )
    return "\n".join(lines)


CSS = """
.badge { font-size: 1.2em; font-weight: bold; }
footer { display: none !important; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Director-AI Demo", css=CSS, theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Director-AI\n"
            "**Real-time LLM hallucination guardrail** — "
            "NLI + RAG fact-checking with token-level streaming halt\n\n"
            "`pip install director-ai`"
        )

        with gr.Tab("Score a Response"):
            gr.Markdown("Enter your facts, a query, and an LLM response to check.")
            with gr.Row():
                with gr.Column():
                    default_facts = (
                        "sky color: The sky is blue due to "
                        "Rayleigh scattering.\n"
                        "boiling point: Water boils at 100 "
                        "degrees Celsius at sea level."
                    )
                    facts = gr.Textbox(
                        label="Knowledge base (key: value, one per line)",
                        lines=4,
                        value=default_facts,
                    )
                    query = gr.Textbox(label="Query", value="What color is the sky?")
                    response = gr.Textbox(
                        label="LLM response to check",
                        value="The sky is green, obviously.",
                    )
                    score_btn = gr.Button("Score", variant="primary")
                with gr.Column():
                    badge_html = gr.HTML(label="Verdict")
                    details_md = gr.Markdown(label="Details")
                    bar_html = gr.HTML(label="Score bar")
                    context_md = gr.Markdown(label="Retrieved context")

            score_btn.click(
                score_response,
                inputs=[facts, query, response],
                outputs=[badge_html, details_md, bar_html, context_md],
            )

            gr.Examples(
                examples=[
                    [
                        "sky color: The sky is blue due to Rayleigh scattering.",
                        "What color is the sky?",
                        "The sky is blue on a clear day.",
                    ],
                    [
                        "sky color: The sky is blue due to Rayleigh scattering.",
                        "What color is the sky?",
                        "The sky is green, obviously.",
                    ],
                    [
                        "capital: Paris is the capital of France.",
                        "What is the capital of France?",
                        "The capital of France is Berlin.",
                    ],
                    [
                        "boiling point: Water boils at 100 C.",
                        "At what temperature does water boil?",
                        "Water boils at 100 degrees Celsius.",
                    ],
                ],
                inputs=[facts, query, response],
            )

        with gr.Tab("Streaming Halt"):
            gr.Markdown(
                "Director-AI monitors coherence **token-by-token** and halts "
                "generation the moment it degrades. Three halt mechanisms:\n\n"
                "1. **Hard limit** — single token below threshold\n"
                "2. **Sliding window** — rolling average drops\n"
                "3. **Downward trend** — coherence decay over N tokens"
            )
            scenario = gr.Dropdown(
                label="Scenario",
                choices=list(STREAMING_SCENARIOS.keys()),
                value=list(STREAMING_SCENARIOS.keys())[0],
            )
            run_btn = gr.Button("Run Demo", variant="primary")
            result_md = gr.Markdown()

            run_btn.click(
                run_streaming_demo,
                inputs=[scenario],
                outputs=[result_md],
            )

        gr.Markdown(
            "---\n"
            "[GitHub](https://github.com/anulum/director-ai) | "
            "[PyPI](https://pypi.org/project/director-ai/) | "
            "AGPL-3.0"
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()
