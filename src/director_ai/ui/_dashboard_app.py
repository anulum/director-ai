# SPDX-License-Identifier: BUSL-1.1
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - safety dashboard Gradio app

"""Gradio application shell for the safety operations dashboard.

Owns the interactive UI only: the input widgets, threshold sliders, and
result tables wired to the tenant-safe builders. The report models live
in :mod:`._dashboard_reports`, the analytics in
:mod:`._dashboard_analytics`, and the builders in
:mod:`.safety_dashboard` — imported lazily at launch time, matching the
optional Gradio dependency.
"""

from __future__ import annotations

from ._dashboard_reports import EVIDENCE_COLUMNS, SOURCE_COLUMNS, TENANT_COLUMNS

__all__ = ["launch_safety_dashboard"]


def launch_safety_dashboard(port: int = 7861, share: bool = False) -> None:
    """Launch the Gradio safety operations dashboard."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "Safety dashboard requires Gradio. Install with: pip install director-ai[ui]"
        ) from exc

    from .safety_dashboard import (
        build_observability_operations_markdown,
        build_safety_dashboard,
    )

    with gr.Blocks(title="Director-AI Safety Operations") as demo:
        gr.Markdown("# Director-AI Safety Operations")

        events = gr.Textbox(label="SafetyEvent JSONL", lines=14)
        feedback = gr.Textbox(label="Feedback JSONL", lines=8)
        with gr.Row():
            halt_threshold = gr.Slider(
                label="Halt-rate alert threshold",
                minimum=0.0,
                maximum=1.0,
                value=0.15,
                step=0.01,
            )
            fp_threshold = gr.Slider(
                label="False-positive alert threshold",
                minimum=0.0,
                maximum=1.0,
                value=0.05,
                step=0.01,
            )
            drift_threshold = gr.Slider(
                label="Drift alert threshold",
                minimum=0.0,
                maximum=1.0,
                value=0.10,
                step=0.01,
            )

        render = gr.Button("Render Dashboard", variant="primary")
        summary = gr.Markdown()
        tenants = gr.Dataframe(headers=TENANT_COLUMNS, label="Tenant halt rates")
        sources = gr.Dataframe(headers=SOURCE_COLUMNS, label="Contradiction sources")
        evidence = gr.Dataframe(headers=EVIDENCE_COLUMNS, label="Recent halt evidence")
        retune = gr.Code(label="Retune command", language="shell")
        operations = gr.Markdown(label="Observability operations report")

        render.click(
            fn=build_safety_dashboard,
            inputs=[events, feedback, halt_threshold, fp_threshold],
            outputs=[summary, tenants, sources, evidence, retune],
        )
        render.click(
            fn=build_observability_operations_markdown,
            inputs=[
                events,
                feedback,
                halt_threshold,
                fp_threshold,
                drift_threshold,
            ],
            outputs=[operations],
        )

    demo.launch(server_port=port, share=share)
