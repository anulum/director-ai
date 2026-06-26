# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""director-ai interactive CLI commands (kb-health, wizard, safety dashboard)."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def _cmd_kb_health(args: list[str]) -> None:
    """Run knowledge base health diagnostics."""
    if args and args[0] in ("-h", "--help", "help"):
        _print_kb_health_help()
        return

    from director_ai.core.config import DirectorConfig
    from director_ai.core.retrieval.kb_health import KBHealthCheck

    cfg = DirectorConfig.from_env()
    store = cfg.build_store()

    min_docs = 1
    max_latency = 100.0
    i = 0
    while i < len(args):
        if args[i] == "--min-docs" and i + 1 < len(args):
            min_docs = int(args[i + 1])
            i += 2
        elif args[i] == "--max-latency" and i + 1 < len(args):
            max_latency = float(args[i + 1])
            i += 2
        else:
            i += 1

    check = KBHealthCheck(
        store,
        min_documents=min_docs,
        max_query_latency_ms=max_latency,
    )
    report = check.run()

    print(report.summary)
    if report.issues:
        for issue in report.issues:
            print(f"  ISSUE: {issue}")
    if report.warnings:
        for warn in report.warnings:
            print(f"  WARNING: {warn}")

    sys.exit(0 if report.healthy else 1)


def _cmd_wizard(args: list[str]) -> None:
    """Launch the interactive configuration wizard."""
    if args and args[0] in ("-h", "--help", "help"):
        _print_wizard_help()
        return

    cli_mode = "--cli" in args
    port = 7860
    share = "--share" in args
    output_path = None

    i = 0
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_path = args[i + 1]
            i += 2
        else:
            i += 1

    from director_ai.ui.config_wizard import launch_cli, launch_gradio

    if cli_mode:
        yaml_str = launch_cli()
        if output_path:
            from pathlib import Path

            Path(output_path).write_text(yaml_str, encoding="utf-8")
            print(f"\nConfig written to {output_path}")
    else:
        try:
            launch_gradio(port=port, share=share)
        except ImportError:
            print("Gradio not installed. Using CLI mode instead.")
            print("Install with: pip install director-ai[ui]\n")
            yaml_str = launch_cli()
            if output_path:
                from pathlib import Path

                Path(output_path).write_text(yaml_str, encoding="utf-8")
                print(f"\nConfig written to {output_path}")


def _cmd_safety_dashboard(args: list[str]) -> None:
    """Launch or render the safety operations dashboard."""
    if args and args[0] in ("-h", "--help", "help"):
        _print_safety_dashboard_help()
        return

    port = 7861
    share = "--share" in args
    text_mode = "--text" in args
    events_path = None
    feedback_path = None
    halt_threshold = 0.15
    false_positive_threshold = 0.05

    i = 0
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--events" and i + 1 < len(args):
            events_path = args[i + 1]
            i += 2
        elif args[i] == "--feedback" and i + 1 < len(args):
            feedback_path = args[i + 1]
            i += 2
        elif args[i] == "--halt-alert-threshold" and i + 1 < len(args):
            halt_threshold = float(args[i + 1])
            i += 2
        elif args[i] == "--false-positive-alert-threshold" and i + 1 < len(args):
            false_positive_threshold = float(args[i + 1])
            i += 2
        else:
            i += 1

    if text_mode or events_path or feedback_path:
        from pathlib import Path

        from director_ai.ui.safety_dashboard import build_safety_dashboard

        events_jsonl = (
            Path(events_path).read_text(encoding="utf-8") if events_path else ""
        )
        feedback_jsonl = (
            Path(feedback_path).read_text(encoding="utf-8") if feedback_path else ""
        )
        summary, tenants, sources, evidence, command = build_safety_dashboard(
            events_jsonl,
            feedback_jsonl,
            halt_threshold,
            false_positive_threshold,
        )
        print(summary)
        print("\nTenant halt rates:")
        for row in tenants:
            print("  " + " | ".join(str(value) for value in row))
        print("\nTop contradiction sources:")
        for row in sources:
            print("  " + " | ".join(str(value) for value in row))
        print("\nRecent halt evidence:")
        for row in evidence:
            print("  " + " | ".join(str(value) for value in row))
        print(f"\nRetune: {command}")
        return

    from director_ai.ui.safety_dashboard import launch_safety_dashboard

    try:
        launch_safety_dashboard(port=port, share=share)
    except ImportError:
        print(
            "Gradio not installed. Use --text or install with: pip install director-ai[ui]"
        )


def _print_kb_health_help() -> None:
    """Print knowledge-base health options without opening the configured store."""
    print(
        "Usage: director-ai kb-health [options]\n"
        "\n"
        "Run knowledge-base document-count and retrieval-latency diagnostics.\n"
        "\n"
        "Options:\n"
        "  --min-docs N           Minimum indexed documents required (default: 1)\n"
        "  --max-latency MS       Maximum average query latency in milliseconds\n"
    )


def _print_wizard_help() -> None:
    """Print configuration wizard options without launching an interface."""
    print(
        "Usage: director-ai wizard [options]\n"
        "\n"
        "Launch the configuration wizard or emit a CLI-generated config file.\n"
        "\n"
        "Options:\n"
        "  --cli                  Run the terminal wizard instead of Gradio\n"
        "  --output FILE          Write generated YAML when using CLI fallback\n"
        "  --port N               Gradio port (default: 7860)\n"
        "  --share                Enable Gradio sharing\n"
    )


def _print_safety_dashboard_help() -> None:
    """Print safety dashboard options without launching UI dependencies."""
    print(
        "Usage: director-ai safety-dashboard [options]\n"
        "\n"
        "Launch the safety operations dashboard or render a text summary.\n"
        "\n"
        "Options:\n"
        "  --text                 Render the dashboard summary in the terminal\n"
        "  --events FILE          Safety events JSONL input for text mode\n"
        "  --feedback FILE        Feedback JSONL input for text mode\n"
        "  --port N               Gradio port (default: 7861)\n"
        "  --share                Enable Gradio sharing\n"
        "  --halt-alert-threshold FLOAT\n"
        "                         Halt-rate alert threshold (default: 0.15)\n"
        "  --false-positive-alert-threshold FLOAT\n"
        "                         False-positive alert threshold (default: 0.05)\n"
    )
