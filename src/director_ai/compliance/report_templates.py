# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HTML/Markdown report templates
"""Report templates for compliance, cost, and swarm health reports.

Generates HTML or Markdown reports from structured data (dicts).
Uses string formatting — no Jinja2 dependency required for basic
templates. Jinja2 is optional for advanced customisation.

Usage::

    from director_ai.compliance.report_templates import render_compliance_html

    html = render_compliance_html({
        "title": "EU AI Act Article 15 Report",
        "period": "2026-04-01 to 2026-04-13",
        "hallucination_rate": 0.042,
        "total_reviews": 15000,
        ...
    })
"""

from __future__ import annotations

import html as html_mod
from datetime import UTC, datetime

__all__ = [
    "render_compliance_html",
    "render_compliance_markdown",
    "render_cost_html",
    "render_swarm_html",
]

_CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
  h1 { border-bottom: 2px solid #2563eb; padding-bottom: 0.5rem; }
  h2 { color: #2563eb; margin-top: 2rem; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }
  th { background: #f3f4f6; font-weight: 600; }
  tr:nth-child(even) { background: #f9fafb; }
  .metric { font-size: 2rem; font-weight: 700; color: #2563eb; }
  .metric-label { font-size: 0.875rem; color: #6b7280; }
  .metric-card { display: inline-block; padding: 1rem 1.5rem;
                 border: 1px solid #e5e7eb; border-radius: 8px; margin: 0.5rem; }
  .alert { background: #fef2f2; border: 1px solid #fecaca;
           padding: 0.75rem; border-radius: 4px; color: #991b1b; }
  .ok { background: #f0fdf4; border: 1px solid #bbf7d0;
        padding: 0.75rem; border-radius: 4px; color: #166534; }
  footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;
           font-size: 0.75rem; color: #9ca3af; }
</style>
"""


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html_mod.escape(str(text))


def _pct(value: float) -> str:
    """Format as percentage string."""
    return f"{value * 100:.2f}%"


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


def render_compliance_html(data: dict) -> str:
    """Render EU AI Act Article 15 compliance report as HTML.

    Expected ``data`` keys:
        title, period, hallucination_rate, total_reviews,
        approved_count, rejected_count, avg_score, avg_latency_ms,
        drift_detected, models (list of dicts with model, reviews, rate).
    """
    models_html = ""
    for m in data.get("models", []):
        models_html += (
            f"<tr><td>{_esc(m.get('model', ''))}</td>"
            f"<td>{m.get('reviews', 0)}</td>"
            f"<td>{_pct(m.get('rate', 0))}</td>"
            f"<td>{m.get('avg_score', 0):.3f}</td></tr>\n"
        )

    drift_class = "alert" if data.get("drift_detected") else "ok"
    drift_text = "Drift DETECTED" if data.get("drift_detected") else "No drift"

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{_esc(data.get("title", "Compliance Report"))}</title>
{_CSS}</head>
<body>
<h1>{_esc(data.get("title", "EU AI Act Article 15 Report"))}</h1>
<p>Period: {_esc(data.get("period", "N/A"))} | Generated: {_now_iso()}</p>

<div>
  <div class="metric-card"><div class="metric">{data.get("total_reviews", 0):,}</div>
    <div class="metric-label">Total Reviews</div></div>
  <div class="metric-card"><div class="metric">{_pct(data.get("hallucination_rate", 0))}</div>
    <div class="metric-label">Hallucination Rate</div></div>
  <div class="metric-card"><div class="metric">{data.get("avg_latency_ms", 0):.1f} ms</div>
    <div class="metric-label">Avg Latency</div></div>
</div>

<div class="{drift_class}"><strong>{drift_text}</strong></div>

<h2>Per-Model Breakdown</h2>
<table>
<tr><th>Model</th><th>Reviews</th><th>Hallucination Rate</th><th>Avg Score</th></tr>
{models_html}</table>

<h2>Summary</h2>
<ul>
  <li>Approved: {data.get("approved_count", 0):,}</li>
  <li>Rejected: {data.get("rejected_count", 0):,}</li>
  <li>Avg coherence score: {data.get("avg_score", 0):.3f}</li>
</ul>

<footer>Director-AI Compliance Report | ANULUM Institute | {_now_iso()}</footer>
</body></html>"""


def render_compliance_markdown(data: dict) -> str:
    """Render compliance report as Markdown."""
    lines = [
        f"# {data.get('title', 'Compliance Report')}",
        "",
        f"**Period:** {data.get('period', 'N/A')}  ",
        f"**Generated:** {_now_iso()}",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Reviews | {data.get('total_reviews', 0):,} |",
        f"| Hallucination Rate | {_pct(data.get('hallucination_rate', 0))} |",
        f"| Avg Latency | {data.get('avg_latency_ms', 0):.1f} ms |",
        f"| Avg Score | {data.get('avg_score', 0):.3f} |",
        f"| Drift Detected | {'Yes' if data.get('drift_detected') else 'No'} |",
        "",
        "## Per-Model Breakdown",
        "",
        "| Model | Reviews | Hall. Rate | Avg Score |",
        "|-------|---------|------------|-----------|",
    ]
    for m in data.get("models", []):
        lines.append(
            f"| {m.get('model', '')} | {m.get('reviews', 0)} | "
            f"{_pct(m.get('rate', 0))} | {m.get('avg_score', 0):.3f} |"
        )
    lines.extend(["", "---", f"*Director-AI | ANULUM Institute | {_now_iso()}*"])
    return "\n".join(lines)


def render_cost_html(data: dict) -> str:
    """Render cost report as HTML.

    Expected ``data`` keys from ``CostAnalyser.report()``.
    """
    models_html = ""
    for key, m in data.get("models", {}).items():
        models_html += (
            f"<tr><td>{_esc(key)}</td>"
            f"<td>{m.get('call_count', 0)}</td>"
            f"<td>{m.get('total_tokens', 0):,}</td>"
            f"<td>{data.get('currency', 'CHF')} {m.get('estimated_cost', 0):.4f}</td></tr>\n"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Cost Report</title>
{_CSS}</head>
<body>
<h1>Token Cost Report</h1>
<p>Generated: {_now_iso()}</p>

<div>
  <div class="metric-card"><div class="metric">{data.get("currency", "CHF")} {data.get("total_cost", 0):.4f}</div>
    <div class="metric-label">Total Cost</div></div>
  <div class="metric-card"><div class="metric">{data.get("total_tokens", 0):,}</div>
    <div class="metric-label">Total Tokens</div></div>
</div>

<h2>Per-Model Breakdown</h2>
<table>
<tr><th>Model / Agent</th><th>Calls</th><th>Tokens</th><th>Cost</th></tr>
{models_html}</table>

<footer>Director-AI Cost Report | ANULUM Institute | {_now_iso()}</footer>
</body></html>"""


def render_swarm_html(data: dict) -> str:
    """Render swarm health report as HTML.

    Expected ``data`` keys from ``SwarmMetrics.report()``.
    """
    swarm = data.get("swarm", {})
    agents_html = ""
    for aid, a in data.get("agents", {}).items():
        q_class = "alert" if a.get("quarantined") else ""
        agents_html += (
            f"<tr class='{q_class}'><td>{_esc(aid)}</td>"
            f"<td>{a.get('handoffs', 0)}</td>"
            f"<td>{_pct(a.get('hallucination_rate', 0))}</td>"
            f"<td>{'Yes' if a.get('quarantined') else 'No'}</td></tr>\n"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Swarm Health Report</title>
{_CSS}</head>
<body>
<h1>Swarm Health Report</h1>
<p>Generated: {_now_iso()}</p>

<div>
  <div class="metric-card"><div class="metric">{swarm.get("active_agents", 0)}</div>
    <div class="metric-label">Active Agents</div></div>
  <div class="metric-card"><div class="metric">{swarm.get("total_handoffs", 0):,}</div>
    <div class="metric-label">Total Handoffs</div></div>
  <div class="metric-card"><div class="metric">{swarm.get("quarantined_agents", 0)}</div>
    <div class="metric-label">Quarantined</div></div>
  <div class="metric-card"><div class="metric">{swarm.get("cascade_events", 0)}</div>
    <div class="metric-label">Cascade Events</div></div>
</div>

<h2>Per-Agent Breakdown</h2>
<table>
<tr><th>Agent</th><th>Handoffs</th><th>Hall. Rate</th><th>Quarantined</th></tr>
{agents_html}</table>

<footer>Director-AI Swarm Report | ANULUM Institute | {_now_iso()}</footer>
</body></html>"""
