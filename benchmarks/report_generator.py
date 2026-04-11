#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Benchmark Report Generator
"""Generate a Markdown summary of the AggreFact and Rust-compute results.

This module loads numbers exclusively from the v2-schema JSON files
written by ``benchmarks.aggrefact_eval.score_and_save()``,
``benchmarks.gemma_aggrefact_routed`` (or any judge that emits the same
schema), and ``benchmarks/results/rust_compute_bench.json``.

Three rules the generator must obey:

1. **No hardcoded numbers.** Every cell in every table comes from a
   JSON file. If a JSON is missing, the section is skipped (with a
   note) — never substituted with an estimate.
2. **No fabricated competitor numbers.** External leaderboard rows
   (FaithLens, Paladin, VERDICT, MiniCheck, GPT-4o, …) are read from
   ``benchmarks/leaderboard_external.json``, which holds the cited
   numbers and their source URL. If the file is missing, the external
   table is omitted.
3. **No marketing language.** Anti-slop applies. Oracle ensemble
   metrics are explicitly labelled "theoretical upper bound", never
   "matches global SOTA".

Usage::

    python -m benchmarks.report_generator \\
        --results benchmarks/results \\
        --output reports/aggrefact_summary_$(date +%Y%m%d_%H%M%S).md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Result loaders ───────────────────────────────────────────────────────


def load_json(path: Path) -> dict[str, Any] | None:
    """Return the parsed JSON at ``path``, or ``None`` if missing/invalid."""
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def find_judge_results(results_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(label, payload)`` pairs for every AggreFact judge JSON.

    Recognises both the v2 ``aggrefact_eval`` schema and the routed
    Gemma schema. The label is taken from the ``model`` field, falling
    back to the file stem.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(results_dir.glob("*.json")):
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        if "global_balanced_accuracy" not in payload:
            continue
        if "labels" not in payload or "datasets_per_sample" not in payload:
            continue
        label = payload.get("model") or path.stem
        # Strip absolute paths from local model labels for readability.
        label = Path(str(label)).name or str(label)
        out.append((label, payload))
    return out


# ── Section formatters ───────────────────────────────────────────────────


def format_judge_table(
    judges: list[tuple[str, dict[str, Any]]],
) -> list[str]:
    """One row per judge JSON. All numbers come from the JSON itself."""
    if not judges:
        return ["_No AggreFact judge JSONs found in the results directory._"]
    rows = ["| Judge | Samples | Sample-pooled BA | Per-dataset avg BA | Method |",
            "|-------|---------|------------------|---------------------|--------|"]
    for label, payload in judges:
        samples = payload.get("samples", "—")
        global_ba = payload.get("global_balanced_accuracy")
        per_ds_avg = payload.get("per_dataset_avg_balanced_accuracy")
        method = payload.get("method", payload.get("backend", "—"))
        global_str = f"{global_ba:.2%}" if isinstance(global_ba, (int, float)) else "—"
        per_ds_str = (
            f"{per_ds_avg:.2%}"
            if isinstance(per_ds_avg, (int, float))
            else "—"
        )
        rows.append(
            f"| {label} | {samples} | {global_str} | {per_ds_str} | {method} |"
        )
    return rows


def format_rust_table(rust_payload: dict[str, Any] | None) -> list[str]:
    """Render the Rust compute benchmark table from rust_compute_bench.json."""
    if not rust_payload or "results" not in rust_payload:
        return [
            "_No `rust_compute_bench.json` found — Rust acceleration "
            "section omitted._"
        ]
    iterations = rust_payload.get("iterations", "—")
    rust_available = rust_payload.get("rust_available", False)
    lines = [
        f"Iterations per function: **{iterations}**. Rust kernel "
        f"available: **{rust_available}**.",
        "",
        "| Function | Python median (µs) | Rust median (µs) | Speedup |",
        "|----------|--------------------:|------------------:|--------:|",
    ]
    for fn_name, fn in rust_payload["results"].items():
        if not isinstance(fn, dict):
            continue
        py = fn.get("python", {}).get("median_us")
        rs = fn.get("rust", {}).get("median_us")
        speedup = fn.get("speedup")
        py_str = f"{py:.2f}" if isinstance(py, (int, float)) else "—"
        rs_str = f"{rs:.2f}" if isinstance(rs, (int, float)) else "—"
        sp_str = (
            f"{speedup:.1f}×"
            if isinstance(speedup, (int, float))
            else "—"
        )
        lines.append(f"| {fn_name} | {py_str} | {rs_str} | {sp_str} |")
    return lines


def format_external_leaderboard(
    leaderboard: list[dict[str, Any]] | None,
) -> list[str]:
    """Render the external leaderboard from a hand-curated JSON file.

    Each entry must include ``system``, ``params``, ``balanced_accuracy``,
    ``source_url``. No estimates, no tilde-prefixed numbers.
    """
    if not leaderboard:
        return [
            "_No external leaderboard JSON provided — external comparison "
            "table omitted. Add `benchmarks/leaderboard_external.json` "
            "with verified rows to populate this section._"
        ]
    rows = [
        "| System | Params | AggreFact BA | Source |",
        "|--------|--------|-------------:|--------|",
    ]
    for entry in leaderboard:
        system = entry.get("system", "—")
        params = entry.get("params", "—")
        ba = entry.get("balanced_accuracy")
        source = entry.get("source_url", "—")
        ba_str = f"{ba:.2%}" if isinstance(ba, (int, float)) else "—"
        rows.append(f"| {system} | {params} | {ba_str} | {source} |")
    return rows


def format_oracle_section(
    sentinel_payload: dict[str, Any] | None,
) -> list[str]:
    """Render the ensemble Oracle metric clearly labelled as upper bound."""
    if not sentinel_payload:
        return [
            "_No sentinel-judge analyser output available. Run "
            "`benchmarks.sentinel_judge_analyzer` once two or more "
            "judges have completed._"
        ]
    voting = sentinel_payload.get("voting_balanced_accuracy")
    routed = sentinel_payload.get("routed_balanced_accuracy")
    fusion = sentinel_payload.get("fusion_balanced_accuracy")
    oracle = sentinel_payload.get("oracle_balanced_accuracy")
    notes = [
        "Sentinel-judge ensemble metrics. **Oracle is the theoretical "
        "upper bound** of any router with perfect knowledge of which "
        "judge is correct on each sample — it is **not** a deployable "
        "system and **not** comparable to single-system leaderboard "
        "results.",
        "",
        "| Strategy | Balanced accuracy |",
        "|----------|------------------:|",
    ]
    for label, value in (
        ("Majority voting", voting),
        ("Per-family routing", routed),
        ("Logistic-regression fusion", fusion),
        ("Oracle (upper bound)", oracle),
    ):
        if isinstance(value, (int, float)):
            notes.append(f"| {label} | {value:.2%} |")
    return notes


# ── Top-level driver ─────────────────────────────────────────────────────


def generate_report(
    results_dir: Path,
    output_path: Path,
    *,
    leaderboard_path: Path | None = None,
    sentinel_path: Path | None = None,
) -> Path:
    """Write the Markdown summary to ``output_path`` and return the path."""
    judges = find_judge_results(results_dir)
    rust = load_json(results_dir / "rust_compute_bench.json")
    external_payload = (
        load_json(leaderboard_path) if leaderboard_path else None
    )
    external = (
        external_payload.get("entries")
        if isinstance(external_payload, dict)
        else None
    )
    sentinel = (
        load_json(sentinel_path) if sentinel_path else None
    )

    lines: list[str] = []
    lines.append("# Director-AI — AggreFact & Compute Benchmark Summary")
    lines.append("")
    lines.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
        f"(local clock)"
    )
    lines.append(
        f"Source: every number in this document is read from a JSON file "
        f"under `{results_dir}`. No values are hardcoded by the generator."
    )
    lines.append("")
    lines.append("## 1. Director-AI judges on AggreFact 29 K")
    lines.append("")
    lines.extend(format_judge_table(judges))
    lines.append("")
    lines.append("## 2. Sentinel-judge ensemble")
    lines.append("")
    lines.extend(format_oracle_section(sentinel))
    lines.append("")
    lines.append("## 3. Rust compute acceleration")
    lines.append("")
    lines.extend(format_rust_table(rust))
    lines.append("")
    lines.append("## 4. External leaderboard (verified rows only)")
    lines.append("")
    lines.extend(format_external_leaderboard(external))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "_Generated by `benchmarks.report_generator`. To add a new "
        "judge, drop its v2-schema JSON into the results directory and "
        "regenerate. To add an external leaderboard row, edit "
        "`benchmarks/leaderboard_external.json` with `system`, "
        "`params`, `balanced_accuracy`, and `source_url` — no "
        "estimates, no tilde-prefixed numbers._"
    )
    lines.append("")
    lines.append(
        "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory of v2-schema judge JSONs and rust_compute_bench.json",
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=Path("benchmarks/leaderboard_external.json"),
        help="Optional JSON file with verified external leaderboard rows",
    )
    parser.add_argument(
        "--sentinel",
        type=Path,
        default=None,
        help="Optional sentinel-judge analyser output JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            f"reports/aggrefact_summary_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        ),
    )
    args = parser.parse_args()

    leaderboard = (
        args.leaderboard if args.leaderboard.is_file() else None
    )
    out = generate_report(
        args.results,
        args.output,
        leaderboard_path=leaderboard,
        sentinel_path=args.sentinel,
    )
    print(f"Report generated: {out}")
