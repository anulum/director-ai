#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Full benchmark campaign runner

"""Run a full reproducible benchmark campaign and publish one packet.

This runner orchestrates end-to-end benchmark domains in one pass:
- Rust vs Python path comparisons,
- latency and load,
- retrieval quality,
- guardrail E2E quality,
- domain and truthfulness benchmark slices.

Outputs:
- `benchmarks/results/full_benchmark_campaign_<UTC>.json`
- `benchmarks/results/full_benchmark_campaign_<UTC>.md`
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks._provenance import resolve_git_sha

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmarks" / "results"


@dataclass(frozen=True)
class CampaignCase:
    name: str
    command: list[str]
    timeout_s: int
    category: str


@dataclass(frozen=True)
class CaseResult:
    name: str
    category: str
    status: str
    return_code: int
    duration_s: float
    command: list[str]
    artifact_files: list[str]
    stdout_tail: str
    stderr_tail: str


def _campaign_cases() -> list[CampaignCase]:
    return [
        CampaignCase(
            name="rust_python_e2e_compare",
            category="rust_vs_python",
            timeout_s=3600,
            command=[
                sys.executable,
                "-m",
                "benchmarks.rust_python_e2e_compare",
                "--iterations",
                "200",
                "--warmup",
                "30",
            ],
        ),
        CampaignCase(
            name="rust_compute_bench",
            category="rust_vs_python",
            timeout_s=1800,
            command=[
                sys.executable,
                "-m",
                "benchmarks.rust_compute_bench",
                "--iterations",
                "5000",
            ],
        ),
        CampaignCase(
            name="e2e_guardrail_compare",
            category="e2e_quality",
            timeout_s=3600,
            command=[
                sys.executable,
                "-m",
                "benchmarks.e2e_eval",
                "--max-samples",
                "200",
                "--compare",
                "--scorer-backend",
                "lite",
                "--output-json",
                str(RESULTS_DIR / "e2e_comparison_full_campaign.json"),
            ],
        ),
        CampaignCase(
            name="retrieval_bench_inmemory",
            category="retrieval_quality",
            timeout_s=1200,
            command=[
                sys.executable,
                "-m",
                "benchmarks.retrieval_bench",
                "--backend",
                "inmemory",
            ],
        ),
        CampaignCase(
            name="latency_bench",
            category="latency",
            timeout_s=1800,
            command=[
                sys.executable,
                "-m",
                "benchmarks.latency_bench",
                "--iterations",
                "100",
                "--warmup",
                "20",
            ],
        ),
        CampaignCase(
            name="load_test",
            category="load",
            timeout_s=1800,
            command=[
                sys.executable,
                "-m",
                "benchmarks.load_test",
                "--concurrency",
                "8",
                "--duration",
                "30",
            ],
        ),
        CampaignCase(
            name="truthfulqa_eval",
            category="truthfulness",
            timeout_s=3600,
            command=[
                sys.executable,
                "-m",
                "benchmarks.truthfulqa_eval",
                "200",
                "--no-nli",
            ],
        ),
        CampaignCase(
            name="halueval_eval",
            category="truthfulness",
            timeout_s=3600,
            command=[
                sys.executable,
                "-m",
                "benchmarks.halueval_eval",
                "300",
                "--no-nli",
            ],
        ),
        CampaignCase(
            name="ragtruth_eval",
            category="truthfulness",
            timeout_s=3600,
            command=[
                sys.executable,
                "-m",
                "benchmarks.ragtruth_eval",
                "--max-samples",
                "500",
            ],
        ),
        CampaignCase(
            name="finance_eval",
            category="domain_quality",
            timeout_s=3600,
            command=[
                sys.executable,
                "-m",
                "benchmarks.finance_eval",
                "--dataset",
                "all",
                "--max-samples",
                "500",
            ],
        ),
        CampaignCase(
            name="legal_eval",
            category="domain_quality",
            timeout_s=3600,
            command=[
                sys.executable,
                "-m",
                "benchmarks.legal_eval",
                "--dataset",
                "all",
                "--max-samples",
                "500",
            ],
        ),
        CampaignCase(
            name="medical_eval",
            category="domain_quality",
            timeout_s=3600,
            command=[
                sys.executable,
                "-m",
                "benchmarks.medical_eval",
                "--dataset",
                "all",
                "--max-samples",
                "500",
            ],
        ),
    ]


def _result_snapshot() -> set[Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return {p.resolve() for p in RESULTS_DIR.glob("**/*") if p.is_file()}


def _run_case(case: CampaignCase) -> CaseResult:
    before = _result_snapshot()
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            case.command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=case.timeout_s,
            check=False,
        )
        status = "passed" if proc.returncode == 0 else "failed"
        return_code = proc.returncode
        stdout_tail = proc.stdout[-1200:]
        stderr_tail = proc.stderr[-1200:]
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        return_code = 124
        stdout_tail = (exc.stdout or "")[-1200:]
        stderr_tail = (exc.stderr or "")[-1200:]
    after = _result_snapshot()
    elapsed = time.perf_counter() - t0
    artifacts = sorted(str(p.relative_to(ROOT)) for p in (after - before))
    return CaseResult(
        name=case.name,
        category=case.category,
        status=status,
        return_code=return_code,
        duration_s=round(elapsed, 2),
        command=case.command,
        artifact_files=artifacts,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Full Benchmark Campaign",
        "",
        f"Generated (UTC): {payload['generated_utc']}",
        f"Commit: `{payload['git_commit']}`",
        f"Python: `{payload['python_version']}`",
        f"Platform: `{payload['platform']}`",
        "",
        "| Case | Category | Status | Duration (s) | Return code | Artifacts |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in payload["cases"]:
        lines.append(
            f"| `{row['name']}` | `{row['category']}` | `{row['status']}` | "
            f"{row['duration_s']:.2f} | {row['return_code']} | {len(row['artifact_files'])} |"
        )
    total = len(payload["cases"])
    passed = sum(1 for c in payload["cases"] if c["status"] == "passed")
    failed = sum(1 for c in payload["cases"] if c["status"] == "failed")
    timeout = sum(1 for c in payload["cases"] if c["status"] == "timeout")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Passed: **{passed}/{total}**",
            f"- Failed: **{failed}**",
            f"- Timeout: **{timeout}**",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "PYTHONPATH=src python -m benchmarks.full_benchmark_campaign",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_campaign(*, strict: bool = False) -> tuple[dict[str, Any], int]:
    cases = _campaign_cases()
    case_results = [_run_case(case) for case in cases]
    payload = {
        "benchmark": "full_benchmark_campaign",
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": resolve_git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cases": [asdict(result) for result in case_results],
    }
    if strict and any(r.status != "passed" for r in case_results):
        return payload, 2
    return payload, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    payload, exit_code = run_campaign(strict=args.strict)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"full_benchmark_campaign_{stamp}.json"
    md_path = RESULTS_DIR / f"full_benchmark_campaign_{stamp}.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Campaign JSON: {json_path}")
    print(f"Campaign Markdown: {md_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
