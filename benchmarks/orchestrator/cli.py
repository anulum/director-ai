# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — orchestrator CLI entry point

"""``python -m benchmarks.orchestrator`` driver.

Examples::

    # Default suite, write to benchmarks/results/<commit>/
    python -m benchmarks.orchestrator

    # Only one case
    python -m benchmarks.orchestrator --only rust_parity_safety

    # Regression check against a stored baseline
    python -m benchmarks.orchestrator \\
        --baseline benchmarks/results/baseline.json

    # Vertex AI custom-job mode: set runner, write to GCS
    python -m benchmarks.orchestrator \\
        --runner vertex --output-dir /gcs/<bucket>/<run>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .cases import default_cases, pytest_coverage_case
from .environment import capture_environment
from .regression import (
    RegressionReport,
    default_rules,
    detect_regressions,
    load_baseline,
)
from .schema import RunReport
from .suite import Suite, SuiteRunner

logger = logging.getLogger("DirectorAI.Orchestrator.CLI")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmarks.orchestrator",
        description=(
            "Systematic benchmark and E2E runner for Director-AI. "
            "Produces a single JSON report per run and, when a "
            "baseline is supplied, a regression diff."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Where to write the RunReport and regression diff.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline RunReport JSON to regress against.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only the named cases (by SuiteCase.name).",
    )
    parser.add_argument(
        "--with-coverage",
        action="store_true",
        help="Append the pytest coverage case (slow).",
    )
    parser.add_argument(
        "--runner",
        choices=("local", "vertex", "ci", "remote"),
        default="local",
        help=("How this run is being invoked; overrides the DIRECTOR_RUN_ENV env var."),
    )
    parser.add_argument(
        "--report-name",
        default="run_report.json",
        help="Filename for the RunReport JSON (inside output-dir).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero on any failed case or any regression finding (CI gate mode)."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase logging verbosity (-v INFO, -vv DEBUG).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    cases = default_cases()
    if args.with_coverage:
        cases.append(pytest_coverage_case())
    suite = Suite(cases=cases)
    if args.only:
        suite = suite.filter(args.only)

    environment = capture_environment(runner=args.runner)
    runner_obj = SuiteRunner(suite, environment=environment)
    report = runner_obj.run()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / args.report_name
    report.to_file(report_path)
    logger.info("wrote run report → %s", report_path)

    regression: RegressionReport | None = None
    if args.baseline is not None:
        baseline = load_baseline(args.baseline)
        rules = default_rules()
        if args.only:
            rules = tuple(r for r in rules if r.case_name in set(args.only))
        regression = detect_regressions(report, baseline, rules)
        regression_path = output_dir / "regression.json"
        regression.to_file(regression_path)
        logger.info(
            "wrote regression diff → %s (clean=%s, findings=%d, skipped=%d)",
            regression_path,
            regression.clean,
            len(regression.findings),
            len(regression.skipped_rules),
        )

    _print_summary(report, regression)

    if args.strict:
        if not report.all_passed:
            logger.error("strict mode: %d case(s) failed", len(report.failed_entries))
            return 1
        if regression is not None and regression.high_severity:
            logger.error(
                "strict mode: %d high-severity regression(s)",
                len(regression.high_severity),
            )
            return 2
    return 0


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def _print_summary(report: RunReport, regression: RegressionReport | None) -> None:
    print()
    print("=" * 72)
    print(f"  run_id       : {report.run_id}")
    print(f"  timestamp    : {report.timestamp_utc}")
    print(f"  commit       : {report.environment.git_commit[:12]}")
    print(f"  package_ver  : {report.environment.package_version}")
    print(f"  hardware     : {report.environment.cpu_model}")
    print(f"                 cpu={report.environment.cpu_count}")
    print(f"                 ram={report.environment.ram_gb}GB")
    if report.environment.gpu_count:
        print(
            f"                 gpu={report.environment.gpu_model} "
            f"x{report.environment.gpu_count} "
            f"({report.environment.gpu_memory_gb}GB)",
        )
    print(f"  runner       : {report.environment.runner}")
    print("-" * 72)
    for entry in report.entries:
        tag = {
            "passed": "OK  ",
            "failed": "FAIL",
            "skipped": "SKIP",
            "warned": "WARN",
        }.get(entry.status, "??  ")
        primary_metric = entry.metrics[0].value if entry.metrics else float("nan")
        print(
            f"  [{tag}] {entry.name:<40} "
            f"({entry.kind}, {entry.wall_clock_seconds:.2f}s, "
            f"m0={primary_metric:.4g})"
        )
        if entry.notes and entry.status != "passed":
            for line in entry.notes.splitlines()[:3]:
                print(f"           {line}")
    print("-" * 72)
    total = len(report.entries)
    failed = sum(1 for e in report.entries if e.status == "failed")
    warned = sum(1 for e in report.entries if e.status == "warned")
    skipped = sum(1 for e in report.entries if e.status == "skipped")
    print(
        f"  {total - failed - warned - skipped}/{total} passed, "
        f"{warned} warned, {skipped} skipped, {failed} failed"
    )
    if regression is not None:
        print(
            f"  regressions: {len(regression.findings)} finding(s), "
            f"{len(regression.skipped_rules)} skipped rule(s), "
            f"clean={regression.clean}"
        )
        for finding in regression.findings:
            print(
                f"    - {finding.rule.case_name}.{finding.rule.metric} "
                f"({finding.severity}): {finding.reason}"
            )
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
