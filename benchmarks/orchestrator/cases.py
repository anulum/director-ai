# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — orchestrator default cases

"""Concrete benchmark cases the orchestrator runs by default.

Each case is a thin wrapper around one existing measurement —
either a pytest subprocess, a direct timing loop, or a call into
the existing ``benchmarks/`` scripts. The thin-wrapper pattern
keeps this module honest: every number that shows up in the
orchestrator's report is produced by code that someone could run
independently.

Three categories shipped here:

* **smoke** — always-on invariants (package imports, Rust
  accelerator presence, agent construction) that catch entire-
  subsystem breakage in seconds.
* **pytest-backed** — runs a targeted subset of ``pytest`` with
  ``--tb=no --quiet --junit-xml`` and parses the XML for
  pass / fail / skip counts. Used for the Rust parity suite,
  the feature-matrix suite, and the injection adversarial suite.
* **latency** — direct timing of the scorer with warm-up and
  percentile reporting. The CPU case always runs; the GPU case
  skips when no CUDA device is available.

Accuracy cases (AggreFact macro, HaluEval QA, distilled sanity)
are defined in :mod:`.cases_accuracy` so this module stays lean
and import-free of heavy datasets.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

from .schema import MetricResult
from .suite import CaseOutput, SuiteCase

logger = logging.getLogger("DirectorAI.Orchestrator.Cases")


def default_cases() -> list[SuiteCase]:
    """Return the default suite (smoke + pytest + latency).

    Accuracy cases are added by the CLI when the operator passes
    ``--with-accuracy`` or sets ``DIRECTOR_BENCH_ACCURACY=1``;
    keeping them out of the default path means ``--only`` runs
    and CI smoke runs finish in under a minute.
    """
    return [
        SuiteCase(
            name="smoke_package_import", kind="smoke", call=_smoke_package_import
        ),
        SuiteCase(
            name="smoke_rust_accelerator", kind="smoke", call=_smoke_rust_accelerator
        ),
        SuiteCase(
            name="smoke_agent_construction",
            kind="smoke",
            call=_smoke_agent_construction,
        ),
        SuiteCase(
            name="smoke_safety_hooks_import",
            kind="smoke",
            call=_smoke_safety_hooks_import,
        ),
        SuiteCase(
            name="rust_parity_safety",
            kind="smoke",
            call=_pytest_case(["tests/test_rust_parity_safety.py"]),
        ),
        SuiteCase(
            name="feature_matrix_safety_hooks",
            kind="e2e",
            call=_pytest_case(
                [
                    "tests/test_agent_safety_hooks.py",
                    "tests/test_cyber_physical.py",
                    "tests/test_containment.py",
                    "tests/test_zk_attestation.py",
                ],
            ),
        ),
        SuiteCase(
            name="injection_adversarial_suite",
            kind="e2e",
            call=_injection_adversarial_case,
        ),
        SuiteCase(
            name="latency_scorer_cpu",
            kind="latency",
            call=_latency_scorer_cpu,
        ),
    ]


# ─── smoke cases ───────────────────────────────────────────────────


def _smoke_package_import() -> CaseOutput:
    # Explicit import + attribute access so ruff sees the names
    # as used (no F401) while the case still proves the chain
    # ``director_ai.core.{CoherenceAgent, CoherenceScorer}`` +
    # ``director_ai.core.safety.InputSanitizer`` all resolve.
    import director_ai
    from director_ai.core import CoherenceAgent, CoherenceScorer
    from director_ai.core.safety import InputSanitizer

    must_exist = (CoherenceAgent, CoherenceScorer, InputSanitizer)
    missing_types = [t.__name__ for t in must_exist if not isinstance(t, type)]
    if missing_types:
        raise RuntimeError(
            f"expected class imports but got non-classes: {missing_types}"
        )
    version = getattr(director_ai, "__version__", "")
    return CaseOutput(
        metrics=[
            MetricResult(
                name="import_ok", value=1.0, unit="bool", higher_is_better=True
            ),
        ],
        notes=f"director_ai.__version__={version or 'unknown'}",
    )


def _smoke_rust_accelerator() -> CaseOutput:
    try:
        import backfire_kernel
    except ImportError:
        return CaseOutput(
            metrics=[
                MetricResult(
                    name="rust_available",
                    value=0.0,
                    unit="bool",
                    higher_is_better=True,
                ),
            ],
            notes="backfire_kernel not installed — Rust accelerators inactive",
            warning=True,
        )
    rust_fns = [name for name in dir(backfire_kernel) if name.startswith("rust_")]
    return CaseOutput(
        metrics=[
            MetricResult(
                name="rust_available",
                value=1.0,
                unit="bool",
                higher_is_better=True,
            ),
            MetricResult(
                name="rust_function_count",
                value=float(len(rust_fns)),
                unit="count",
                higher_is_better=True,
            ),
        ],
        notes=f"{len(rust_fns)} rust_* functions exposed",
    )


def _smoke_agent_construction() -> CaseOutput:
    from director_ai.core import CoherenceAgent

    agent = CoherenceAgent()
    # Exercise the read-only invariant: a fresh agent has no hooks.
    attached = [
        name
        for name in ("containment_guard", "grounding_hook", "passport_verifier")
        if getattr(agent, name, None) is not None
    ]
    if attached:
        return CaseOutput(
            metrics=[
                MetricResult(
                    name="default_hooks_attached",
                    value=float(len(attached)),
                    unit="count",
                    higher_is_better=False,
                ),
            ],
            notes=f"unexpected hook attachment: {attached}",
            warning=True,
        )
    return CaseOutput(
        metrics=[
            MetricResult(
                name="default_hooks_attached",
                value=0.0,
                unit="count",
                higher_is_better=False,
            ),
        ],
    )


def _smoke_safety_hooks_import() -> CaseOutput:
    names = [
        "director_ai.core.cyber_physical.GroundingHook",
        "director_ai.core.containment.ContainmentGuard",
        "director_ai.core.zk_attestation.PassportVerifier",
    ]
    imported = 0
    missing: list[str] = []
    for qual in names:
        module_path, attr = qual.rsplit(".", 1)
        try:
            mod = __import__(module_path, fromlist=[attr])
            getattr(mod, attr)
            imported += 1
        except (ImportError, AttributeError) as exc:
            missing.append(f"{qual}: {exc}")
    return CaseOutput(
        metrics=[
            MetricResult(
                name="safety_hooks_imported",
                value=float(imported),
                unit="count",
                higher_is_better=True,
            ),
            MetricResult(
                name="safety_hooks_total",
                value=float(len(names)),
                unit="count",
                higher_is_better=True,
            ),
        ],
        notes="; ".join(missing) if missing else "all 3 hook packages importable",
        warning=bool(missing),
    )


# ─── pytest-backed case factory ────────────────────────────────────


def _pytest_case(test_paths: list[str]):
    """Factory: return a callable that runs ``pytest`` on the
    given paths and parses the JUnit XML for counts."""

    def _run() -> CaseOutput:
        pytest_exe = sys.executable
        xml_target = Path("benchmarks/results/_tmp_pytest_junit.xml")
        xml_target.parent.mkdir(parents=True, exist_ok=True)
        if xml_target.exists():
            xml_target.unlink()

        cmd = [
            pytest_exe,
            "-m",
            "pytest",
            *test_paths,
            "--tb=no",
            "--quiet",
            "-p",
            "no:cacheprovider",
            "--no-header",
            f"--junit-xml={xml_target}",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
        )

        passed = failed = skipped = error = total = 0
        if xml_target.exists():
            tree = ET.parse(xml_target)
            root = tree.getroot()
            testsuites = (
                [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
            )
            for ts in testsuites:
                total += int(ts.get("tests", "0") or 0)
                failed += int(ts.get("failures", "0") or 0)
                error += int(ts.get("errors", "0") or 0)
                skipped += int(ts.get("skipped", "0") or 0)
            passed = max(0, total - failed - error - skipped)
            xml_target.unlink()

        return CaseOutput(
            metrics=[
                MetricResult(
                    name="pass_count",
                    value=float(passed),
                    unit="count",
                    higher_is_better=True,
                ),
                MetricResult(
                    name="fail_count",
                    value=float(failed + error),
                    unit="count",
                    higher_is_better=False,
                ),
                MetricResult(
                    name="skip_count",
                    value=float(skipped),
                    unit="count",
                    higher_is_better=False,
                ),
                MetricResult(
                    name="total_count",
                    value=float(total),
                    unit="count",
                    higher_is_better=True,
                ),
            ],
            dataset_size=total,
            notes=(
                f"pytest exit={proc.returncode}; "
                f"stdout tail: {proc.stdout[-400:] if proc.stdout else ''!r}"
            ),
            warning=(failed + error) > 0 or proc.returncode != 0,
        )

    return _run


# ─── injection adversarial case ────────────────────────────────────


def _injection_adversarial_case() -> CaseOutput:
    """Run the adversarial pattern suites against a
    :class:`CoherenceScorer`-backed review function and report
    the aggregate detection rate across the hallucination
    transformation suite and the injection suite.

    Delegates to ``director_ai.testing.adversarial_suite`` so the
    numbers this case reports are identical to the stand-alone
    script's output.
    """
    from director_ai.core.scoring.scorer import CoherenceScorer
    from director_ai.testing.adversarial_suite import (
        AdversarialTester,
        InjectionAdversarialTester,
    )

    scorer = CoherenceScorer(threshold=0.6, use_nli=False)

    def review_fn(prompt: str, response: str) -> tuple[bool, float]:
        approved, score = scorer.review(prompt, response)
        return approved, float(score.score)

    hallucination_report = AdversarialTester(
        review_fn=review_fn,
        prompt="What is the capital of France?",
    ).run()

    from director_ai.core.safety.injection import InjectionDetector

    detector = InjectionDetector()

    def detect_fn(*, intent: str, response: str):
        return detector.detect(intent=intent, response=response)

    injection_report = InjectionAdversarialTester(detect_fn=detect_fn).run()

    total = hallucination_report.total_patterns + injection_report.total_patterns
    detected = hallucination_report.detected + injection_report.detected
    rate = detected / total if total else 0.0
    return CaseOutput(
        metrics=[
            MetricResult(
                name="detection_rate",
                value=rate,
                unit="ratio",
                higher_is_better=True,
            ),
            MetricResult(
                name="detected_count",
                value=float(detected),
                unit="count",
                higher_is_better=True,
            ),
            MetricResult(
                name="pattern_count",
                value=float(total),
                unit="count",
                higher_is_better=True,
            ),
            MetricResult(
                name="hallucination_detection_rate",
                value=hallucination_report.detection_rate,
                unit="ratio",
                higher_is_better=True,
            ),
            MetricResult(
                name="injection_detection_rate",
                value=injection_report.detection_rate,
                unit="ratio",
                higher_is_better=True,
            ),
        ],
        dataset_size=total,
        notes=(
            f"hallucination={hallucination_report.detected}/"
            f"{hallucination_report.total_patterns}; "
            f"injection={injection_report.detected}/"
            f"{injection_report.total_patterns}"
        ),
    )


# ─── latency case ──────────────────────────────────────────────────


def _latency_scorer_cpu() -> CaseOutput:
    """Percentile latency of the baseline scorer on a fixed
    prompt / response set.

    Always runs on CPU — the dedicated GPU latency case lives
    in :mod:`.cases_gpu` and is auto-skipped when no CUDA device
    is present. Warm-up is 20 iterations (amortise the first
    import + JIT costs); measurement is 200 iterations.
    """
    from director_ai.core.scoring.scorer import CoherenceScorer

    scorer = CoherenceScorer(threshold=0.6, use_nli=False)
    prompt = "Paris is the capital of France."
    response = "Paris is the capital of France and has a population of 2.1 million."

    warmup = 20
    measure = 200
    for _ in range(warmup):
        scorer.review(prompt, response)

    samples: list[float] = []
    for _ in range(measure):
        t0 = time.perf_counter()
        scorer.review(prompt, response)
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    p50 = samples[len(samples) // 2]
    p95 = samples[int(len(samples) * 0.95)]
    p99 = samples[int(len(samples) * 0.99)]
    mean = sum(samples) / len(samples)

    return CaseOutput(
        metrics=[
            MetricResult(
                name="p50_latency_ms",
                value=p50,
                unit="ms",
                higher_is_better=False,
            ),
            MetricResult(
                name="p95_latency_ms",
                value=p95,
                unit="ms",
                higher_is_better=False,
            ),
            MetricResult(
                name="p99_latency_ms",
                value=p99,
                unit="ms",
                higher_is_better=False,
            ),
            MetricResult(
                name="mean_latency_ms",
                value=mean,
                unit="ms",
                higher_is_better=False,
            ),
        ],
        dataset_size=measure,
        notes=f"warmup={warmup} measure={measure}",
    )


# ─── pytest coverage case (opt-in) ─────────────────────────────────


def pytest_coverage_case() -> SuiteCase:
    """Returns a case that runs pytest with coverage on the full
    test suite. Not included in :func:`default_cases` — it is too
    slow for the default CI path. The CLI wires it under
    ``--with-coverage``.
    """

    def _run() -> CaseOutput:
        if shutil.which(sys.executable) is None:
            return CaseOutput(
                metrics=[],
                notes="python interpreter not discoverable",
                warning=True,
            )
        xml_target = Path("benchmarks/results/_tmp_coverage.xml")
        xml_target.parent.mkdir(parents=True, exist_ok=True)
        if xml_target.exists():
            xml_target.unlink()
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "--tb=no",
            "--quiet",
            "--cov=director_ai",
            f"--cov-report=xml:{xml_target}",
            "--cov-report=term",
            "-p",
            "no:cacheprovider",
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=1800
        )
        if xml_target.exists():
            root = ET.parse(xml_target).getroot()
            line_rate = float(root.get("line-rate", "0") or 0.0)
            xml_target.unlink()
        else:
            line_rate = 0.0
        return CaseOutput(
            metrics=[
                MetricResult(
                    name="coverage_percent",
                    value=line_rate * 100.0,
                    unit="%",
                    higher_is_better=True,
                ),
            ],
            notes=f"pytest exit={proc.returncode}",
            warning=proc.returncode != 0,
        )

    return SuiteCase(name="pytest_coverage", kind="smoke", call=_run)
