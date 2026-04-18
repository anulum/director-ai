# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — benchmark orchestrator

"""Systematic, honest, replicable benchmark + E2E test runner.

The orchestrator pulls every measurable behaviour of Director-AI
into one cohesive report that can be produced locally or on
Vertex AI Custom Training. Every result carries full provenance
(commit SHA, hardware fingerprint, dataset hash, seed) so a
re-run from a different operator can be cross-checked against a
prior run without ambiguity.

Entry point: ``python -m benchmarks.orchestrator`` (see
``cli.py``).

Key modules:

* :mod:`.schema` — dataclasses + validators for every JSON
  artefact the orchestrator emits. A result file that does not
  round-trip through :class:`RunReport.from_json` fails the
  regression gate — no silent schema drift.
* :mod:`.environment` — collects the ``(commit, hardware,
  dataset_hash, seed)`` tuple that makes a run replicable.
* :mod:`.suite` — declares the suite of measurements and
  executes each one with uniform error handling.
* :mod:`.regression` — loads a baseline report, compares each
  metric against the new run, flags regressions against
  per-metric thresholds.
* :mod:`.cli` — argparse entry point.
"""

from __future__ import annotations

from .environment import EnvironmentFingerprint, capture_environment
from .regression import RegressionReport, RegressionRule, detect_regressions
from .schema import (
    MetricResult,
    RunReport,
    SuiteEntry,
    validate_report,
)
from .suite import Suite, SuiteRunner

__all__ = [
    "EnvironmentFingerprint",
    "MetricResult",
    "RegressionReport",
    "RegressionRule",
    "RunReport",
    "Suite",
    "SuiteEntry",
    "SuiteRunner",
    "capture_environment",
    "detect_regressions",
    "validate_report",
]
