# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — orchestrator suite definition + runner

"""Declarative benchmark suite.

A :class:`Suite` is a list of :class:`SuiteCase` instances; each
case knows its name, kind (accuracy / latency / e2e / smoke), and
a callable that produces a ``list[MetricResult]`` when invoked.
:class:`SuiteRunner` iterates the suite, times each case, catches
exceptions (so one broken case does not halt the full run), and
collects the output into a :class:`RunReport` that
:mod:`.regression` can then diff against a baseline.

Two layers of isolation:

* **Per-case**: every case runs in its own try/except; a failure
  is recorded as ``status="failed"`` with the exception text in
  ``notes``. The run continues.
* **Suite-level**: :meth:`SuiteRunner.run` never raises on case
  failures. The only way the runner raises is when the
  environment capture itself fails — which means the whole run
  is untrustworthy anyway.
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
import traceback
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from .environment import capture_environment
from .schema import (
    EnvironmentRecord,
    MetricResult,
    RunReport,
    SuiteEntry,
)

logger = logging.getLogger("DirectorAI.Orchestrator.Suite")

MetricList = list[MetricResult]
CaseCallable = Callable[[], "CaseOutput"]


@dataclass(frozen=True)
class CaseOutput:
    """What a suite case returns to the runner.

    ``metrics`` is the measurement payload (zero or more metrics).
    ``dataset_hash`` / ``dataset_size`` / ``seed`` fill the
    corresponding :class:`SuiteEntry` fields so the report
    captures the exact inputs that produced the numbers.
    """

    metrics: MetricList = field(default_factory=list)
    dataset_hash: str = ""
    dataset_size: int = 0
    seed: int = 0
    notes: str = ""
    warning: bool = False


@dataclass
class SuiteCase:
    """One measurable benchmark inside a :class:`Suite`."""

    name: str
    kind: str
    call: CaseCallable
    skip_reason: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SuiteCase.name must be non-empty")
        if self.kind not in {"accuracy", "latency", "e2e", "smoke"}:
            raise ValueError(
                f"SuiteCase.kind must be accuracy/latency/e2e/smoke; got {self.kind!r}"
            )


@dataclass
class Suite:
    """Ordered collection of :class:`SuiteCase`s."""

    cases: list[SuiteCase]
    name: str = "default"

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for c in self.cases:
            if c.name in seen:
                raise ValueError(
                    f"duplicate case name {c.name!r} in suite {self.name!r}"
                )
            seen.add(c.name)

    def filter(self, names: Iterable[str]) -> Suite:
        """Return a new suite containing only the requested cases.

        Used by the CLI ``--only`` flag so a developer can run one
        case quickly without executing the whole suite.
        """
        wanted = set(names)
        unknown = wanted - {c.name for c in self.cases}
        if unknown:
            raise ValueError(f"unknown case(s): {sorted(unknown)}")
        return Suite(
            cases=[c for c in self.cases if c.name in wanted],
            name=f"{self.name}[filtered]",
        )


class SuiteRunner:
    """Executes a :class:`Suite` and assembles a :class:`RunReport`.

    Parameters
    ----------
    suite :
        The :class:`Suite` to execute.
    environment :
        Pre-captured :class:`EnvironmentRecord`. When ``None`` the
        runner calls :func:`capture_environment` itself. Passing
        a pre-captured one is useful for tests that want a
        deterministic fingerprint.
    """

    def __init__(
        self,
        suite: Suite,
        *,
        environment: EnvironmentRecord | None = None,
        logger_obj: logging.Logger | None = None,
    ) -> None:
        self._suite = suite
        self._environment = environment
        self._logger = logger_obj or logger

    def run(self) -> RunReport:
        env = self._environment or capture_environment()
        entries: list[SuiteEntry] = []
        for case in self._suite.cases:
            entries.append(self._run_case(case))
        return RunReport(
            run_id=str(uuid.uuid4()),
            timestamp_utc=_now_iso_utc(),
            environment=env,
            entries=tuple(entries),
            notes=f"suite={self._suite.name}",
        )

    def _run_case(self, case: SuiteCase) -> SuiteEntry:
        if case.skip_reason:
            self._logger.info("skipping %s: %s", case.name, case.skip_reason)
            return SuiteEntry(
                name=case.name,
                kind=case.kind,
                status="skipped",
                metrics=(),
                wall_clock_seconds=0.0,
                notes=case.skip_reason,
            )

        self._logger.info("running case %s (%s)", case.name, case.kind)
        t0 = time.perf_counter()
        try:
            output = case.call()
        except Exception as exc:
            # Broad catch is deliberate — a benchmark runner must
            # survive any arbitrary failure in a user-supplied case
            # callable; the exception text is preserved in the
            # SuiteEntry.notes so the regression report shows it.
            elapsed = time.perf_counter() - t0
            tb = traceback.format_exc(limit=3)
            self._logger.warning(
                "case %s failed after %.2fs: %s", case.name, elapsed, exc
            )
            return SuiteEntry(
                name=case.name,
                kind=case.kind,
                status="failed",
                metrics=(),
                wall_clock_seconds=elapsed,
                notes=f"{type(exc).__name__}: {exc}\n{tb}",
            )

        elapsed = time.perf_counter() - t0
        status: str = "warned" if output.warning else "passed"
        metrics_tuple = tuple(output.metrics)
        return SuiteEntry(
            name=case.name,
            kind=case.kind,
            # ``SuiteEntry.__post_init__`` narrows the status string.
            status=_narrow_status_str(status),
            metrics=metrics_tuple,
            wall_clock_seconds=elapsed,
            dataset_hash=output.dataset_hash,
            dataset_size=output.dataset_size,
            seed=output.seed,
            notes=output.notes,
        )


def _now_iso_utc() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _narrow_status_str(value: str):
    if value not in {"passed", "failed", "skipped", "warned"}:
        raise ValueError(f"invalid status {value!r}")
    # Return the string as-is — SuiteEntry's __post_init__ does the
    # Literal narrowing at construction. Keeping this helper so the
    # caller's intent (passed → literal) is explicit in one place.
    return value


def build_default_suite(case_sources: Sequence[SuiteCase] | None = None) -> Suite:
    """Build the default production suite.

    Callers can pass ``case_sources`` directly; when absent, the
    orchestrator loads the standard set from :mod:`.cases`.
    Keeping the coupling loose here means the orchestrator package
    compiles even when a single optional case source is broken.
    """
    if case_sources is None:
        # Late-import: the cases module pulls in optional heavy
        # deps (transformers, datasets) that this top-level
        # module must not require just to construct a Suite.
        from . import cases as _cases_mod

        case_sources = _cases_mod.default_cases()
    return Suite(cases=list(case_sources), name="director_ai.default")
