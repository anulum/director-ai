# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ModuleTestSuite

"""Score a materialised module against a reference corpus.

Two metrics:

* **Mean absolute error** (MAE) — average ``|score - label|``.
  Lower is better.
* **Spearman rank correlation** — robust to monotone
  transforms of the score. Higher is better.

Both metrics are reported so the engine can promote on whichever
matters for the deployment. A module that wins MAE does not
necessarily win rank correlation; operators who care about
relative ranking over absolute calibration select accordingly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .builder import BoundedSandbox, SandboxTimeout, Scorer


@dataclass(frozen=True)
class ScoredSample:
    """One labelled sample."""

    prompt: str
    label: float

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("ScoredSample.prompt must be non-empty")
        if not 0.0 <= self.label <= 1.0:
            raise ValueError(
                f"ScoredSample.label must be in [0, 1]; got {self.label!r}"
            )


@dataclass(frozen=True)
class SuiteResult:
    """Outcome of one :meth:`ModuleTestSuite.evaluate` call."""

    sample_count: int
    mean_absolute_error: float
    spearman_rank_correlation: float
    timed_out: int = 0
    exceptions: int = 0

    @property
    def ok(self) -> bool:
        """``True`` when every sample scored without errors or
        timeouts — the engine refuses to promote an unreliable
        module even if its metrics look good."""
        return self.timed_out == 0 and self.exceptions == 0


class ModuleTestSuite:
    """Evaluate a :class:`Scorer` against a reference corpus.

    Parameters
    ----------
    samples :
        Non-empty corpus of :class:`ScoredSample` records.
    sandbox :
        Optional :class:`BoundedSandbox`. Default bounded to
        0.5 s per call — enough for a single-prompt pure-Python
        scorer.
    """

    def __init__(
        self,
        *,
        samples: Sequence[ScoredSample],
        sandbox: BoundedSandbox | None = None,
    ) -> None:
        if not samples:
            raise ValueError("samples must be non-empty")
        self._samples = tuple(samples)
        self._sandbox = sandbox or BoundedSandbox(timeout_seconds=0.5)

    def evaluate(self, scorer: Scorer) -> SuiteResult:
        predictions: list[float] = []
        labels: list[float] = []
        timeouts = 0
        exceptions = 0
        for sample in self._samples:
            try:
                value = self._sandbox.run(scorer, sample.prompt)
            except SandboxTimeout:
                timeouts += 1
                continue
            except Exception:  # pragma: no cover — defensive
                exceptions += 1
                continue
            predictions.append(value)
            labels.append(sample.label)
        if not predictions:
            return SuiteResult(
                sample_count=len(self._samples),
                mean_absolute_error=1.0,
                spearman_rank_correlation=0.0,
                timed_out=timeouts,
                exceptions=exceptions,
            )
        mae = sum(abs(p - l) for p, l in zip(predictions, labels, strict=True)) / len(predictions)
        rho = _spearman(predictions, labels)
        return SuiteResult(
            sample_count=len(self._samples),
            mean_absolute_error=mae,
            spearman_rank_correlation=rho,
            timed_out=timeouts,
            exceptions=exceptions,
        )


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation — Pearson on rank-transformed
    inputs, tie-corrected via average ranks.

    Returns 0.0 when either input has zero variance (no
    meaningful correlation to report).
    """
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ranks_a = _ranks(a)
    ranks_b = _ranks(b)
    n = len(a)
    mean_a = sum(ranks_a) / n
    mean_b = sum(ranks_b) / n
    cov = sum(
        (x - mean_a) * (y - mean_b)
        for x, y in zip(ranks_a, ranks_b, strict=True)
    ) / n
    var_a = sum((x - mean_a) ** 2 for x in ranks_a) / n
    var_b = sum((y - mean_b) ** 2 for y in ranks_b) / n
    denom = (var_a * var_b) ** 0.5
    if denom <= 0:
        return 0.0
    return float(cov / denom)


def _ranks(values: Sequence[float]) -> list[float]:
    """Fractional ranks with ties averaged."""
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        average_rank = (i + j) / 2 + 1  # 1-based average
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = average_rank
        i = j + 1
    return ranks
