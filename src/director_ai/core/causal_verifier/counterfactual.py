# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CounterfactualVerifier

"""Generate a handful of counterfactual branches around a decision
point and report which branches preserve a caller-supplied safety
invariant.

The verifier is the glue that turns the :class:`CausalGraph` +
:class:`Intervention` primitives into an actionable safety check.
It does not model uncertainty — every branch is deterministic.
Monte-Carlo-style uncertainty lives in :mod:`irreversibility` and
can be composed on top of this verifier later.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

try:
    from backfire_kernel import rust_sum_i64

    _RUST_COUNTERFACTUAL = True
except Exception:  # pragma: no cover - optional dependency
    _RUST_COUNTERFACTUAL = False

    def rust_sum_i64(_values: list[int]) -> int:
        raise RuntimeError("backfire_kernel rust_sum_i64 is unavailable")


from ..types import (
    CounterfactualFactChange,
    CounterfactualHaltDiagnostic,
    EvidenceChunk,
)
from .graph import CausalGraph
from .intervention import Intervention

SafetyInvariant = Callable[[Mapping[str, object]], bool]

BranchOutcome = Literal["safe", "unsafe"]


def _clip(text: str, limit: int = 500) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 3]}..."


def _counterfactual_float(values: Mapping[str, object], key: str) -> float:
    value = values[key]
    if isinstance(value, str | int | float):
        return float(value)
    raise TypeError(f"causal value {key!r} is not numeric")


@dataclass(frozen=True)
class CounterfactualBranch:
    """One what-if branch and its verdict."""

    label: str
    intervention: Intervention
    values: Mapping[str, object]
    outcome: BranchOutcome


@dataclass(frozen=True)
class Verdict:
    """Aggregate verdict across every counterfactual branch."""

    total: int
    safe: int
    unsafe_branches: tuple[CounterfactualBranch, ...]

    @property
    def unsafe(self) -> int:
        return self.total - self.safe

    @property
    def safety_rate(self) -> float:
        return self.safe / self.total if self.total else 0.0


class CounterfactualVerifier:
    """Run a set of interventions and grade each against a safety
    invariant.

    Parameters
    ----------
    graph :
        The :class:`CausalGraph` to operate on.
    safety_invariant :
        Callable that receives the post-intervention variable mapping
        and returns ``True`` when the branch is safe.
    """

    def __init__(
        self,
        graph: CausalGraph,
        *,
        safety_invariant: SafetyInvariant,
    ) -> None:
        self._graph = graph
        self._invariant = safety_invariant

    def verify(
        self,
        *,
        inputs: Mapping[str, object],
        branches: Iterable[tuple[str, Intervention]],
    ) -> Verdict:
        """Evaluate each ``(label, intervention)`` branch and
        aggregate the results.

        Raises :class:`ValueError` when the branch list is empty —
        a verdict over zero branches has no operational meaning.
        """
        results: list[CounterfactualBranch] = []
        unsafe_results: list[CounterfactualBranch] = []
        for label, intervention in branches:
            values = intervention.apply(self._graph, inputs)
            outcome: BranchOutcome = "safe" if self._invariant(values) else "unsafe"
            branch = CounterfactualBranch(
                label=label,
                intervention=intervention,
                values=values,
                outcome=outcome,
            )
            results.append(branch)
            if outcome == "unsafe":
                unsafe_results.append(branch)
        if not results:
            raise ValueError("at least one branch is required")
        return Verdict(
            total=len(results),
            safe=_sum_int([1 if b.outcome == "safe" else 0 for b in results]),
            unsafe_branches=tuple(unsafe_results),
        )

    @classmethod
    def explain_halt_fact_change(
        cls,
        *,
        observed_score: float,
        threshold: float,
        evidence_chunks: Iterable[EvidenceChunk],
        proposed_fact: str,
    ) -> CounterfactualHaltDiagnostic:
        """Answer which single retrieved fact change would prevent a halt."""
        chunks = tuple(evidence_chunks)
        required_delta = max(0.0, threshold - observed_score)
        clipped_proposal = _clip(proposed_fact)
        question = "what single fact change would have prevented this halt?"
        if not chunks:
            return CounterfactualHaltDiagnostic(
                question=question,
                observed_score=observed_score,
                threshold=threshold,
                best_change=None,
                candidates=[],
            )

        graph = CausalGraph()
        graph.add("observed_score", lambda _: observed_score)
        graph.add("score_delta", lambda _: 0.0)
        graph.add(
            "adjusted_score",
            lambda p: min(
                1.0,
                _counterfactual_float(p, "observed_score")
                + _counterfactual_float(p, "score_delta"),
            ),
            parents=("observed_score", "score_delta"),
        )
        graph.add(
            "halted",
            lambda p: _counterfactual_float(p, "adjusted_score") < threshold,
            parents=("adjusted_score",),
        )
        verifier = cls(
            graph,
            safety_invariant=lambda values: not bool(values["halted"]),
        )
        branches = [
            (
                f"fact:{index}:{chunk.source or 'unknown'}",
                Intervention({"score_delta": required_delta}),
            )
            for index, chunk in enumerate(chunks)
        ]
        verdict = verifier.verify(inputs={}, branches=branches)
        unsafe_labels = {branch.label for branch in verdict.unsafe_branches}
        candidates = [
            CounterfactualFactChange(
                fact_source=chunk.source,
                original_fact=_clip(chunk.text),
                proposed_fact=clipped_proposal,
                required_score_delta=required_delta,
                prevented_halt=label not in unsafe_labels,
            )
            for (label, _), chunk in zip(branches, chunks, strict=True)
        ]
        best_change = next(
            (candidate for candidate in candidates if candidate.prevented_halt),
            None,
        )
        return CounterfactualHaltDiagnostic(
            question=question,
            observed_score=observed_score,
            threshold=threshold,
            best_change=best_change,
            candidates=candidates,
        )


def _sum_int(values: list[int]) -> int:
    if _RUST_COUNTERFACTUAL:
        with suppress(Exception):
            return int(rust_sum_i64(values))
    return sum(values)
