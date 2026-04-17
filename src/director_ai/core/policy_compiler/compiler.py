# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — PolicyCompiler

"""Compile extracted rules into a :class:`PolicyBundle`.

A :class:`PolicyBundle` is the shape the registry hot-swaps and the
existing :class:`~director_ai.core.safety.policy.Policy` consumes.
The compiler:

* Delegates rule extraction to any :class:`RuleExtractor` (the
  shipped :class:`RegexRuleExtractor` is deterministic; callers
  swap in an LLM extractor as a drop-in).
* Deduplicates rules by ``id`` across all documents so the same
  phrase appearing in two docs lands as one rule.
* Optionally calibrates the thresholds of threshold-bearing rules
  via :func:`split_conformal_threshold` — split conformal on a
  labelled calibration set, targeting a caller-supplied coverage
  level (default 0.90).
* Produces a ``Policy`` object (the existing YAML engine) via
  :meth:`PolicyBundle.to_policy`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .extractor import RegexRuleExtractor, RuleExtractor
from .rule import CompiledRule

if TYPE_CHECKING:  # pragma: no cover — import only for typing
    from director_ai.core.safety.policy import Policy


@dataclass(frozen=True)
class PolicyBundle:
    """Immutable snapshot of a compiled policy.

    ``version`` is a monotonically increasing integer assigned by
    the compiler; the registry uses it to break ties when two
    bundles are registered at the same moment.
    """

    version: int
    rules: tuple[CompiledRule, ...] = field(default_factory=tuple)

    def to_policy(self) -> Policy:
        """Turn the bundle into a runtime :class:`Policy`. Local
        import avoids an import cycle — ``safety.policy`` may
        eventually want to reference the compiler."""
        from director_ai.core.safety.policy import Policy

        forbidden: list[str] = []
        patterns: list[dict] = []
        max_length = 0
        required_citations_pattern = ""
        required_citations_min = 0
        for r in self.rules:
            if r.kind == "forbidden":
                forbidden.append(r.value)
            elif r.kind == "pattern":
                patterns.append({"name": r.name, "regex": r.value, "action": r.action})
            elif r.kind == "max_length":
                try:
                    max_length = max(max_length, int(r.value))
                except ValueError:
                    continue
            elif r.kind == "required_citations":
                try:
                    required_citations_min = max(required_citations_min, int(r.value))
                except ValueError:
                    continue
                if not required_citations_pattern:
                    # Default marker — callers who want a project-specific
                    # shape override the Policy afterwards.
                    required_citations_pattern = r"\[\d+\]"
        return Policy(
            forbidden=forbidden,
            patterns=patterns,
            max_length=max_length,
            required_citations_pattern=required_citations_pattern,
            required_citations_min=required_citations_min,
        )


class PolicyCompiler:
    """Entry point for compiling documents into a :class:`PolicyBundle`.

    Parameters
    ----------
    extractor :
        Any :class:`RuleExtractor`. Defaults to a fresh
        :class:`RegexRuleExtractor`.
    """

    def __init__(self, *, extractor: RuleExtractor | None = None) -> None:
        self._extractor: RuleExtractor = extractor or RegexRuleExtractor()
        self._next_version = 1

    def compile(self, documents: Iterable[str]) -> PolicyBundle:
        """Extract, deduplicate, and wrap into a bundle."""
        rules: list[CompiledRule] = []
        for doc in documents:
            rules.extend(self._extractor.extract(doc))
        deduped = _dedup(rules)
        version = self._next_version
        self._next_version += 1
        return PolicyBundle(version=version, rules=tuple(deduped))

    def calibrate(
        self,
        bundle: PolicyBundle,
        *,
        scores: Sequence[float],
        target_coverage: float = 0.90,
    ) -> PolicyBundle:
        """Return a new bundle with thresholds on each rule set to
        the split-conformal α-quantile of ``scores`` (empirical
        non-conformity scores from a held-out set, in ``[0, 1]``).

        A single threshold is applied to every threshold-bearing
        rule in the bundle — per-rule calibration sets are a
        follow-up. The foundation implementation is enough to let
        the rest of the stack consume a calibrated threshold.
        """
        if not 0.0 < target_coverage < 1.0:
            raise ValueError(
                f"target_coverage must be in (0, 1); got {target_coverage}"
            )
        if not scores:
            raise ValueError("scores must be non-empty")
        threshold = split_conformal_threshold(scores, target_coverage)
        new_rules = tuple(
            CompiledRule(
                id=r.id,
                kind=r.kind,
                value=r.value,
                name=r.name,
                action=r.action,
                threshold=threshold,
                source=r.source,
            )
            for r in bundle.rules
        )
        return PolicyBundle(version=bundle.version, rules=new_rules)


def split_conformal_threshold(
    scores: Sequence[float],
    target_coverage: float,
) -> float:
    """Return the split-conformal ``α``-quantile cut-off.

    Given non-conformity ``scores`` from a calibration fold and a
    target coverage ``1 - α``, returns the ``(1 - α)(n + 1)`` /
    ``n`` order-statistic — the Vovk/Gammerman/Shafer construction
    that keeps the miscoverage bound on exchangeable future draws.

    Scores outside ``[0, 1]`` are clipped — the rule threshold is
    stored in the same range.
    """
    sorted_scores = sorted(max(0.0, min(1.0, s)) for s in scores)
    n = len(sorted_scores)
    # q_index is 1-based on the sorted list; clamp to [0, n-1].
    q_index = min(max(int((target_coverage * (n + 1)) - 1), 0), n - 1)
    return float(sorted_scores[q_index])


def _dedup(rules: Iterable[CompiledRule]) -> list[CompiledRule]:
    seen: set[str] = set()
    out: list[CompiledRule] = []
    for r in rules:
        if r.id in seen:
            continue
        seen.add(r.id)
        out.append(r)
    return out
