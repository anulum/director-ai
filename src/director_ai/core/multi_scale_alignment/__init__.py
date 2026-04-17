# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — multi-scale alignment verifier

"""Hierarchical alignment check from agent to planetary scale.

The guardrail evaluates a proposed action against a stack of
value scorers, one per alignment scale, and reports conflicts
when a scale says ``allow`` while another scale says ``deny``.

* :data:`AlignmentScale` — ordered enumeration ``agent`` →
  ``swarm`` → ``org`` → ``planetary``.
* :class:`ValueVector` — immutable mapping of value name to a
  weight in ``[0, 1]``. Equality-based hashing so two vectors
  with the same weights are interchangeable in dicts.
* :class:`ScaleScorer` — Protocol for anything that scores an
  action at one scale. The shipped :class:`ValueLatticeScorer`
  applies a :class:`ValueVector` to a labelled action and
  returns a calibrated safety score.
* :class:`HierarchicalAligner` — composes per-scale scores into
  a :class:`ScaleScoreTable` and flags every scale where the
  score falls below the operator-supplied allow threshold.
* :class:`ScaleConflictDetector` — tests for cross-scale
  disagreement using a conformal threshold on the pairwise
  score deltas. A single-scale miss is a local problem; a
  large delta between scales is a structural one.
* :class:`AlignmentReport` — per-scale scores + conflicts + a
  composite alignment score.

The composite score folds every scale into a single ``[0, 1]``
number using a caller-supplied weight vector over the four
scales. The default weights lean slightly toward higher scales
so systemic misalignment outvotes local misalignment when both
are present.
"""

from .aligner import (
    AlignmentReport,
    HierarchicalAligner,
    ScaleScoreTable,
)
from .conflict import ScaleConflict, ScaleConflictDetector
from .scorer import (
    Action,
    AlignmentScale,
    ScaleScorer,
    ValueLatticeScorer,
    ValueVector,
)

__all__ = [
    "Action",
    "AlignmentReport",
    "AlignmentScale",
    "HierarchicalAligner",
    "ScaleConflict",
    "ScaleConflictDetector",
    "ScaleScorer",
    "ScaleScoreTable",
    "ValueLatticeScorer",
    "ValueVector",
]
