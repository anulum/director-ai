# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Labelling cockpit items

"""The item a reviewer labels in the active-labelling cockpit.

One :class:`LabelItem` is a scored guard decision plus, once a reviewer has seen
it, the ground-truth label of whether the response was actually a hallucination
or grounded. The cockpit ranks unlabelled items for review and, from the labelled
ones, measures error and recommends a threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = ["GROUNDED", "HALLUCINATION", "LABELS", "LabelItem"]

GROUNDED = "grounded"
HALLUCINATION = "hallucination"
LABELS = frozenset({GROUNDED, HALLUCINATION})


@dataclass(frozen=True)
class LabelItem:
    """One scored guard decision, optionally labelled by a reviewer.

    Parameters
    ----------
    item_id:
        Stable identifier for the scored interaction.
    score:
        The coherence score the guard assigned, in ``[0, 1]``.
    guard_approved:
        What the guard decided (``True`` = approved/passed).
    domain:
        The domain of the interaction, for per-domain threshold tuning.
    label:
        The reviewer's ground truth: ``"grounded"`` / ``"hallucination"``, or
        ``None`` when not yet labelled.
    prompt, response:
        Optional text carried into the exported training/eval packet.
    """

    item_id: str
    score: float
    guard_approved: bool
    domain: str = ""
    label: str | None = None
    prompt: str = ""
    response: str = ""

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id is required")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be a finite value in [0, 1]")
        if self.label is not None and self.label not in LABELS:
            raise ValueError(f"label must be one of {sorted(LABELS)} or None")

    @property
    def labelled(self) -> bool:
        """Whether a reviewer has assigned a ground-truth label."""
        return self.label is not None

    @property
    def is_hallucination(self) -> bool:
        """Whether the reviewer labelled this a hallucination."""
        return self.label == HALLUCINATION

    def to_packet_row(self) -> dict[str, Any]:
        """Serialise to a row for the training/eval packet."""
        return {
            "item_id": self.item_id,
            "score": self.score,
            "guard_approved": self.guard_approved,
            "domain": self.domain,
            "label": self.label,
            "prompt": self.prompt,
            "response": self.response,
        }
