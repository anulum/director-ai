# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — agent trajectory simulator package

"""Monte-Carlo agent trajectory simulator.

Given a prompt and an :class:`Actor` that can draw token sequences
with seeded randomness, :class:`TrajectorySimulator` runs N
independent simulations, scores each end-state with a
:class:`CoherenceScorer`-shaped verdict producer, and aggregates
the results into a :class:`PreflightVerdict`: halt rate, mean
coherence, conformal-style credible interval, and a recommended
action (``proceed`` / ``warn`` / ``halt``).

The preflight check happens **before** the real generation, so a
high-risk prompt can be diverted, escalated, or blocked without
first paying the full streaming scoring cost. A large fraction of
simulated halts is a strong signal that the actual generation
will also halt — catching it in preflight saves tokens and lets
the gateway return a structured 422 before the user sees a
broken stream.

The :class:`Actor` Protocol is the stable boundary: distilled
policy models, Rust-accelerated samplers, and calibrated-on-trace
actors all slot in as drop-in implementations.
"""

from .simulator import (
    Actor,
    PreflightVerdict,
    TrajectoryResult,
    TrajectorySimulator,
    VerdictProducer,
)

__all__ = [
    "Actor",
    "PreflightVerdict",
    "TrajectoryResult",
    "TrajectorySimulator",
    "VerdictProducer",
]
