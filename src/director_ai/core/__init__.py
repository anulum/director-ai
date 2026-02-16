# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Core Package (Coherence Engine)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Coherence Engine — consumer-ready AI output verification.

Quick start::

    from director_ai.core import CoherenceAgent

    agent = CoherenceAgent()
    result = agent.process("What color is the sky?")
    print(result.output, result.coherence)
"""

from .actor import LLMGenerator, MockGenerator
from .agent import CoherenceAgent
from .kernel import SafetyKernel
from .knowledge import GroundTruthStore
from .scorer import CoherenceScorer
from .types import CoherenceScore, ReviewResult

__all__ = [
    "CoherenceScore",
    "ReviewResult",
    "CoherenceScorer",
    "SafetyKernel",
    "MockGenerator",
    "LLMGenerator",
    "GroundTruthStore",
    "CoherenceAgent",
]
