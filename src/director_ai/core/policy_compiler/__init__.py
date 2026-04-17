# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — policy compiler package

"""Policy Compiler (roadmap 2026-2030, Tier 1 Batch 2 b, foundation).

Compile free-text compliance documents into :class:`Policy` modules
the rest of the stack can enforce. Three pieces:

* :class:`CompiledRule` — one extracted rule, with a conformal
  threshold slot for channels that need calibrated cut-offs.
* :class:`RuleExtractor` — Protocol for any extractor. A
  :class:`StubExtractor` ships for tests and bootstrap use (regex
  over the common compliance phrasings); an LLM-backed extractor
  is expected as a drop-in on top of the same Protocol.
* :class:`PolicyCompiler` — orchestrates extraction, optional
  split-conformal calibration, and produces a
  :class:`PolicyBundle` that can be turned into the existing
  :class:`~director_ai.core.safety.policy.Policy`.
* :class:`PolicyRegistry` — thread-safe atomic hot-swap store.
  ``register()`` never lets a reader see a half-built bundle.

Foundation scope: deterministic parsing, calibration, and hot-swap.
LLM extractor, domain-specific calibrators, and document
ingestion pipelines are follow-ups — the Protocol boundary is
stable so those can land without API churn.
"""

from .compiler import CompiledRule, PolicyBundle, PolicyCompiler
from .extractor import RuleExtractor, StubExtractor
from .registry import PolicyRegistry

__all__ = [
    "CompiledRule",
    "PolicyBundle",
    "PolicyCompiler",
    "PolicyRegistry",
    "RuleExtractor",
    "StubExtractor",
]
