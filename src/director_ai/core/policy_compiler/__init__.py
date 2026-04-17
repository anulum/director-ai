# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — policy compiler package

"""Compile free-text compliance documents into :class:`Policy`
modules the rest of the stack can enforce.

Four pieces:

* :class:`CompiledRule` — one extracted rule, with a conformal
  threshold slot for channels that need calibrated cut-offs.
* :class:`RuleExtractor` — Protocol for any extractor. The
  :class:`RegexRuleExtractor` that ships here parses imperative
  compliance style (``must not`` / ``shall not`` / ``maximum`` /
  ``at least``) deterministically; LLM-backed extractors drop in
  on the same Protocol.
* :class:`PolicyCompiler` — orchestrates extraction, optional
  split-conformal calibration, and produces a
  :class:`PolicyBundle` that can be turned into the existing
  :class:`~director_ai.core.safety.policy.Policy`.
* :class:`PolicyRegistry` — thread-safe atomic hot-swap store.
  ``register()`` never lets a reader see a half-built bundle,
  and monotonic versioning prevents rollback unless explicitly
  enabled.
"""

from .compiler import CompiledRule, PolicyBundle, PolicyCompiler
from .extractor import RegexRuleExtractor, RuleExtractor
from .registry import PolicyRegistry

__all__ = [
    "CompiledRule",
    "PolicyBundle",
    "PolicyCompiler",
    "PolicyRegistry",
    "RegexRuleExtractor",
    "RuleExtractor",
]
