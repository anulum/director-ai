# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Experimental Hook Namespace

"""Explicit namespace for research hooks disabled by default.

The modules exposed here are outside the default halt path. Import them through
this namespace only after setting ``DIRECTOR_AI_ENABLE_EXPERIMENTAL_HOOKS=1`` or
calling :func:`enable_experimental_hooks` in the current process.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

__all__ = [
    "EXPERIMENTAL_HOOKS",
    "ExperimentalFeatureError",
    "available_hooks",
    "disable_experimental_hooks",
    "enable_experimental_hooks",
    "experimental_hooks_enabled",
    "load_hook",
]

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_ENV_FLAG = "DIRECTOR_AI_ENABLE_EXPERIMENTAL_HOOKS"
_ENABLED_IN_PROCESS = False

EXPERIMENTAL_HOOKS: dict[str, str] = {
    "agent_identity": "director_ai.core.agent_identity",
    "autopoietic": "director_ai.core.autopoietic",
    "causal_verifier": "director_ai.core.causal_verifier",
    "continual_adversarial": "director_ai.core.continual_adversarial",
    "defense_genome": "director_ai.core.defense_genome",
    "emergence_oracle": "director_ai.core.emergence_oracle",
    "federated_privacy": "director_ai.core.federated_privacy",
    "formal_verification": "director_ai.core.formal_verification",
    "irreversibility": "director_ai.core.irreversibility",
    "knowledge_graph": "director_ai.core.knowledge_graph",
    "meta_guard": "director_ai.core.meta_guard",
    "multi_scale_alignment": "director_ai.core.multi_scale_alignment",
    "multimodal_guard": "director_ai.core.multimodal_guard",
    "ontology": "director_ai.core.ontology",
    "provenance": "director_ai.core.provenance",
    "self_evolving": "director_ai.core.self_evolving",
    "sustainability": "director_ai.core.sustainability",
    "swarm_economics": "director_ai.core.swarm_economics",
    "swarm_equilibrium": "director_ai.core.swarm_equilibrium",
    "symbolic_chain": "director_ai.core.symbolic_chain",
    "trace_safe": "director_ai.core.trace_safe",
    "trajectory": "director_ai.core.trajectory",
    "zk_attestation": "director_ai.core.zk_attestation",
}


class ExperimentalFeatureError(ImportError):
    """Raised when a disabled research hook is requested."""


def enable_experimental_hooks() -> None:
    """Enable research hook loading for this Python process."""
    global _ENABLED_IN_PROCESS
    _ENABLED_IN_PROCESS = True


def disable_experimental_hooks() -> None:
    """Disable process-local research hook loading."""
    global _ENABLED_IN_PROCESS
    _ENABLED_IN_PROCESS = False


def experimental_hooks_enabled() -> bool:
    """Return whether research hook loading is currently enabled."""
    env_value = os.environ.get(_ENV_FLAG, "").strip().lower()
    return _ENABLED_IN_PROCESS or env_value in _TRUE_VALUES


def available_hooks() -> tuple[str, ...]:
    """Return supported research hook names."""
    return tuple(sorted(EXPERIMENTAL_HOOKS))


def load_hook(name: str) -> ModuleType:
    """Load a research hook module after the explicit gate is open."""
    if name not in EXPERIMENTAL_HOOKS:
        raise KeyError(f"unknown experimental hook {name!r}")
    if not experimental_hooks_enabled():
        raise ExperimentalFeatureError(
            f"{name!r} is an experimental Director-AI hook. "
            f"Set {_ENV_FLAG}=1 or call enable_experimental_hooks() before loading it."
        )
    return importlib.import_module(EXPERIMENTAL_HOOKS[name])


def __getattr__(name: str) -> ModuleType:
    if name in EXPERIMENTAL_HOOKS:
        module = load_hook(name)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
