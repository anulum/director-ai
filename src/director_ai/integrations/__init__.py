# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — LLM Provider Integrations

"""LLM provider adapters for Director-Class AI.

Usage::

    from director_ai.integrations import OpenAIProvider, AnthropicProvider

    provider = OpenAIProvider(api_key="sk-...")
    agent = CoherenceAgent(generator=provider)
"""

from .providers import (
    AnthropicProvider,
    HuggingFaceProvider,
    LLMProvider,
    LocalProvider,
    OpenAIProvider,
)

__all__ = [
    "AnthropicProvider",
    "HuggingFaceProvider",
    "LLMProvider",
    "LocalProvider",
    "OpenAIProvider",
]

# Ecosystem integrations — import directly from submodules:
#   from director_ai.integrations.langchain import DirectorAIGuard
#   from director_ai.integrations.llamaindex import DirectorAIPostprocessor
