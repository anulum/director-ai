# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” LLM Provider Integrations

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

# Ecosystem integrations â€” import directly from submodules:
#   from director_ai.integrations.langchain import DirectorAIGuard
#   from director_ai.integrations.llamaindex import DirectorAIPostprocessor
