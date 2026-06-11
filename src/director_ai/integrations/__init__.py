# SPDX-License-Identifier: Apache-2.0
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

from .inference_server_hooks import (
    InferenceHookDecision,
    InferenceHookRequest,
    InferenceServerHook,
    InferenceServerHookPolicy,
    build_inference_server_hook,
)
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
    "InferenceHookDecision",
    "InferenceHookRequest",
    "InferenceServerHook",
    "InferenceServerHookPolicy",
    "LLMProvider",
    "LocalProvider",
    "OpenAIProvider",
    "build_inference_server_hook",
]

# Ecosystem integrations — import directly from submodules:
#   from director_ai.integrations.langchain import DirectorAIGuard
#   from director_ai.integrations.llamaindex import DirectorAIPostprocessor
