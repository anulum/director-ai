# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — LLM Provider Integrations
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
LLM provider adapters for Director-Class AI.

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
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "HuggingFaceProvider",
    "LocalProvider",
]
