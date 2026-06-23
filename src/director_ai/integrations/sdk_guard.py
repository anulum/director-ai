# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Native SDK interceptors for common LLM clients.

Usage::

    from director_ai import guard
    client = guard(OpenAI(), facts={"refund": "within 30 days"})
    resp = client.chat.completions.create(...)  # auto-scored
"""

from __future__ import annotations

from typing import Any

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.exceptions import (
    InjectionDetectedError as InjectionDetectedError,
)

from .sdk_proxies.anthropic import _anthropic_response_text as _anthropic_response_text
from .sdk_proxies.anthropic import _AnthropicMessagesProxy as _AnthropicMessagesProxy
from .sdk_proxies.anthropic import (
    _extract_anthropic_event_text as _extract_anthropic_event_text,
)
from .sdk_proxies.anthropic import _GuardedAnthropicStream as _GuardedAnthropicStream
from .sdk_proxies.anthropic import _has_anthropic_shape as _has_anthropic_shape
from .sdk_proxies.base import STREAM_CHECK_INTERVAL as STREAM_CHECK_INTERVAL
from .sdk_proxies.base import _ascore_and_gate as _ascore_and_gate
from .sdk_proxies.base import _check_injection as _check_injection
from .sdk_proxies.base import _extract_prompt as _extract_prompt
from .sdk_proxies.base import _handle_failure as _handle_failure
from .sdk_proxies.base import _handle_injection_failure as _handle_injection_failure
from .sdk_proxies.base import _log as _log
from .sdk_proxies.base import _score_and_gate as _score_and_gate
from .sdk_proxies.base import _score_var as _score_var
from .sdk_proxies.base import get_score as get_score
from .sdk_proxies.base import score as score
from .sdk_proxies.bedrock import _bedrock_response_text as _bedrock_response_text
from .sdk_proxies.bedrock import _BedrockProxy as _BedrockProxy
from .sdk_proxies.bedrock import _extract_bedrock_prompt as _extract_bedrock_prompt
from .sdk_proxies.bedrock import (
    _extract_bedrock_stream_delta as _extract_bedrock_stream_delta,
)
from .sdk_proxies.bedrock import _GuardedBedrockStream as _GuardedBedrockStream
from .sdk_proxies.bedrock import _has_bedrock_shape as _has_bedrock_shape
from .sdk_proxies.cohere import _CohereProxy as _CohereProxy
from .sdk_proxies.cohere import _GuardedCohereStream as _GuardedCohereStream
from .sdk_proxies.cohere import _has_cohere_shape as _has_cohere_shape
from .sdk_proxies.gemini import _extract_gemini_prompt as _extract_gemini_prompt
from .sdk_proxies.gemini import _GeminiProxy as _GeminiProxy
from .sdk_proxies.gemini import _GuardedGeminiStream as _GuardedGeminiStream
from .sdk_proxies.gemini import _has_gemini_shape as _has_gemini_shape
from .sdk_proxies.mistral import _has_mistral_shape as _has_mistral_shape
from .sdk_proxies.mistral import _mistral_response_text as _mistral_response_text
from .sdk_proxies.mistral import _MistralChatProxy as _MistralChatProxy
from .sdk_proxies.openai import _extract_stream_delta as _extract_stream_delta
from .sdk_proxies.openai import _GuardedOpenAIStream as _GuardedOpenAIStream
from .sdk_proxies.openai import _has_openai_shape as _has_openai_shape
from .sdk_proxies.openai import _openai_response_text as _openai_response_text
from .sdk_proxies.openai import _OpenAICompletionsProxy as _OpenAICompletionsProxy
from .sdk_proxies.pydantic_ai import (
    _extract_pydantic_ai_prompt as _extract_pydantic_ai_prompt,
)
from .sdk_proxies.pydantic_ai import _has_pydantic_ai_shape as _has_pydantic_ai_shape
from .sdk_proxies.pydantic_ai import (
    _pydantic_ai_content_text as _pydantic_ai_content_text,
)
from .sdk_proxies.pydantic_ai import (
    _pydantic_ai_output_text as _pydantic_ai_output_text,
)
from .sdk_proxies.pydantic_ai import _PydanticAIProxy as _PydanticAIProxy


def guard(
    client: Any,
    *,
    facts: dict[str, str] | None = None,
    store: GroundTruthStore | None = None,
    threshold: float = 0.3,
    use_nli: bool | None = None,
    on_fail: str = "raise",
    injection_detection: bool = False,
    injection_threshold: float = 0.7,
    require_model_backed_nli: bool = False,
    injection_require_model_backed_nli: bool = False,
    injection_fail_closed_on_error: bool = False,
) -> Any:
    """Wrap an LLM SDK client with coherence scoring.

    Supports seven SDK shapes:

    - **OpenAI-compatible** (``client.chat.completions.create``):
      OpenAI, vLLM, Groq, LiteLLM, Ollama, Together.
    - **Anthropic** (``client.messages.create``).
    - **AWS Bedrock** (``client.converse`` / ``client.converse_stream``).
    - **Google Gemini** (``client.generate_content``).
    - **Mistral** (``client.chat.complete``).
    - **Cohere** (``client.chat`` without ``client.completions``).
    - **Pydantic AI** (``agent.run_sync`` / ``agent.run``).

    When *injection_detection* is enabled, each response is additionally
    checked for prompt injection via intent-grounded NLI divergence.
    The *injection_threshold* controls sensitivity (0.0–1.0).

    Returns the guarded client. Some SDK clients are mutated in place; others
    return a proxy when their public surface cannot be safely patched.
    **Always use the return value**:
    ``client = guard(client, ...)``.
    """
    if on_fail not in ("raise", "log", "metadata"):
        raise ValueError(
            f"on_fail must be 'raise', 'log', or 'metadata', got {on_fail!r}",
        )

    gts = store or GroundTruthStore()
    if facts:
        for k, v in facts.items():
            gts.add(k, v)
    scorer = CoherenceScorer(
        threshold=threshold,
        ground_truth_store=gts,
        use_nli=use_nli,
        require_model_backed_nli=require_model_backed_nli,
    )
    inj_threshold = injection_threshold if injection_detection else None
    if injection_detection:
        scorer.enable_injection_detection(
            injection_threshold=injection_threshold,
            require_model_backed_nli=injection_require_model_backed_nli,
            fail_closed_on_error=injection_fail_closed_on_error,
        )

    if _has_openai_shape(client):
        client.chat.completions = _OpenAICompletionsProxy(
            client.chat.completions,
            scorer,
            on_fail,
            injection_threshold=inj_threshold,
        )
    elif _has_anthropic_shape(client):
        client.messages = _AnthropicMessagesProxy(
            client.messages,
            scorer,
            on_fail,
            injection_threshold=inj_threshold,
        )
    elif _has_bedrock_shape(client):
        client = _BedrockProxy(
            client,
            scorer,
            on_fail,
            injection_threshold=inj_threshold,
        )
    elif _has_gemini_shape(client):
        client = _GeminiProxy(
            client,
            scorer,
            on_fail,
            injection_threshold=inj_threshold,
        )
    elif _has_mistral_shape(client):
        client.chat = _MistralChatProxy(
            client.chat,
            scorer,
            on_fail,
            injection_threshold=inj_threshold,
        )
    elif _has_pydantic_ai_shape(client):
        client = _PydanticAIProxy(
            client,
            scorer,
            on_fail,
            injection_threshold=inj_threshold,
        )
    elif _has_cohere_shape(client):
        client = _CohereProxy(
            client,
            scorer,
            on_fail,
            injection_threshold=inj_threshold,
        )
    else:
        raise TypeError(
            f"Unsupported client type: {type(client).__qualname__}. "
            "Expected a supported LLM SDK shape.",
        )
    return client


__all__ = ["guard", "score", "get_score"]
