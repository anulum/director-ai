# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Pydantic AI SDK guard proxy."""

from __future__ import annotations

import json
from typing import Any

from .base import _ascore_and_gate, _score_and_gate


def _has_pydantic_ai_shape(client: Any) -> bool:
    """Return True for Pydantic AI ``Agent`` instances with run APIs."""
    module = type(client).__module__
    if not module.startswith("pydantic_ai"):
        return False
    return callable(getattr(client, "run_sync", None)) and callable(
        getattr(client, "run", None),
    )


class _PydanticAIProxy:
    """Guard Pydantic AI ``Agent.run_sync`` and ``Agent.run`` outputs."""

    def __init__(
        self,
        agent: Any,
        scorer: Any,
        on_fail: str,
        *,
        injection_threshold: float | None = None,
    ) -> None:
        self._agent = agent
        self._scorer = scorer
        self._on_fail = on_fail
        self._injection_threshold = injection_threshold

    def run_sync(self, *args: Any, **kwargs: Any) -> Any:
        """Run a synchronous Pydantic AI agent call and gate its output."""
        prompt = _extract_pydantic_ai_prompt(args, kwargs)
        result = self._agent.run_sync(*args, **kwargs)
        text = _pydantic_ai_output_text(result)
        _score_and_gate(
            self._scorer,
            self._on_fail,
            prompt,
            text,
            injection_threshold=self._injection_threshold,
        )
        return result

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run an asynchronous Pydantic AI agent call and gate its output."""
        prompt = _extract_pydantic_ai_prompt(args, kwargs)
        result = await self._agent.run(*args, **kwargs)
        text = _pydantic_ai_output_text(result)
        await _ascore_and_gate(
            self._scorer,
            self._on_fail,
            prompt,
            text,
            injection_threshold=self._injection_threshold,
        )
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)


def _extract_pydantic_ai_prompt(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Extract prompt text from Pydantic AI run arguments."""
    prompt = kwargs.get("user_prompt")
    if prompt is None and args:
        prompt = args[0]
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list | tuple):
        return " ".join(_pydantic_ai_content_text(part) for part in prompt)
    return str(prompt)


def _pydantic_ai_content_text(content: Any) -> str:
    """Return text for one Pydantic AI prompt content part."""
    if isinstance(content, str):
        return content
    text = getattr(content, "content", None)
    if text is not None:
        return str(text)
    return str(content)


def _pydantic_ai_output_text(result: Any) -> str:
    """Serialise Pydantic AI run output for guard scoring."""
    output = getattr(result, "output", result)
    if isinstance(output, str):
        return output
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    model_dump_json = getattr(output, "model_dump_json", None)
    if callable(model_dump_json):
        return str(model_dump_json())
    if isinstance(output, dict | list | tuple):
        return json.dumps(output, sort_keys=True, default=str)
    return str(output)
