# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for hybrid LLM-judge hardening."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
from typing import ClassVar, cast

import pytest

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.scoring._llm_judge import LLMJudge
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


@dataclass(frozen=True)
class _JudgeCall:
    """Captured OpenAI-compatible judge request."""

    model: str
    messages: list[dict[str, str]]
    max_tokens: int
    response_format: dict[str, str]


class _OpenAICompletions:
    """OpenAI chat-completions protocol surface used by ``LLMJudge``."""

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        response_format: dict[str, str],
    ) -> SimpleNamespace:
        """Capture the request and return the next configured verdict."""
        _OpenAIClient.calls.append(
            _JudgeCall(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format=response_format,
            ),
        )
        reply = _OpenAIClient.replies.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=reply)),
            ],
            usage=SimpleNamespace(prompt_tokens=31, completion_tokens=7),
        )


class _OpenAIChat:
    """Container matching ``client.chat.completions``."""

    def __init__(self) -> None:
        self.completions = _OpenAICompletions()


class _OpenAIClient:
    """Small OpenAI-compatible client installed as a local protocol module."""

    calls: ClassVar[list[_JudgeCall]] = []
    replies: ClassVar[list[str]] = []

    def __init__(self) -> None:
        self.chat = _OpenAIChat()


def _install_openai_protocol_fake(
    monkeypatch: pytest.MonkeyPatch,
    *replies: str,
) -> None:
    """Install a deterministic OpenAI-compatible module for public calls."""
    _OpenAIClient.calls = []
    _OpenAIClient.replies = list(replies)
    module = ModuleType("openai")
    module.__spec__ = ModuleSpec("openai", loader=None)
    module.__dict__["OpenAI"] = _OpenAIClient
    monkeypatch.setitem(sys.modules, "openai", module)


def test_hybrid_hardening_unit_guard_declares_this_companion() -> None:
    """The helper-heavy hybrid guard should point at this public companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_hybrid_hardening.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_hybrid_hardening_real_surface.py" in reason


def test_public_llm_judge_check_uses_structured_json_provider_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``LLMJudge.check`` should call the provider with structured JSON I/O."""
    _install_openai_protocol_fake(
        monkeypatch,
        '{"verdict": "NO", "confidence": 90}',
    )
    costs: list[tuple[str, int, int]] = []

    def record_cost(model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record the public cost-callback invocation."""
        costs.append((model, prompt_tokens, completion_tokens))

    judge = LLMJudge(
        provider="openai",
        model="gpt-test-judge",
        cost_callback=record_cost,
    )
    adjusted = judge.check(
        "Refund approvals require a signed operator receipt.",
        "Refund approvals can be issued without a receipt.",
        0.5,
    )

    assert adjusted == pytest.approx(0.581)
    assert costs == [("gpt-test-judge", 31, 7)]
    assert len(_OpenAIClient.calls) == 1
    call = _OpenAIClient.calls[0]
    assert call.model == "gpt-test-judge"
    assert call.max_tokens == 50
    assert call.response_format == {"type": "json_object"}
    assert [message["role"] for message in call.messages] == ["system", "user"]
    payload = json.loads(call.messages[1]["content"])
    assert payload == {
        "prompt": "Refund approvals require a signed operator receipt.",
        "response": "Refund approvals can be issued without a receipt.",
        "nli_divergence": 0.5,
        "question": "Is the response factually correct relative to the prompt?",
        "schema": {"verdict": "YES|NO", "confidence": "0-100"},
    }


def test_public_scorer_hybrid_path_redacts_before_external_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``CoherenceScorer`` should redact PII before the external judge call."""
    _install_openai_protocol_fake(
        monkeypatch,
        '{"verdict": "NO", "confidence": 90}',
    )
    store = GroundTruthStore()
    store.add(
        "transfer approval",
        "Transfer approvals require a signed operator receipt.",
    )
    scorer = CoherenceScorer(
        use_nli=False,
        ground_truth_store=store,
        llm_judge_enabled=True,
        llm_judge_provider="openai",
        llm_judge_model="gpt-test-judge",
        llm_judge_confidence_threshold=0.51,
        scorer_backend="hybrid",
        privacy_mode=True,
    )

    score = scorer.calculate_factual_divergence(
        "transfer approval",
        "Send the receipt to jane@example.com and SSN 123-45-6789.",
    )

    assert score == pytest.approx(0.9338333333333333)
    assert len(_OpenAIClient.calls) == 1
    payload = json.loads(_OpenAIClient.calls[0].messages[1]["content"])
    response = cast(str, payload["response"])
    assert "jane@example.com" not in response
    assert "123-45-6789" not in response
    assert "[EMAIL]" in response
    assert "[SSN]" in response


def test_public_scorer_hybrid_path_caches_repeated_judge_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated public scorer calls should reuse the judge cache."""
    _install_openai_protocol_fake(
        monkeypatch,
        '{"verdict": "YES", "confidence": 80}',
    )
    store = GroundTruthStore()
    store.add(
        "weather",
        "Today is partly cloudy with mild temperatures.",
    )
    scorer = CoherenceScorer(
        use_nli=False,
        ground_truth_store=store,
        llm_judge_enabled=True,
        llm_judge_provider="openai",
        llm_judge_model="gpt-test-judge",
        llm_judge_confidence_threshold=0.51,
        scorer_backend="hybrid",
    )

    first = scorer.calculate_factual_divergence(
        "weather",
        "The weather will be sunny and warm tomorrow.",
    )
    second = scorer.calculate_factual_divergence(
        "weather",
        "The weather will be sunny and warm tomorrow.",
    )

    assert first == pytest.approx(0.808)
    assert second == pytest.approx(first)
    assert len(_OpenAIClient.calls) == 1
