# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — LLM Judge Tests
"""Module-specific tests for LLM judge escalation."""

from __future__ import annotations

import logging
import sys
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from director_ai.core.scoring._llm_judge import LLMJudge


def test_local_provider_initializes_configured_model(monkeypatch):
    calls = []

    def fake_init(self, model_path, device=None, *, model_revision=None):
        calls.append((model_path, device, model_revision))
        self._local_judge_model = object()
        self._local_judge_tokenizer = object()
        self._local_judge_device = device or "cpu"

    monkeypatch.setattr(LLMJudge, "_init_local_judge", fake_init)

    judge = LLMJudge(
        provider="local",
        model="local/judge",
        model_revision="abc123",
        device="cuda:0",
    )

    assert calls == [("local/judge", "cuda:0", "abc123")]
    assert judge.enabled is True


@pytest.mark.parametrize(
    ("task_type", "score", "expected"),
    [
        ("dialogue", 0.20, True),
        ("fact_check", 0.29, False),
        ("unknown", 0.74, True),
    ],
)
def test_should_escalate_uses_task_specific_thresholds(task_type, score, expected):
    judge = LLMJudge(provider="openai", confidence_threshold=0.25)

    assert judge.should_escalate(score, task_type=task_type) is expected


def test_check_dispatches_local_provider(monkeypatch):
    judge = LLMJudge(provider="local")
    monkeypatch.setattr(judge, "_local_judge_check", Mock(return_value=0.42))

    assert judge.check("prompt", "response", 0.5) == 0.42
    judge._local_judge_check.assert_called_once_with("prompt", "response", 0.5)


def test_check_passes_redactor_to_external_provider(monkeypatch):
    judge = LLMJudge(provider="openai")
    redactor = Mock()
    external_check = Mock(return_value=0.33)
    monkeypatch.setattr(judge, "_llm_judge_check", external_check)

    assert judge.check("prompt", "response", 0.5, redactor=redactor) == 0.33
    external_check.assert_called_once_with(
        "prompt",
        "response",
        0.5,
        redactor=redactor,
    )


def test_local_judge_check_falls_back_without_model():
    judge = LLMJudge(provider="local")

    assert judge._local_judge_check("prompt", "response", 0.61) == 0.61


def test_local_judge_check_delegates_when_model_is_available(monkeypatch):
    judge = LLMJudge(provider="local")
    judge._local_judge_model = object()
    judge._local_judge_tokenizer = object()
    local_infer = Mock(return_value=0.27)
    monkeypatch.setattr(judge, "_local_judge_infer", local_infer)

    assert judge._local_judge_check("prompt", "response", 0.61) == 0.27
    local_infer.assert_called_once_with("prompt", "response", 0.61)


def test_external_judge_applies_privacy_redactor_before_call(monkeypatch):
    captured = {}
    judge = LLMJudge(provider="openai", model="gpt-4o-mini", privacy_mode=True)

    def fake_call(model, messages, fallback):
        captured["messages"] = messages
        return '{"verdict": "NO", "confidence": 60}'

    monkeypatch.setattr(judge, "_call_llm_judge", fake_call)

    result = judge._llm_judge_check(
        "name: Alice",
        "email: alice@example.test",
        0.4,
        redactor=lambda text: text.replace("Alice", "[REDACTED]"),
    )

    assert result > 0.4
    assert "Alice" not in captured["messages"][1]["content"]
    assert "[REDACTED]" in captured["messages"][1]["content"]


# -- External data-egress warning (0H) ---------------------------------------


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_external_provider_warns_on_construction(provider):
    with pytest.warns(UserWarning, match="EXTERNAL"):
        LLMJudge(provider=provider, model="m")


def test_warning_reports_privacy_mode_off():
    with pytest.warns(UserWarning, match="OFF \\(no redaction\\)"):
        LLMJudge(provider="openai")


def test_warning_reports_privacy_mode_on():
    with pytest.warns(UserWarning, match="on \\(PII redacted\\)"):
        LLMJudge(provider="openai", privacy_mode=True)


def test_local_provider_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LLMJudge(provider="local")  # no model → no load, no egress
        LLMJudge(provider="")
    assert not [w for w in caught if "EXTERNAL" in str(w.message)]


def test_external_egress_is_logged(monkeypatch, caplog):
    judge = LLMJudge(provider="openai", model="gpt-4o-mini")
    monkeypatch.setattr(
        judge,
        "_call_llm_judge",
        lambda m, msgs, fb: '{"verdict": "YES", "confidence": 80}',
    )
    with caplog.at_level(logging.INFO, logger="DirectorAI"):
        judge._llm_judge_check("prompt", "response", 0.4)
    assert any("external egress to openai" in r.message for r in caplog.records)


def test_anthropic_call_uses_system_prompt_and_reports_usage(monkeypatch):
    calls = []
    callback = Mock()

    class FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(text='{"verdict": "YES", "confidence": 70}')],
                usage=SimpleNamespace(input_tokens=12, output_tokens=4),
            )

    fake_client = SimpleNamespace(messages=FakeMessages())
    fake_anthropic = SimpleNamespace(Anthropic=lambda: fake_client)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    judge = LLMJudge(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        cost_callback=callback,
    )
    messages = [
        {"role": "system", "content": "judge only"},
        {"role": "user", "content": "payload"},
    ]

    reply = judge._call_llm_judge("claude-haiku-4-5-20251001", messages, 0.5)

    assert reply == '{"verdict": "YES", "confidence": 70}'
    assert calls == [
        {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 50,
            "system": "judge only",
            "messages": [{"role": "user", "content": "payload"}],
        },
    ]
    callback.assert_called_once_with("claude-haiku-4-5-20251001", 12, 4)


def test_anthropic_call_wraps_plain_prompt_and_handles_empty_content(monkeypatch):
    calls = []

    class FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=[], usage=None)

    fake_client = SimpleNamespace(messages=FakeMessages())
    fake_anthropic = SimpleNamespace(Anthropic=lambda: fake_client)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    judge = LLMJudge(provider="anthropic", model="claude-haiku-4-5-20251001")

    assert judge._call_llm_judge("claude-haiku-4-5-20251001", "plain", 0.5) == ""
    assert calls[0]["system"] == ""
    assert calls[0]["messages"] == [{"role": "user", "content": "plain"}]


def test_unknown_provider_returns_no_judge_reply():
    judge = LLMJudge(provider="custom", model="custom-model")

    assert judge._call_llm_judge("custom-model", "prompt", 0.5) is None


@pytest.mark.parametrize(
    "reply",
    [
        '{"verdict": "MAYBE", "confidence": 50}',
        '{"verdict": "YES", "confidence": 101}',
        '{"confidence": 50}',
        "not json",
    ],
)
def test_strict_parser_rejects_invalid_judge_payloads(reply):
    assert LLMJudge._parse_judge_reply_strict(reply) is None


def test_lenient_parser_clamps_json_confidence_and_falls_back_to_text():
    assert LLMJudge._parse_judge_reply('{"verdict": "YES", "confidence": 140}') == (
        True,
        1.0,
    )
    assert LLMJudge._parse_judge_reply("yes, supported") == (True, 0.5)
