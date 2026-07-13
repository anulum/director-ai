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
from typing import Any
from unittest.mock import Mock

import pytest

from director_ai.core.scoring._llm_judge import LLMJudge


def test_local_provider_initializes_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local provider construction should load the configured checkpoint once."""
    calls: list[tuple[str, str | None, str | None]] = []

    def fake_init(
        self: LLMJudge,
        model_path: str,
        device: str | None = None,
        *,
        model_revision: str | None = None,
    ) -> None:
        """Capture local model initialization without loading a real model."""
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
def test_should_escalate_uses_task_specific_thresholds(
    task_type: str,
    score: float,
    expected: bool,
) -> None:
    """Task-specific thresholds should override the default confidence gate."""
    judge = LLMJudge(provider="openai", confidence_threshold=0.25)

    assert judge.should_escalate(score, task_type=task_type) is expected


def test_check_dispatches_local_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public check API should route local providers to local inference."""
    judge = LLMJudge(provider="local")
    local_check = Mock(return_value=0.42)
    monkeypatch.setattr(judge, "_local_judge_check", local_check)

    assert judge.check("prompt", "response", 0.5) == 0.42
    local_check.assert_called_once_with("prompt", "response", 0.5)


def test_check_passes_redactor_to_external_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public check API should forward privacy redactors to remote judges."""
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


def test_local_judge_check_falls_back_without_model() -> None:
    """Local judge checks should preserve fallback scores without a model."""
    judge = LLMJudge(provider="local")

    assert judge._local_judge_check("prompt", "response", 0.61) == 0.61


def test_local_judge_check_delegates_when_model_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local judge checks should delegate when model and tokenizer are loaded."""
    judge = LLMJudge(provider="local")
    judge._local_judge_model = object()
    judge._local_judge_tokenizer = object()
    local_infer = Mock(return_value=0.27)
    monkeypatch.setattr(judge, "_local_judge_infer", local_infer)

    assert judge._local_judge_check("prompt", "response", 0.61) == 0.27
    local_infer.assert_called_once_with("prompt", "response", 0.61)


def test_external_judge_applies_privacy_redactor_before_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External judge prompts should be redacted before provider egress."""
    captured: list[list[dict[str, str]]] = []
    judge = LLMJudge(provider="openai", model="gpt-4o-mini", privacy_mode=True)

    def fake_call(
        model: str,
        messages: str | list[dict[str, str]],
        fallback: float,
        max_tokens: int = 50,
    ) -> str:
        """Capture provider payloads and return a rejecting judge verdict."""
        assert model == "gpt-4o-mini"
        assert fallback == 0.4
        assert isinstance(messages, list)
        captured.append(messages)
        return '{"verdict": "NO", "confidence": 60}'

    monkeypatch.setattr(judge, "_call_llm_judge", fake_call)

    result = judge._llm_judge_check(
        "name: Alice",
        "email: alice@example.test",
        0.4,
        redactor=lambda text: text.replace("Alice", "[REDACTED]"),
    )

    assert result > 0.4
    assert "Alice" not in captured[0][1]["content"]
    assert "[REDACTED]" in captured[0][1]["content"]


# -- External data-egress warning (0H) ---------------------------------------


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_external_provider_warns_on_construction(provider: str) -> None:
    """Remote provider construction should warn about external data egress."""
    with pytest.warns(UserWarning, match="EXTERNAL"):
        LLMJudge(provider=provider, model="m")


def test_warning_reports_privacy_mode_off() -> None:
    """External-provider warnings should report unredacted privacy mode."""
    with pytest.warns(UserWarning, match="OFF \\(no redaction\\)"):
        LLMJudge(provider="openai")


def test_warning_reports_privacy_mode_on() -> None:
    """External-provider warnings should report redacted privacy mode."""
    with pytest.warns(UserWarning, match="on \\(PII redacted\\)"):
        LLMJudge(provider="openai", privacy_mode=True)


def test_local_provider_does_not_warn() -> None:
    """Local or disabled providers should not emit data-egress warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LLMJudge(provider="local")  # no model → no load, no egress
        LLMJudge(provider="")
    assert not [w for w in caught if "EXTERNAL" in str(w.message)]


def test_openai_call_reports_token_usage_to_cost_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI calls should report prompt and completion token usage."""
    costs: list[tuple[str, int, int]] = []
    with pytest.warns(UserWarning):
        judge = LLMJudge(
            provider="openai",
            model="gpt-4o-mini",
            cost_callback=lambda model, prompt, completion: costs.append(
                (model, prompt, completion)
            ),
        )

    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=7)
    message = SimpleNamespace(content='{"verdict": "YES", "confidence": 80}')
    result = SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=message)])

    def create_completion(**kwargs: Any) -> SimpleNamespace:
        """Return a deterministic OpenAI-compatible chat completion."""
        assert kwargs["model"] == "gpt-4o-mini"
        return result

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion))
    )

    def openai_client() -> SimpleNamespace:
        """Return the deterministic OpenAI-compatible client."""
        return client

    fake_openai = SimpleNamespace(OpenAI=openai_client)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    content = judge._call_llm_judge("gpt-4o-mini", "judge this", fallback=0.5)

    assert content == '{"verdict": "YES", "confidence": 80}'
    assert costs == [("gpt-4o-mini", 12, 7)]


def test_external_egress_is_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """External judge calls should emit an explicit egress audit log."""
    judge = LLMJudge(provider="openai", model="gpt-4o-mini")

    def fake_call(
        model: str,
        messages: str | list[dict[str, str]],
        fallback: float,
        max_tokens: int = 50,
    ) -> str:
        """Return a deterministic accepting judge verdict."""
        assert model == "gpt-4o-mini"
        assert isinstance(messages, list)
        assert fallback == 0.4
        return '{"verdict": "YES", "confidence": 80}'

    monkeypatch.setattr(
        judge,
        "_call_llm_judge",
        fake_call,
    )
    with caplog.at_level(logging.INFO, logger="DirectorAI"):
        judge._llm_judge_check("prompt", "response", 0.4)
    assert any("external egress to openai" in r.message for r in caplog.records)


def test_anthropic_call_uses_system_prompt_and_reports_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic calls should split system prompts and report token usage."""
    calls: list[dict[str, Any]] = []
    callback = Mock()

    class FakeMessages:
        """Anthropic messages endpoint protocol fake."""

        def create(self, **kwargs: Any) -> SimpleNamespace:
            """Capture the Anthropic request and return a deterministic reply."""
            calls.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(text='{"verdict": "YES", "confidence": 70}')],
                usage=SimpleNamespace(input_tokens=12, output_tokens=4),
            )

    fake_client = SimpleNamespace(messages=FakeMessages())

    def anthropic_client() -> SimpleNamespace:
        """Return the deterministic Anthropic-compatible client."""
        return fake_client

    fake_anthropic = SimpleNamespace(Anthropic=anthropic_client)
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


def test_anthropic_call_wraps_plain_prompt_and_handles_empty_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic calls should wrap string prompts and tolerate empty content."""
    calls: list[dict[str, Any]] = []

    class FakeMessages:
        """Anthropic messages endpoint protocol fake."""

        def create(self, **kwargs: Any) -> SimpleNamespace:
            """Capture the Anthropic request and return empty content."""
            calls.append(kwargs)
            return SimpleNamespace(content=[], usage=None)

    fake_client = SimpleNamespace(messages=FakeMessages())

    def anthropic_client() -> SimpleNamespace:
        """Return the deterministic Anthropic-compatible client."""
        return fake_client

    fake_anthropic = SimpleNamespace(Anthropic=anthropic_client)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    judge = LLMJudge(provider="anthropic", model="claude-haiku-4-5-20251001")

    assert judge._call_llm_judge("claude-haiku-4-5-20251001", "plain", 0.5) == ""
    assert calls[0]["system"] == ""
    assert calls[0]["messages"] == [{"role": "user", "content": "plain"}]


def test_unknown_provider_returns_no_judge_reply() -> None:
    """Unknown providers should fail closed without a judge reply."""
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
def test_strict_parser_rejects_invalid_judge_payloads(reply: str) -> None:
    """The strict parser should reject malformed or out-of-range payloads."""
    assert LLMJudge._parse_judge_reply_strict(reply) is None


def test_lenient_parser_clamps_json_confidence_and_falls_back_to_text() -> None:
    """The lenient parser should clamp JSON and infer simple text verdicts."""
    assert LLMJudge._parse_judge_reply('{"verdict": "YES", "confidence": 140}') == (
        True,
        1.0,
    )
    assert LLMJudge._parse_judge_reply("yes, supported") == (True, 0.5)


# -- WCA-5: rubric scoring + ensemble aggregation -----------------------------


class TestRubricJudge:
    def test_constructor_rejects_bad_rubric_and_ensemble(self) -> None:
        with pytest.raises(ValueError, match="rubric must be a boolean"):
            LLMJudge(rubric="yes")
        with pytest.raises(ValueError, match="ensemble_n must be an integer"):
            LLMJudge(ensemble_n=True)
        with pytest.raises(ValueError, match="between 1 and 5"):
            LLMJudge(ensemble_n=0)
        with pytest.raises(ValueError, match="between 1 and 5"):
            LLMJudge(ensemble_n=6)

    def test_rubric_messages_carry_dimension_schema(self) -> None:
        messages = LLMJudge._build_rubric_messages("prompt", "response", 0.4)

        assert '"grounding"' in messages[0]["content"]
        assert '"fabrication"' in messages[0]["content"]
        assert '"contradiction"' in messages[0]["content"]
        assert "untrusted data" in messages[0]["content"]
        assert '"nli_divergence": 0.4' in messages[1]["content"]

    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            # composite = (0.4*5 + 0.3*5 + 0.3*5)/5 = 1.0 -> agrees
            (
                '{"grounding":5,"fabrication":0,"contradiction":0,"confidence":90}',
                (True, 0.9),
            ),
            # composite = (0.4*0 + 0.3*0 + 0.3*0)/5 = 0.0 -> disagrees
            (
                '{"grounding":0,"fabrication":5,"contradiction":5,"confidence":80}',
                (False, 0.8),
            ),
            # boundary: composite exactly 0.5 counts as agrees
            (
                '{"grounding":2,"fabrication":2,"contradiction":2,"confidence":50}',
                (True, 0.5),
            ),
        ],
    )
    def test_rubric_parser_composites(self, reply: str, expected: tuple) -> None:
        agrees, confidence = LLMJudge._parse_rubric_reply_strict(reply)

        assert agrees is expected[0]
        assert confidence == pytest.approx(expected[1])

    @pytest.mark.parametrize(
        "reply",
        [
            "not json",
            "[1, 2]",
            '{"grounding":6,"fabrication":0,"contradiction":0,"confidence":50}',
            '{"grounding":5,"fabrication":-1,"contradiction":0,"confidence":50}',
            '{"grounding":5,"fabrication":0,"contradiction":0,"confidence":101}',
            '{"grounding":5,"fabrication":0,"confidence":50}',
            '{"grounding":"high","fabrication":0,"contradiction":0,"confidence":5}',
        ],
    )
    def test_rubric_parser_rejects_invalid(self, reply: str) -> None:
        assert LLMJudge._parse_rubric_reply_strict(reply) is None

    def test_rubric_check_uses_rubric_messages_and_budget(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        judge = LLMJudge(provider="openai", model="gpt-4o-mini", rubric=True)
        seen: dict[str, Any] = {}

        def fake_call(
            model: str,
            messages: str | list[dict[str, str]],
            fallback: float,
            max_tokens: int = 50,
        ) -> str:
            seen["max_tokens"] = max_tokens
            seen["system"] = messages[0]["content"]
            return '{"grounding":5,"fabrication":0,"contradiction":0,"confidence":90}'

        monkeypatch.setattr(judge, "_call_llm_judge", fake_call)

        result = judge._llm_judge_check("prompt", "response", 0.5)

        assert seen["max_tokens"] == 120
        assert '"grounding"' in seen["system"]
        assert result < 0.5  # agreeing judge pulls divergence down


class TestEnsembleAggregation:
    def _judge(self, replies: list[str | None], **kwargs: Any) -> LLMJudge:
        judge = LLMJudge(provider="openai", model="gpt-4o-mini", **kwargs)
        queue = list(replies)

        def fake_call(
            model: str,
            messages: str | list[dict[str, str]],
            fallback: float,
            max_tokens: int = 50,
        ) -> str | None:
            return queue.pop(0)

        judge._call_llm_judge = fake_call  # type: ignore[method-assign]
        return judge

    def test_majority_vote_wins(self) -> None:
        judge = self._judge(
            [
                '{"verdict": "YES", "confidence": 80}',
                '{"verdict": "NO", "confidence": 90}',
                '{"verdict": "YES", "confidence": 60}',
            ],
            ensemble_n=3,
        )

        result = judge._llm_judge_check("prompt", "response", 0.5)

        # Majority YES; conf = mean(0.8, 0.6) * agreement(2/3)
        assert result < 0.5

    def test_disagreement_damps_confidence(self) -> None:
        unanimous = self._judge(
            ['{"verdict": "YES", "confidence": 80}'] * 3,
            ensemble_n=3,
        )
        split = self._judge(
            [
                '{"verdict": "YES", "confidence": 80}',
                '{"verdict": "YES", "confidence": 80}',
                '{"verdict": "NO", "confidence": 80}',
            ],
            ensemble_n=3,
        )

        unanimous_score = unanimous._llm_judge_check("p", "r", 0.5)
        split_score = split._llm_judge_check("p2", "r2", 0.5)

        # Split panel -> weaker judge influence -> closer to the NLI 0.5.
        assert abs(split_score - 0.5) < abs(unanimous_score - 0.5)

    def test_tie_follows_first_verdict(self) -> None:
        judge = self._judge(
            [
                '{"verdict": "NO", "confidence": 70}',
                '{"verdict": "YES", "confidence": 70}',
                "not json",
            ],
            ensemble_n=3,
        )

        result = judge._llm_judge_check("prompt", "response", 0.5)

        # Two valid verdicts tie -> first (NO) wins -> divergence rises.
        assert result > 0.5

    def test_all_invalid_falls_back_to_nli(self) -> None:
        judge = self._judge(["not json", None, "{}"], ensemble_n=3)

        assert judge._llm_judge_check("prompt", "response", 0.42) == 0.42

    def test_single_call_path_unchanged(self) -> None:
        judge = self._judge(['{"verdict": "NO", "confidence": 100}'])

        result = judge._llm_judge_check("prompt", "response", 0.5)

        assert result > 0.5
