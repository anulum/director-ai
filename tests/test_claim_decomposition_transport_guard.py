# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Claim Decomposition Transport Guard Tests

"""Guard tests for the default provider transports.

These fake the ``openai``/``anthropic`` SDK modules in ``sys.modules`` to
exercise the real retry/cost/parse branches without network access. The
real-surface companion is ``tests/test_claim_decomposition.py`` (transport
injection, no patching).
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from director_ai.core.scoring.claim_decomposition import AtomicClaimDecomposer

_REPLY = json.dumps({"claims": ["A fact."]})


def _split(text: str) -> list[str]:
    return [text]


def _quiet_decomposer(provider: str, **kwargs) -> AtomicClaimDecomposer:
    with pytest.warns(UserWarning, match="third-party"):
        return AtomicClaimDecomposer(provider=provider, model="m", **kwargs)


class _FakeOpenAIModule(types.ModuleType):
    """Fake openai SDK whose client returns queued outcomes."""

    def __init__(self, outcomes, usage=None):
        super().__init__("openai")
        self.calls: list[dict] = []
        self._outcomes = list(outcomes)
        self._usage = usage

        module = self

        class _Completions:
            def create(self, **kwargs):
                module.calls.append(kwargs)
                outcome = module._outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                message = types.SimpleNamespace(content=outcome)
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)],
                    usage=module._usage,
                )

        class _Client:
            def __init__(self):
                self.chat = types.SimpleNamespace(completions=_Completions())

        self.OpenAI = _Client


class _FakeAnthropicModule(types.ModuleType):
    """Fake anthropic SDK whose client returns queued outcomes."""

    def __init__(self, outcomes, usage=None):
        super().__init__("anthropic")
        self.calls: list[dict] = []
        self._outcomes = list(outcomes)
        self._usage = usage

        module = self

        class _Messages:
            def create(self, **kwargs):
                module.calls.append(kwargs)
                outcome = module._outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return types.SimpleNamespace(content=outcome, usage=module._usage)

        class _Client:
            def __init__(self):
                self.messages = _Messages()

        self.Anthropic = _Client


class TestOpenAITransport:
    def test_happy_path_reports_cost(self, monkeypatch):
        usage = types.SimpleNamespace(prompt_tokens=11, completion_tokens=7)
        fake = _FakeOpenAIModule([_REPLY], usage=usage)
        monkeypatch.setitem(sys.modules, "openai", fake)
        costs: list[tuple[str, int, int]] = []
        decomposer = _quiet_decomposer(
            "openai",
            cost_callback=lambda model, p, c: costs.append((model, p, c)),
        )

        result = decomposer.decompose("passage", sentence_splitter=_split)

        assert result.backend == "llm"
        assert result.claims == ("A fact.",)
        assert costs == [("m", 11, 7)]
        assert fake.calls[0]["response_format"] == {"type": "json_object"}

    def test_transient_error_is_retried(self, monkeypatch):
        fake = _FakeOpenAIModule([ConnectionError("down"), _REPLY])
        monkeypatch.setitem(sys.modules, "openai", fake)
        monkeypatch.setattr("time.sleep", lambda seconds: None)
        decomposer = _quiet_decomposer("openai")

        result = decomposer.decompose("passage", sentence_splitter=_split)

        assert result.backend == "llm"
        assert len(fake.calls) == 2

    def test_exhausted_retries_fall_back(self, monkeypatch):
        fake = _FakeOpenAIModule([ConnectionError("down")] * 3)
        monkeypatch.setitem(sys.modules, "openai", fake)
        monkeypatch.setattr("time.sleep", lambda seconds: None)
        decomposer = _quiet_decomposer("openai")

        result = decomposer.decompose("passage", sentence_splitter=_split)

        assert result.backend == "sentence-fallback"
        assert len(fake.calls) == 3

    def test_missing_sdk_falls_back_without_retry(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "openai", None)
        decomposer = _quiet_decomposer("openai")

        result = decomposer.decompose("passage", sentence_splitter=_split)

        assert result.backend == "sentence-fallback"

    def test_empty_content_falls_back(self, monkeypatch):
        fake = _FakeOpenAIModule([None])
        monkeypatch.setitem(sys.modules, "openai", fake)
        decomposer = _quiet_decomposer("openai")

        result = decomposer.decompose("passage", sentence_splitter=_split)

        assert result.backend == "sentence-fallback"


class TestAnthropicTransport:
    def _block(self, text: str):
        return types.SimpleNamespace(text=text)

    def test_happy_path_reports_cost_and_splits_system(self, monkeypatch):
        usage = types.SimpleNamespace(input_tokens=13, output_tokens=5)
        fake = _FakeAnthropicModule([[self._block(_REPLY)]], usage=usage)
        monkeypatch.setitem(sys.modules, "anthropic", fake)
        costs: list[tuple[str, int, int]] = []
        decomposer = _quiet_decomposer(
            "anthropic",
            cost_callback=lambda model, p, c: costs.append((model, p, c)),
        )

        result = decomposer.decompose("passage", sentence_splitter=_split)

        assert result.backend == "llm"
        assert costs == [("m", 13, 5)]
        call = fake.calls[0]
        assert "information extraction engine" in call["system"]
        assert call["messages"][0]["role"] == "user"

    def test_transient_error_is_retried(self, monkeypatch):
        fake = _FakeAnthropicModule(
            [ConnectionError("down"), [self._block(_REPLY)]],
        )
        monkeypatch.setitem(sys.modules, "anthropic", fake)
        monkeypatch.setattr("time.sleep", lambda seconds: None)
        decomposer = _quiet_decomposer("anthropic")

        result = decomposer.decompose("passage", sentence_splitter=_split)

        assert result.backend == "llm"
        assert len(fake.calls) == 2

    def test_exhausted_retries_fall_back(self, monkeypatch):
        fake = _FakeAnthropicModule([ConnectionError("down")] * 3)
        monkeypatch.setitem(sys.modules, "anthropic", fake)
        monkeypatch.setattr("time.sleep", lambda seconds: None)
        decomposer = _quiet_decomposer("anthropic")

        result = decomposer.decompose("passage", sentence_splitter=_split)

        assert result.backend == "sentence-fallback"
        assert len(fake.calls) == 3

    def test_missing_sdk_falls_back_without_retry(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "anthropic", None)
        decomposer = _quiet_decomposer("anthropic")

        result = decomposer.decompose("passage", sentence_splitter=_split)

        assert result.backend == "sentence-fallback"

    def test_empty_content_falls_back(self, monkeypatch):
        fake = _FakeAnthropicModule([[]])
        monkeypatch.setitem(sys.modules, "anthropic", fake)
        decomposer = _quiet_decomposer("anthropic")

        result = decomposer.decompose("passage", sentence_splitter=_split)

        assert result.backend == "sentence-fallback"

    def test_block_without_text_attribute_falls_back(self, monkeypatch):
        fake = _FakeAnthropicModule([[types.SimpleNamespace(kind="image")]])
        monkeypatch.setitem(sys.modules, "anthropic", fake)
        decomposer = _quiet_decomposer("anthropic")

        result = decomposer.decompose("passage", sentence_splitter=_split)

        assert result.backend == "sentence-fallback"
