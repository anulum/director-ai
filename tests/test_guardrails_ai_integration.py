# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Guardrails AI integration tests

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from director_ai.core.types import CoherenceScore


def _install_fake_guardrails(monkeypatch):
    validators = types.ModuleType("guardrails.validators")
    registered: dict[str, tuple[str, type]] = {}

    class ValidationResult:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class PassResult(ValidationResult):
        def __init__(self, **kwargs):
            super().__init__(outcome="pass", **kwargs)

    class FailResult(ValidationResult):
        def __init__(self, error_message="", **kwargs):
            super().__init__(outcome="fail", error_message=error_message, **kwargs)

    class Validator:
        def __init__(self, on_fail=None, **kwargs):
            self.on_fail = on_fail
            self.init_kwargs = kwargs

        def validate(self, value, metadata):
            return self._validate(value, metadata)

    def register_validator(name, data_type, has_guardrails_endpoint=False):
        def decorator(cls):
            registered[name] = (data_type, cls)
            cls.registered_name = name
            cls.has_guardrails_endpoint = has_guardrails_endpoint
            return cls

        return decorator

    validators.FailResult = FailResult
    validators.PassResult = PassResult
    validators.ValidationResult = ValidationResult
    validators.Validator = Validator
    validators.register_validator = register_validator

    guardrails = types.ModuleType("guardrails")
    monkeypatch.setitem(sys.modules, "guardrails", guardrails)
    monkeypatch.setitem(sys.modules, "guardrails.validators", validators)
    return registered


def _fake_review_pass(self, prompt, response, session=None, tenant_id=""):
    assert prompt == "What is the support SLA?"
    assert response == "The support SLA is four hours."
    assert tenant_id == "tenant-a"
    cs = CoherenceScore(score=0.94, approved=True, h_logical=0.01, h_factual=0.02)
    return True, cs


def _fake_review_fail(self, prompt, response, session=None, tenant_id=""):
    assert prompt == "What is the support SLA?"
    assert response == "The SLA is one minute and includes free refunds."
    assert tenant_id == "tenant-a"
    cs = CoherenceScore(score=0.12, approved=False, h_logical=0.7, h_factual=0.8)
    return False, cs


def test_build_guardrails_validator_registers_dependency_light_class(monkeypatch):
    registered = _install_fake_guardrails(monkeypatch)
    from director_ai.integrations.guardrails_ai import build_guardrails_validator

    validator = build_guardrails_validator(
        facts={"sla": "Support SLA is four hours."},
        threshold=0.4,
        use_nli=False,
    )

    assert "director-ai-coherence" in registered
    assert registered["director-ai-coherence"][0] == "string"
    assert validator.scorer is not None


def test_guardrails_validator_passes_with_tenant_safe_metadata(monkeypatch):
    _install_fake_guardrails(monkeypatch)
    from director_ai.core import CoherenceScorer
    from director_ai.integrations.guardrails_ai import build_guardrails_validator

    validator = build_guardrails_validator(
        facts={"sla": "Support SLA is four hours."},
        threshold=0.4,
        use_nli=False,
    )

    with patch.object(CoherenceScorer, "review", _fake_review_pass):
        result = validator.validate(
            "The support SLA is four hours.",
            {"prompt": "What is the support SLA?", "tenant_id": "tenant-a"},
        )

    assert result.outcome == "pass"
    assert result.metadata["director_ai"]["approved"] is True
    assert result.metadata["director_ai"]["score"] == pytest.approx(0.94)


def test_guardrails_validator_fails_without_leaking_raw_output(monkeypatch):
    _install_fake_guardrails(monkeypatch)
    from director_ai.core import CoherenceScorer
    from director_ai.integrations.guardrails_ai import build_guardrails_validator

    validator = build_guardrails_validator(
        facts={"sla": "Support SLA is four hours."},
        threshold=0.4,
        use_nli=False,
    )

    with patch.object(CoherenceScorer, "review", _fake_review_fail):
        result = validator.validate(
            "The SLA is one minute and includes free refunds.",
            {"query": "What is the support SLA?", "tenant": "tenant-a"},
        )

    assert result.outcome == "fail"
    assert "one minute" not in result.error_message
    assert result.metadata["director_ai"]["approved"] is False
    assert result.metadata["director_ai"]["h_factual"] == pytest.approx(0.8)


def test_attach_guardrails_validator_uses_guard_use(monkeypatch):
    _install_fake_guardrails(monkeypatch)
    from director_ai.integrations.guardrails_ai import attach_guardrails_validator

    class FakeGuard:
        def __init__(self):
            self.validators = []

        def use(self, validator):
            self.validators.append(validator)
            return self

    guard = FakeGuard()
    returned = attach_guardrails_validator(
        guard,
        facts={"sla": "Support SLA is four hours."},
        threshold=0.4,
    )

    assert returned is guard
    assert len(guard.validators) == 1


def test_attach_guardrails_validator_rejects_guard_without_use(monkeypatch):
    _install_fake_guardrails(monkeypatch)
    from director_ai.integrations.guardrails_ai import attach_guardrails_validator

    with pytest.raises(TypeError, match="use"):
        attach_guardrails_validator(object())


def test_guardrails_validator_reads_chat_metadata_and_coerces_values(monkeypatch):
    _install_fake_guardrails(monkeypatch)
    from director_ai.core import CoherenceScorer
    from director_ai.integrations.guardrails_ai import build_guardrails_validator

    validator = build_guardrails_validator(threshold=0.4, use_nli=False)

    def review(self, prompt, response, session=None, tenant_id=""):
        assert prompt == "What is the support SLA? Include escalation."
        assert response == ""
        assert tenant_id == ""
        cs = CoherenceScore(score=0.93, approved=True, h_logical=0.01, h_factual=0.02)
        return True, cs

    with patch.object(CoherenceScorer, "review", review):
        result = validator.validate(
            None,
            {
                "messages": [
                    {"role": "assistant", "content": "Ignored."},
                    {
                        "role": "user",
                        "content": [
                            {"text": "What is the support SLA?"},
                            {"text": "Include escalation."},
                        ],
                    },
                ],
            },
        )

    assert result.outcome == "pass"


def test_guardrails_result_fallback_supports_minimal_result_constructors(monkeypatch):
    registered = _install_fake_guardrails(monkeypatch)

    class MinimalFailResult:
        def __init__(self, error_message=""):
            self.outcome = "fail"
            self.error_message = error_message

    import guardrails.validators as validators

    validators.FailResult = MinimalFailResult

    from director_ai.core import CoherenceScorer
    from director_ai.integrations.guardrails_ai import build_guardrails_validator

    validator = build_guardrails_validator(threshold=0.4, use_nli=False)

    with patch.object(CoherenceScorer, "review", _fake_review_fail):
        result = validator.validate(
            "The SLA is one minute and includes free refunds.",
            {"input": "What is the support SLA?", "tenant_id": "tenant-a"},
        )

    assert registered["director-ai-coherence"][1].registered_name
    assert result.outcome == "fail"
    assert result.error_message.startswith("Director-AI coherence validation failed")


def test_guardrails_adapter_raises_actionable_error_without_optional_dependency(
    monkeypatch,
):
    monkeypatch.delitem(sys.modules, "guardrails.validators", raising=False)
    monkeypatch.delitem(sys.modules, "guardrails", raising=False)
    from director_ai.integrations.guardrails_ai import build_guardrails_validator

    with pytest.raises(ImportError, match="guardrails-ai"):
        build_guardrails_validator()


def test_guardrails_helpers_cover_metadata_and_store_edge_cases():
    from director_ai.core import GroundTruthStore
    from director_ai.integrations import guardrails_ai

    store = GroundTruthStore()
    assert guardrails_ai._build_store(facts={"unused": "fact"}, store=store) is store

    assert guardrails_ai._coerce_text(42) == "42"
    assert guardrails_ai._prompt_from_metadata(None) == ""
    assert guardrails_ai._prompt_from_metadata({"messages": "not-a-list"}) == ""
    assert guardrails_ai._tenant_from_metadata(None) == ""

    assert (
        guardrails_ai._prompt_from_messages(
            [
                "not-a-dict",
                {"role": "assistant", "content": "ignored"},
                {"role": "user", "content": ""},
                {"role": "user", "content": [{"text": "  "}, {"image": "ignored"}]},
                {"role": "user", "content": "  What is the SLA?  "},
            ]
        )
        == "What is the SLA?"
    )

    assert (
        guardrails_ai._prompt_from_messages(
            [
                {"role": "assistant", "content": "ignored"},
                {"role": "user", "content": [{"text": "  "}, {"image": "ignored"}]},
                {"role": "user", "content": 123},
                "trailing-non-dict",
            ]
        )
        == ""
    )
