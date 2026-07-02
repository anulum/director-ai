# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Guardrails AI real-surface tests
"""Protocol-preserving Guardrails AI adapter coverage."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _install_guardrails_protocol_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "guardrails"
    package.mkdir()
    (package / "__init__.py").write_text(
        """
from .validators import ValidationResult


class Guard:
    def __init__(self):
        self.validators = []

    def use(self, validator):
        self.validators.append(validator)
        return self

    def parse(self, value, *, metadata=None):
        if not self.validators:
            raise RuntimeError("no validators attached")
        return self.validators[-1].validate(value, metadata or {})
""".lstrip(),
        encoding="utf-8",
    )
    (package / "validators.py").write_text(
        """
REGISTERED = {}


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

    def validate(self, value, metadata=None):
        return self._validate(value, metadata or {})


def register_validator(name, data_type, has_guardrails_endpoint=False):
    def decorator(cls):
        REGISTERED[name] = {
            "data_type": data_type,
            "validator": cls,
            "has_guardrails_endpoint": has_guardrails_endpoint,
        }
        cls.registered_name = name
        cls.registered_data_type = data_type
        return cls

    return decorator
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()


def test_guardrails_guard_parse_uses_real_director_scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guardrails-style parsing should run the production coherence scorer."""
    _install_guardrails_protocol_package(tmp_path, monkeypatch)

    from guardrails import Guard

    from director_ai.integrations.guardrails_ai import attach_guardrails_validator

    guard = attach_guardrails_validator(
        Guard(),
        facts={"sla": "Support SLA is four hours."},
        threshold=0.4,
        use_nli=False,
    )

    result = guard.parse(
        "The support SLA is four hours.",
        metadata={
            "prompt": "What is the support SLA?",
            "tenant_id": "tenant-a",
        },
    )

    assert result.outcome == "pass"
    assert result.metadata["director_ai"]["approved"] is True
    assert result.metadata["director_ai"]["tenant_id"] == "tenant-a"
    assert result.metadata["director_ai"]["score"] >= 0.4
    assert result.value_override == "The support SLA is four hours."


def test_guardrails_protocol_failure_is_tenant_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejected Guardrails results should report score metadata without raw text."""
    _install_guardrails_protocol_package(tmp_path, monkeypatch)

    from guardrails import Guard
    from guardrails.validators import REGISTERED

    from director_ai.integrations.guardrails_ai import build_guardrails_validator_class

    validator_class = build_guardrails_validator_class()
    validator = validator_class(
        facts={"sla": "Support SLA is four hours."},
        threshold=0.4,
        use_nli=False,
    )
    guard = Guard().use(validator)

    result = guard.parse(
        "The SLA is one minute and includes free refunds.",
        metadata={
            "query": "What is the support SLA?",
            "tenant": "tenant-a",
        },
    )

    assert REGISTERED["director-ai-coherence"]["data_type"] == "string"
    assert result.outcome == "fail"
    assert "one minute" not in result.error_message
    assert result.metadata["director_ai"]["approved"] is False
    assert result.metadata["director_ai"]["tenant_id"] == "tenant-a"
    assert result.metadata["director_ai"]["score"] < 0.4
