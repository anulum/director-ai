# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Guardrails AI Integration
"""Guardrails AI custom validator adapter.

The adapter is dependency-light: importing this module does not import
``guardrails``. Guardrails AI is imported only when a validator class or
instance is built.
"""

from __future__ import annotations

from typing import Any

from director_ai.core import CoherenceScorer, GroundTruthStore

DEFAULT_VALIDATOR_NAME = "director-ai-coherence"


def build_guardrails_validator_class(
    *,
    name: str = DEFAULT_VALIDATOR_NAME,
    data_type: str = "string",
) -> type:
    """Return and register a Guardrails AI validator class.

    The returned class follows Guardrails AI's custom-validator contract:
    subclass ``Validator``, implement ``_validate(value, metadata)``, and return
    ``PassResult`` or ``FailResult``.
    """

    guardrails = _load_guardrails_validator_api()

    validator_base = guardrails.validator

    @guardrails.register_validator(name=name, data_type=data_type)
    class DirectorAICoherenceValidator(validator_base):  # type: ignore[misc, valid-type]
        """Guardrails AI validator backed by Director-AI coherence scoring."""

        def __init__(
            self,
            *,
            facts: dict[str, str] | None = None,
            store: GroundTruthStore | None = None,
            threshold: float = 0.5,
            use_nli: bool | None = None,
            on_fail: Any = None,
        ) -> None:
            super().__init__(
                on_fail=on_fail,
                threshold=threshold,
                use_nli=use_nli,
            )
            self._store = _build_store(facts=facts, store=store)
            self._scorer = CoherenceScorer(
                threshold=threshold,
                ground_truth_store=self._store,
                use_nli=use_nli,
            )

        def _validate(self, value: Any, metadata: dict[str, Any]) -> Any:
            response = _coerce_text(value)
            prompt = _prompt_from_metadata(metadata) or response
            tenant_id = _tenant_from_metadata(metadata)
            approved, score = self._scorer.review(
                prompt,
                response,
                tenant_id=tenant_id,
            )
            result_metadata = {
                "director_ai": {
                    "approved": approved,
                    "score": score.score,
                    "h_logical": score.h_logical,
                    "h_factual": score.h_factual,
                    "tenant_id": tenant_id,
                }
            }
            if approved:
                return _result(
                    guardrails.pass_result,
                    metadata=result_metadata,
                    value_override=value,
                )
            return _result(
                guardrails.fail_result,
                error_message=(
                    "Director-AI coherence validation failed "
                    f"(score={score.score:.3f})."
                ),
                metadata=result_metadata,
            )

        @property
        def scorer(self) -> CoherenceScorer:
            return self._scorer

    return DirectorAICoherenceValidator


def build_guardrails_validator(
    *,
    facts: dict[str, str] | None = None,
    store: GroundTruthStore | None = None,
    threshold: float = 0.5,
    use_nli: bool | None = None,
    on_fail: Any = None,
    name: str = DEFAULT_VALIDATOR_NAME,
    data_type: str = "string",
) -> Any:
    """Build a registered Guardrails AI validator instance."""

    validator_class = build_guardrails_validator_class(name=name, data_type=data_type)
    return validator_class(
        facts=facts,
        store=store,
        threshold=threshold,
        use_nli=use_nli,
        on_fail=on_fail,
    )


def attach_guardrails_validator(guard: Any, **kwargs: Any) -> Any:
    """Attach the Director-AI validator to ``Guard().use(...)`` style guards."""

    validator = build_guardrails_validator(**kwargs)
    use = getattr(guard, "use", None)
    if use is None:
        raise TypeError("Guardrails guard object must expose a use() method")
    return use(validator)


def _load_guardrails_validator_api() -> Any:
    try:
        from guardrails.validators import (  # type: ignore[import-not-found]
            FailResult,
            PassResult,
            Validator,
            register_validator,
        )
    except ImportError as exc:
        raise ImportError(
            "Guardrails AI integration requires optional dependency "
            "'guardrails-ai'. Install it alongside director-ai in an "
            "environment where that package resolves."
        ) from exc
    return _GuardrailsValidatorAPI(
        fail_result=FailResult,
        pass_result=PassResult,
        validator=Validator,
        register_validator=register_validator,
    )


class _GuardrailsValidatorAPI:
    def __init__(
        self,
        *,
        fail_result: type,
        pass_result: type,
        validator: type,
        register_validator: Any,
    ) -> None:
        self.fail_result = fail_result
        self.pass_result = pass_result
        self.validator = validator
        self.register_validator = register_validator


def _build_store(
    *,
    facts: dict[str, str] | None,
    store: GroundTruthStore | None,
) -> GroundTruthStore:
    if store is not None:
        return store
    gt = GroundTruthStore()
    for key, fact in (facts or {}).items():
        gt.add(key, fact)
    return gt


def _result(result_type: type, **kwargs: Any) -> Any:
    try:
        return result_type(**kwargs)
    except TypeError:
        minimal = {
            key: value
            for key, value in kwargs.items()
            if key in {"error_message", "fix_value", "on_fix"}
        }
        return result_type(**minimal)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _prompt_from_metadata(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    for key in ("prompt", "query", "question", "input"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = metadata.get("messages")
    if isinstance(messages, list):
        return _prompt_from_messages(messages)
    return ""


def _prompt_from_messages(messages: list[Any]) -> str:
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(part.get("text", "")).strip()
                for part in content
                if isinstance(part, dict) and str(part.get("text", "")).strip()
            ]
            if parts:
                return " ".join(parts)
    return ""


def _tenant_from_metadata(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    for key in ("tenant_id", "tenant"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


__all__ = [
    "DEFAULT_VALIDATOR_NAME",
    "attach_guardrails_validator",
    "build_guardrails_validator",
    "build_guardrails_validator_class",
]
