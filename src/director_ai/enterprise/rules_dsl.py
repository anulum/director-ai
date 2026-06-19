# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Enterprise Custom Rules DSL

"""Strict JSON/YAML ruleset loader for enterprise policy operators."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from director_ai.core.policy_compiler import CompiledRule, PolicyBundle
from director_ai.core.policy_compiler.rule import RuleAction, RuleKind
from director_ai.core.safety.policy import Policy

RulesDslVersion = Literal["director.rules.v1"]
RuleValue = str | int


class RulesDslError(ValueError):
    """Raised when a custom rules DSL document is invalid."""


class CustomRule(BaseModel):
    """One validated custom policy rule."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9_.:-]+$")
    kind: RuleKind
    value: RuleValue
    name: str | None = Field(default=None, min_length=1, max_length=96)
    action: RuleAction | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def _validate_rule_shape(self) -> Self:
        if self.kind in {"forbidden", "pattern"}:
            value = self.value
            if not isinstance(value, str) or not value:
                raise ValueError(f"{self.kind} value must be a non-empty string")
            if len(value) > 4096:
                raise ValueError(f"{self.kind} value must be at most 4096 characters")
            if self.kind == "pattern":
                try:
                    re.compile(value)
                except re.error as exc:
                    raise ValueError(
                        f"invalid regex for rule {self.id!r}: {exc}"
                    ) from exc
        if self.kind in {"max_length", "required_citations"} and (
            not isinstance(self.value, int) or self.value <= 0
        ):
            raise ValueError(f"{self.kind} value must be a positive integer")
        return self

    @property
    def compiled_name(self) -> str:
        """Return the runtime rule name shown in policy violations."""
        return self.name or self.id

    @property
    def compiled_action(self) -> RuleAction:
        """Return the runtime action, defaulting to the compiler contract."""
        return self.action or "block"

    def to_compiled_rule(self) -> CompiledRule:
        """Convert into the existing policy compiler rule contract."""
        return CompiledRule(
            id=self.id,
            kind=self.kind,
            value=str(self.value),
            name=self.compiled_name,
            action=self.compiled_action,
            threshold=self.threshold,
            source=self.source or "",
        )


class CustomRuleset(BaseModel):
    """Versioned JSON/YAML ruleset with strict schema validation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    version: RulesDslVersion
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    rules: tuple[CustomRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_duplicate_rule_ids(self) -> Self:
        seen: set[str] = set()
        duplicates: list[str] = []
        for rule in self.rules:
            if rule.id in seen:
                duplicates.append(rule.id)
            seen.add(rule.id)
        if duplicates:
            duplicate_list = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"duplicate rule id: {duplicate_list}")
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CustomRuleset:
        """Load a ruleset from a parsed JSON/YAML mapping."""
        try:
            ruleset: CustomRuleset = cls.model_validate(data)
        except ValidationError as exc:
            raise RulesDslError(str(exc)) from exc
        return ruleset

    @classmethod
    def from_json(cls, text: str) -> CustomRuleset:
        """Load a ruleset from JSON text."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RulesDslError(f"invalid JSON ruleset: {exc}") from exc
        if not isinstance(data, dict):
            raise RulesDslError("ruleset root must be a JSON object")
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, text: str) -> CustomRuleset:
        """Load a ruleset from YAML text."""
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency is locked
            raise RulesDslError("PyYAML is required to load YAML rulesets") from exc
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise RulesDslError("ruleset root must be a YAML mapping")
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> CustomRuleset:
        """Load a ruleset from a JSON, YAML, or YML file."""
        ruleset_path = Path(path)
        text = ruleset_path.read_text(encoding="utf-8")
        if ruleset_path.suffix.lower() == ".json":
            return cls.from_json(text)
        return cls.from_yaml(text)

    def to_policy_bundle(self, *, version: int = 1) -> PolicyBundle:
        """Convert into an immutable policy compiler bundle."""
        if version <= 0:
            raise RulesDslError("policy bundle version must be a positive integer")
        return PolicyBundle(
            version=version,
            rules=tuple(rule.to_compiled_rule() for rule in self.rules),
        )

    def to_policy(self) -> Policy:
        """Convert into the runtime safety policy."""
        return self.to_policy_bundle().to_policy()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the ruleset into deterministic JSON-compatible data."""
        serialised: dict[str, Any] = self.model_dump(mode="json", exclude_none=True)
        return serialised

    def to_json(self) -> str:
        """Serialise the ruleset into deterministic JSON text."""
        return json.dumps(self.to_dict(), sort_keys=True)
