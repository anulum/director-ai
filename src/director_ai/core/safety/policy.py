# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” YAML Policy Engine

"""Declarative policy enforcement for LLM output.

Load rules from a YAML file (or dict) and check responses for
forbidden phrases, required citations, length limits, and regex patterns.

Usage::

    policy = Policy.from_yaml("policy.yaml")
    violations = policy.check("The sky is blue.")
    if violations:
        print("BLOCKED:", violations)

YAML format::

    forbidden:
      - "ignore previous instructions"
      - "as an AI language model"
    required_citations:
      min_count: 1
      pattern: "\\[\\d+\\]"
    style:
      max_length: 2000
    patterns:
      - name: no_profanity
        regex: "\\bdamn\\b|\\bhell\\b"
        action: block
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Violation:
    """Single policy violation."""

    rule: str
    detail: str


@dataclass
class Policy:
    """Declarative output policy with YAML/dict loading.

    Parameters
    ----------
    forbidden : list[str] â€” phrases that trigger immediate block.
    patterns : list[dict] â€” regex rules with name/regex/action keys.
    max_length : int â€” max response character count (0 = unlimited).
    required_citations_pattern : str â€” regex for citation markers.
    required_citations_min : int â€” minimum citation count (0 = disabled).

    """

    forbidden: list[str] = field(default_factory=list)
    patterns: list[dict] = field(default_factory=list)
    max_length: int = 0
    required_citations_pattern: str = ""
    required_citations_min: int = 0

    _compiled_forbidden: list[re.Pattern] = field(
        default_factory=list,
        repr=False,
    )
    _compiled_patterns: list[tuple[str, re.Pattern, str]] = field(
        default_factory=list,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._compiled_forbidden = [
            re.compile(re.escape(phrase), re.IGNORECASE) for phrase in self.forbidden
        ]
        self._compiled_patterns = []
        for p in self.patterns:
            name = p.get("name", "unnamed")
            regex = p.get("regex", "")
            action = p.get("action", "block")
            if regex:
                try:
                    compiled = re.compile(regex, re.IGNORECASE)
                except re.error as e:
                    raise ValueError(
                        f"Invalid regex in policy pattern '{name}': {e}",
                    ) from e
                self._compiled_patterns.append((name, compiled, action))

    @classmethod
    def from_dict(cls, data: dict) -> Policy:
        """Build a Policy from a plain dict (parsed YAML)."""
        forbidden = data.get("forbidden", [])
        patterns = data.get("patterns", [])
        style = data.get("style", {})
        cit = data.get("required_citations", {})
        return cls(
            forbidden=forbidden,
            patterns=patterns,
            max_length=style.get("max_length", 0),
            required_citations_pattern=cit.get("pattern", ""),
            required_citations_min=cit.get("min_count", 0),
        )

    @classmethod
    def from_yaml(cls, path: str) -> Policy:
        """Load policy from a YAML file (falls back to JSON)."""
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        try:
            import yaml

            data = yaml.safe_load(raw)
        except ImportError:
            import json

            data = json.loads(raw)
        if not isinstance(data, dict):
            return cls()
        return cls.from_dict(data)

    def check(self, text: str) -> list[Violation]:
        """Check text against all policy rules. Returns violations."""
        violations: list[Violation] = []

        for i, pat in enumerate(self._compiled_forbidden):
            if pat.search(text):
                violations.append(
                    Violation(
                        rule="forbidden",
                        detail=self.forbidden[i],
                    ),
                )

        if self.max_length > 0 and len(text) > self.max_length:
            violations.append(
                Violation(
                    rule="max_length",
                    detail=f"{len(text)} > {self.max_length}",
                ),
            )

        if self.required_citations_min > 0 and self.required_citations_pattern:
            matches = re.findall(self.required_citations_pattern, text)
            if len(matches) < self.required_citations_min:
                violations.append(
                    Violation(
                        rule="required_citations",
                        detail=(
                            f"found {len(matches)}, need {self.required_citations_min}"
                        ),
                    ),
                )

        for name, pat, action in self._compiled_patterns:
            if pat.search(text):
                violations.append(
                    Violation(
                        rule=f"pattern:{name}",
                        detail=action,
                    ),
                )

        return violations
