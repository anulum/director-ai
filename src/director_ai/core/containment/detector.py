# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — sandbox breakout detector

"""Pattern matchers for the three practical breakout modes.

* **Production target in sandbox context** — the agent emits an
  outbound URL or hostname that belongs to a real system
  (``api.openai.com``, an internal billing gateway) while the
  anchor says ``sandbox`` or ``simulator``.
* **Anti-anchor prompt injection** — free-text fields in the
  proposed action contain phrases aimed at getting the model to
  ignore the scope (``ignore sandbox``, ``you are in production``,
  ``disregard containment``).
* **Claimed vs. anchored scope mismatch** — the agent asserts a
  scope in the event payload that disagrees with the verified
  anchor.

Each matcher returns a :class:`BreakoutFinding` with enough detail
for downstream audit logs; the aggregator returns the list without
short-circuiting so the operator sees every concurrent failure
mode at once.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from .scope import ContainmentScope

# Hosts that must never be reachable from a sandboxed agent. The
# wildcard at the front of a pattern (``*.openai.com``) is expanded
# to a regex at load time to avoid repeated recompilation.
_DEFAULT_PRODUCTION_HOSTS: frozenset[str] = frozenset(
    {
        "api.openai.com",
        "api.anthropic.com",
        "api.stripe.com",
        "api.github.com",
        "kubernetes.default.svc",
    },
)

# Phrases the model emits when prompt-injected to cross the
# boundary. Lowercased for cheap substring matching; runs on raw
# action text, not semantically parsed fields, so false positives
# are tolerable (the worst outcome is a human-reviewed action).
_DEFAULT_ANTI_ANCHOR_PHRASES: tuple[str, ...] = (
    "ignore sandbox",
    "ignore the sandbox",
    "disregard containment",
    "you are in production",
    "switch to production",
    "bypass reality anchor",
    "pretend this is real",
    "act as if live",
)


@dataclass(frozen=True)
class BreakoutFinding:
    """A single suspicious signal.

    ``category`` is one of ``"production_target"``,
    ``"anti_anchor_injection"``, ``"scope_mismatch"``, ``"policy"``.
    ``severity`` is ``"high"`` for findings that warrant an
    immediate block, ``"medium"`` for a warn+review, ``"low"``
    for log-only.
    """

    category: str
    severity: str
    detail: str


@dataclass
class BreakoutDetector:
    """Rule-based scanner for the breakout signals above.

    Parameters
    ----------
    production_hosts : set[str] | None
        Hostnames whose presence in a sandboxed action is a high-
        severity finding. Defaults to the built-in list; operators
        extend it with internal billing and infra hostnames.
    anti_anchor_phrases : tuple[str, ...] | None
        Case-insensitive substrings that indicate a scope-override
        prompt injection.
    max_text_length : int
        Defensive cap on the scanned free-text to keep the detector
        O(n·m) bounded even when the model emits a pathological
        payload. 0 disables.
    """

    production_hosts: set[str] = field(
        default_factory=lambda: set(_DEFAULT_PRODUCTION_HOSTS),
    )
    anti_anchor_phrases: tuple[str, ...] = _DEFAULT_ANTI_ANCHOR_PHRASES
    max_text_length: int = 64_000

    def __post_init__(self) -> None:
        if self.max_text_length < 0:
            raise ValueError("max_text_length must be non-negative")
        if any(not p for p in self.anti_anchor_phrases):
            raise ValueError("anti_anchor_phrases must not contain empty entries")
        # Precompile URL/hostname regex once; we scan every event.
        self._host_pattern: re.Pattern[str] = re.compile(
            r"(?i)(?:https?://)?([a-z0-9][a-z0-9.\-]*[a-z0-9])",
        )

    def scan(
        self,
        event: Mapping[str, object],
        anchored_scope: ContainmentScope,
        claimed_scope: ContainmentScope | None = None,
    ) -> list[BreakoutFinding]:
        """Return every finding the event triggers.

        ``event`` is a free-form dict: the detector looks at the
        ``url``, ``hostname`` and ``text`` fields directly and
        folds the string representation of everything else through
        the anti-anchor phrase check.
        """
        findings: list[BreakoutFinding] = []

        if anchored_scope != "production":
            findings.extend(self._scan_production_targets(event, anchored_scope))

        findings.extend(self._scan_anti_anchor(event))

        if claimed_scope is not None and claimed_scope != anchored_scope:
            findings.append(
                BreakoutFinding(
                    category="scope_mismatch",
                    severity="high",
                    detail=(
                        f"event claims scope={claimed_scope} but anchor "
                        f"attests scope={anchored_scope}"
                    ),
                ),
            )

        return findings

    def _scan_production_targets(
        self,
        event: Mapping[str, object],
        anchored_scope: ContainmentScope,
    ) -> list[BreakoutFinding]:
        findings: list[BreakoutFinding] = []
        hosts_in_event: set[str] = set()

        direct_host = event.get("hostname")
        if isinstance(direct_host, str) and direct_host:
            hosts_in_event.add(direct_host.lower())

        url = event.get("url")
        if isinstance(url, str):
            for match in self._host_pattern.finditer(url):
                hosts_in_event.add(match.group(1).lower())

        for host in hosts_in_event & self.production_hosts:
            findings.append(
                BreakoutFinding(
                    category="production_target",
                    severity="high",
                    detail=(
                        f"attempt to reach production host {host!r} from "
                        f"scope={anchored_scope}"
                    ),
                ),
            )
        return findings

    def _scan_anti_anchor(self, event: Mapping[str, object]) -> list[BreakoutFinding]:
        haystack = self._gather_text(event)
        if not haystack:
            return []
        lowered = haystack.lower()
        findings: list[BreakoutFinding] = []
        for phrase in self.anti_anchor_phrases:
            if phrase in lowered:
                findings.append(
                    BreakoutFinding(
                        category="anti_anchor_injection",
                        severity="high",
                        detail=f"detected phrase {phrase!r} in event text",
                    ),
                )
        return findings

    def _gather_text(self, event: Mapping[str, object]) -> str:
        """Concatenate every string-valued field for phrase scanning.

        Nested dicts/lists are stringified one level deep — we do
        not recurse indefinitely because the attacker can't hide
        effective prompt injection inside deep structure anyway.
        """
        parts: list[str] = []
        for value in event.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (list, tuple, dict)):
                parts.append(str(value))
        joined = "\n".join(parts)
        if self.max_text_length and len(joined) > self.max_text_length:
            joined = joined[: self.max_text_length]
        return joined
