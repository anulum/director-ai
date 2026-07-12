# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NeMo Guardrails rails-as-config loader

"""Load NeMo Guardrails rails configuration into a Director ``Policy``.

Teams migrating from NeMo Guardrails keep their rails as config — a
``config.yml`` plus Colang ``.co`` flow files. This loader maps the
**honest subset** of those files onto Director's native declarative
:class:`~director_ai.core.safety.policy.Policy`:

* Colang v1 **topical rails** — a ``define flow`` whose statements pair
  ``user <intent>`` with a ``bot refuse …`` response marks every example
  utterance of that intent as a forbidden phrase (word-boundary,
  case-insensitive — the Policy engine's semantics).
* ``config.yml`` **self-check / content-safety rails** — the presence of
  recognised input/output flow names enables Director's dependency-free
  moderation detectors (keyword toxicity + regex PII) as the semantic
  equivalent; the substitution is recorded in ``notes``.

Everything else — bot message definitions, subflows, ``execute``
actions, variables, conditionals, model/prompt configuration — is
**reported, not silently dropped**: each unmapped construct lands in
``RailsLoadResult.unsupported`` so an operator sees exactly what did not
carry over. Guardrails AI RAIL XML is intentionally out of scope: that
ecosystem is integrated natively as a validator
(:mod:`director_ai.integrations.guardrails_ai`), so ``.rail``/``.xml``
inputs raise with that pointer instead of a half-faithful translation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from director_ai.core.safety.policy import Policy

_RECOGNISED_MODERATION_FLOWS = (
    "self check input",
    "self check output",
    "content safety check",
)
_QUOTED = re.compile(r'"([^"]+)"')
_MODERATION_NOTE = (
    "NeMo self-check/content-safety rails are mapped to Director's "
    "dependency-free moderation detectors (keyword toxicity + regex "
    "PII); the semantics are equivalent in intent, not a re-implementation "
    "of the NeMo prompt-based checks."
)


@dataclass(frozen=True)
class RailsLoadResult:
    """Outcome of loading a rails configuration.

    Parameters
    ----------
    policy:
        The Director policy assembled from every construct that mapped.
    source_format:
        ``"nemo-config"`` (directory or config.yml), ``"colang"`` (a
        single ``.co`` file), or ``"nemo-config+colang"`` for a config
        directory carrying both.
    forbidden_from_intents:
        Mapping of refusal-flow intent name to the example utterances
        that became forbidden phrases.
    moderation_enabled:
        Whether recognised self-check rails enabled the moderation
        detectors.
    unsupported:
        Every construct that was seen but not mapped, verbatim enough
        for an operator to audit the gap.
    notes:
        Human-readable notes about semantic substitutions.
    """

    policy: Policy
    source_format: str
    forbidden_from_intents: dict[str, tuple[str, ...]]
    moderation_enabled: bool
    unsupported: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible summary (without the Policy object)."""
        return {
            "source_format": self.source_format,
            "forbidden_phrases": len(self.policy.forbidden),
            "forbidden_from_intents": {
                intent: list(phrases)
                for intent, phrases in self.forbidden_from_intents.items()
            },
            "moderation_enabled": self.moderation_enabled,
            "unsupported": list(self.unsupported),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class _ColangScan:
    """Intermediate result of scanning one Colang source."""

    intent_phrases: dict[str, tuple[str, ...]]
    refusal_intents: tuple[str, ...]
    unsupported: tuple[str, ...]


def load_rails_config(path: str | Path) -> RailsLoadResult:
    """Load a NeMo Guardrails config directory, ``config.yml``, or ``.co`` file.

    Parameters
    ----------
    path:
        A NeMo config **directory** (``config.yml`` + ``*.co``), a single
        YAML config file, or a single Colang ``.co`` file.

    Returns
    -------
    RailsLoadResult
        The assembled policy plus an explicit audit of what did and did
        not map.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        For RAIL XML inputs (natively integrated elsewhere) and other
        unsupported file types.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"rails configuration not found: {root}")

    if root.is_dir():
        config_file = root / "config.yml"
        if not config_file.is_file():
            config_file = root / "config.yaml"
        colang_files = sorted(root.glob("*.co"))
        if not config_file.is_file() and not colang_files:
            raise ValueError(
                f"no config.yml or *.co files found under {root}",
            )
        return _assemble(
            config_data=_read_yaml(config_file) if config_file.is_file() else None,
            colang_sources=[f.read_text(encoding="utf-8") for f in colang_files],
        )

    suffix = root.suffix.lower()
    if suffix in (".rail", ".xml"):
        raise ValueError(
            "RAIL XML is not translated: Director integrates Guardrails AI "
            "natively as a validator — see "
            "director_ai.integrations.guardrails_ai.",
        )
    if suffix == ".co":
        return _assemble(
            config_data=None,
            colang_sources=[root.read_text(encoding="utf-8")],
        )
    if suffix in (".yml", ".yaml"):
        return _assemble(config_data=_read_yaml(root), colang_sources=[])
    raise ValueError(f"unsupported rails configuration file type: {root.name}")


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML config, falling back to JSON when PyYAML is absent."""
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(raw)
    except ImportError:  # pragma: no cover - PyYAML is a core dependency
        import json

        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"rails config must be a mapping: {path}")
    return data


def _scan_colang(source: str) -> _ColangScan:
    """Scan one Colang v1 source for the topical-rails subset."""
    intent_phrases: dict[str, list[str]] = {}
    refusal_intents: list[str] = []
    unsupported: list[str] = []

    current_kind = ""
    current_name = ""
    flow_statements: list[str] = []

    def _close_flow() -> None:
        if current_kind != "flow":
            return
        flow_intents = [
            stmt.removeprefix("user ").strip()
            for stmt in flow_statements
            if stmt.startswith("user ")
        ]
        refuses = any(stmt.startswith("bot refuse") for stmt in flow_statements)
        for stmt in flow_statements:
            if stmt.startswith("user ") or stmt.startswith("bot refuse"):
                continue
            unsupported.append(f"flow {current_name}: {stmt}")
        if refuses:
            refusal_intents.extend(flow_intents)
        elif flow_intents:
            unsupported.append(
                f"flow {current_name}: no 'bot refuse' response — intents "
                f"{', '.join(flow_intents)} not mapped",
            )

    for raw_line in source.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            _close_flow()
            current_kind = ""
            current_name = ""
            flow_statements = []
            tokens = line.split()
            if tokens[0] != "define" or len(tokens) < 3:
                unsupported.append(line.strip())
                continue
            kind = tokens[1]
            name = " ".join(tokens[2:])
            if kind == "user":
                current_kind, current_name = "user", name
                intent_phrases.setdefault(name, [])
            elif kind == "flow":
                current_kind, current_name = "flow", name
            else:
                current_kind = "other"
                unsupported.append(f"define {kind} {name}")
            continue

        statement = line.strip()
        if current_kind == "user":
            quoted = _QUOTED.findall(statement)
            if quoted:
                intent_phrases[current_name].extend(quoted)
            else:
                unsupported.append(f"user {current_name}: {statement}")
        elif current_kind == "flow":
            flow_statements.append(statement)
        elif current_kind == "other":
            continue
        else:
            unsupported.append(statement)
    _close_flow()

    return _ColangScan(
        intent_phrases={
            name: tuple(phrases) for name, phrases in intent_phrases.items()
        },
        refusal_intents=tuple(dict.fromkeys(refusal_intents)),
        unsupported=tuple(unsupported),
    )


def _scan_nemo_config(
    data: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Scan a NeMo ``config.yml`` mapping for the recognised rails subset."""
    unsupported: list[str] = []
    moderation = False

    for key in data:
        if key != "rails":
            unsupported.append(f"config key: {key}")

    rails = data.get("rails")
    if not isinstance(rails, dict):
        return moderation, unsupported

    for direction, block in rails.items():
        if direction not in ("input", "output"):
            unsupported.append(f"rails.{direction}")
            continue
        flows = block.get("flows", []) if isinstance(block, dict) else []
        if not isinstance(flows, list):
            unsupported.append(f"rails.{direction}.flows: not a list")
            continue
        for flow in flows:
            flow_name = str(flow).strip().lower()
            if flow_name.startswith(_RECOGNISED_MODERATION_FLOWS):
                moderation = True
            else:
                unsupported.append(f"rails.{direction}.flows: {flow}")
    return moderation, unsupported


def _assemble(
    *,
    config_data: dict[str, Any] | None,
    colang_sources: list[str],
) -> RailsLoadResult:
    """Merge config and Colang scans into one policy and audit record."""
    unsupported: list[str] = []
    notes: list[str] = []
    moderation = False

    if config_data is not None:
        moderation, config_unsupported = _scan_nemo_config(config_data)
        unsupported.extend(config_unsupported)
        if moderation:
            notes.append(_MODERATION_NOTE)

    forbidden_from_intents: dict[str, tuple[str, ...]] = {}
    for source in colang_sources:
        scan = _scan_colang(source)
        unsupported.extend(scan.unsupported)
        for intent in scan.refusal_intents:
            phrases = scan.intent_phrases.get(intent, ())
            if phrases:
                forbidden_from_intents[intent] = phrases
            else:
                unsupported.append(
                    f"refusal intent {intent}: no example utterances defined",
                )

    forbidden: list[str] = []
    for phrases in forbidden_from_intents.values():
        forbidden.extend(phrases)

    policy = Policy(forbidden=forbidden)
    if moderation:
        from director_ai.core.safety.moderation import (
            KeywordToxicityDetector,
            RegexPIIDetector,
        )

        policy = policy.with_moderation(
            [KeywordToxicityDetector(), RegexPIIDetector()],
        )

    if config_data is not None and colang_sources:
        source_format = "nemo-config+colang"
    elif config_data is not None:
        source_format = "nemo-config"
    else:
        source_format = "colang"

    return RailsLoadResult(
        policy=policy,
        source_format=source_format,
        forbidden_from_intents=forbidden_from_intents,
        moderation_enabled=moderation,
        unsupported=tuple(dict.fromkeys(unsupported)),
        notes=tuple(notes),
    )
