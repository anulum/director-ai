# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Config field grouping for UI wizard
"""Categorise ``DirectorConfig`` fields into semantic groups for the
configuration wizard UI.

Each field is assigned to a group with display metadata (label,
description, widget type). The wizard introspects this at runtime
to render appropriate controls.

Usage::

    from director_ai.ui._field_groups import get_field_groups

    groups = get_field_groups()
    for group_name, fields in groups.items():
        print(f"--- {group_name} ---")
        for f in fields:
            print(f"  {f['name']}: {f['widget']} ({f['description']})")
"""

from __future__ import annotations

import dataclasses
from dataclasses import fields as dc_fields

__all__ = ["get_field_groups", "FieldMeta", "FIELD_GROUPS"]


# Group definitions: group_name → list of field name prefixes
_GROUP_PREFIXES: dict[str, list[str]] = {
    "Scoring": [
        "coherence_threshold",
        "hard_limit",
        "soft_limit",
        "scorer_backend",
        "w_logic",
        "w_fact",
        "strict_mode",
        "use_nli",
    ],
    "NLI Model": [
        "nli_",
        "onnx_",
        "lora_",
    ],
    "Retrieval": [
        "vector_backend",
        "embedding_model",
        "chroma_",
        "hybrid_retrieval",
        "reranker_",
        "retrieval_abstention",
        "parent_child_",
        "adaptive_retrieval_",
        "hyde_",
        "query_decomposition_",
        "contextual_compression_",
        "multi_vector_",
    ],
    "LLM Provider": [
        "llm_",
        "privacy_mode",
    ],
    "Injection Detection": [
        "injection_",
    ],
    "Thresholds (per task)": [
        "threshold_",
        "adaptive_threshold",
    ],
    "Server": [
        "server_",
        "cors_",
        "rate_limit_",
    ],
    "Enterprise": [
        "redis_",
        "cache_",
        "audit_",
        "tenant_",
    ],
    "Security": [
        "sanitize_",
        "sanitizer_",
        "redact_",
        "api_key",
        "metrics_require_auth",
    ],
    "Observability": [
        "metrics_",
        "log_",
        "otel_",
    ],
    "Compliance": [
        "compliance_",
    ],
}

# Widget type inference by Python type
_TYPE_WIDGETS: dict[type, str] = {
    bool: "toggle",
    int: "number",
    float: "slider",
    str: "text",
}


class FieldMeta:
    """Metadata for a single config field."""

    __slots__ = ("name", "group", "field_type", "default", "widget", "description")

    def __init__(
        self,
        name: str,
        group: str,
        field_type: type,
        default,
        widget: str,
        description: str = "",
    ) -> None:
        self.name = name
        self.group = group
        self.field_type = field_type
        self.default = default
        self.widget = widget
        self.description = description

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "group": self.group,
            "type": self.field_type.__name__,
            "default": self.default,
            "widget": self.widget,
            "description": self.description,
        }


def _classify_field(name: str) -> str:
    """Assign a field to its group based on prefix matching."""
    for group, prefixes in _GROUP_PREFIXES.items():
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix):
                return group
    return "Other"


def _infer_widget(field_type: type) -> str:
    """Infer UI widget type from Python type."""
    return _TYPE_WIDGETS.get(field_type, "text")


def get_field_groups() -> dict[str, list[dict]]:
    """Introspect ``DirectorConfig`` and group fields for UI rendering.

    Returns a dict mapping group name → list of field metadata dicts.
    Each dict has keys: name, group, type, default, widget, description.
    """
    from director_ai.core.config import DirectorConfig

    groups: dict[str, list[dict]] = {}

    for f in dc_fields(DirectorConfig):
        group = _classify_field(f.name)
        widget = _infer_widget(f.type if isinstance(f.type, type) else str)

        meta = FieldMeta(
            name=f.name,
            group=group,
            field_type=f.type if isinstance(f.type, type) else str,
            default=(
                f.default if f.default is not dataclasses.MISSING else None
            ),
            widget=widget,
            description="",
        )

        groups.setdefault(group, []).append(meta.to_dict())

    return groups


# Pre-computed group names for iteration order
FIELD_GROUPS = list(_GROUP_PREFIXES.keys()) + ["Other"]
