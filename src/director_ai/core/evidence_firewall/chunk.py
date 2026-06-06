# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Evidence firewall chunk view

"""A normalised, read-only view over a retrieved chunk.

Backends and ingestion paths write provenance metadata under several historical
key spellings (``kb_content_hash`` vs ``content_sha256``; ``expires_at`` vs
``expires_unix`` vs ``expires``). :class:`RetrievedChunk` resolves those aliases
once so the admission checks read a single, typed surface instead of poking the
raw metadata dict. It never mutates the source mapping.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

__all__ = ["RetrievedChunk"]

# Metadata key aliases, most-specific first. The first present, non-empty value
# wins so newer signed-write keys take precedence over legacy ingestion keys.
_TENANT_KEYS = ("tenant_id", "kb_tenant_id")
_TEXT_DIGEST_KEYS = ("text_sha256", "content_sha256")
_PROVENANCE_DIGEST_KEYS = (
    "text_sha256",
    "content_sha256",
    "value_sha256",
    "kb_content_hash",
)
_EXPIRY_KEYS = ("expires_unix", "expires_at", "expires")
_CREATED_KEYS = ("created_at", "kb_source_timestamp", "source_timestamp")
_SENSITIVITY_KEYS = ("sensitivity", "kb_sensitivity", "sensitivity_label")
_SOURCE_OWNER_KEYS = (
    "source_owner",
    "kb_source_key",
    "source_id",
    "source",
    "external_id",
)
_USE_CASE_KEYS = ("allowed_use_cases", "kb_allowed_use_cases", "allowed_use")
_SIGNATURE_KEY = "kb_signature_verified"
_VERSION_KEY = "kb_version"


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk, with provenance metadata resolved to typed fields.

    Parameters
    ----------
    chunk_id:
        The stable identifier of the chunk (the vector store ``id``).
    text:
        The chunk text that would be handed to the model.
    metadata:
        The raw metadata mapping as stored by the retrieval backend. Copied
        defensively; the original is never mutated.
    """

    chunk_id: str
    text: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_query_result(cls, result: Mapping[str, Any]) -> RetrievedChunk:
        """Build a chunk from a vector-store query result row.

        Accepts the ``{"id", "text", "metadata"}`` shape produced by
        :class:`~director_ai.core.retrieval.vector_store.base.VectorBackend`
        implementations.
        """
        return cls(
            chunk_id=str(result.get("id", "")),
            text=str(result.get("text", "")),
            metadata=result.get("metadata") or {},
        )

    # ── tenant / ownership ────────────────────────────────────────────────

    @property
    def tenant_id(self) -> str:
        """Owning tenant; empty string for a shared, non-tenant corpus."""
        return self._first_str(_TENANT_KEYS)

    @property
    def source_owner(self) -> str:
        """Recorded source owner/key; empty when unknown."""
        return self._first_str(_SOURCE_OWNER_KEYS)

    # ── provenance / integrity ────────────────────────────────────────────

    @property
    def signature_verified(self) -> bool:
        """Whether the chunk's write was signature-verified at ingest time."""
        return self.metadata.get(_SIGNATURE_KEY) is True

    @property
    def has_provenance(self) -> bool:
        """Whether any provenance marker is present (digest, version, or
        verified signature)."""
        if self.signature_verified:
            return True
        if str(self.metadata.get(_VERSION_KEY, "")).strip():
            return True
        return bool(self._first_str(_PROVENANCE_DIGEST_KEYS))

    @property
    def recorded_text_digest(self) -> str:
        """Recorded SHA-256 hex digest *of the chunk text*, if any.

        Only digests known to hash the literal text/content are returned; the
        value-only ``kb_content_hash`` is excluded because it hashes the fact
        value under a different convention and would not match a text recompute.
        """
        return self._first_str(_TEXT_DIGEST_KEYS)

    def computed_text_digest(self) -> str:
        """Return the SHA-256 hex digest of the current chunk text."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    # ── lifecycle ─────────────────────────────────────────────────────────

    @property
    def expires_at_unix(self) -> float | None:
        """Expiry as epoch seconds, or ``None`` when no expiry is recorded."""
        for key in _EXPIRY_KEYS:
            if key in self.metadata:
                parsed = _to_unix(self.metadata[key])
                if parsed is not None:
                    return parsed
        return None

    @property
    def created_at_unix(self) -> float | None:
        """Creation time as epoch seconds, or ``None`` when not recorded."""
        for key in _CREATED_KEYS:
            if key in self.metadata:
                parsed = _to_unix(self.metadata[key])
                if parsed is not None:
                    return parsed
        return None

    # ── classification ────────────────────────────────────────────────────

    @property
    def sensitivity(self) -> str:
        """Sensitivity label, lower-cased; ``"unclassified"`` when absent."""
        label = self._first_str(_SENSITIVITY_KEYS)
        return label.lower() if label else "unclassified"

    @property
    def allowed_use_cases(self) -> frozenset[str]:
        """Declared admissible use cases; empty set means unrestricted."""
        for key in _USE_CASE_KEYS:
            if key not in self.metadata:
                continue
            value = self.metadata[key]
            if isinstance(value, str):
                parts = [p.strip().lower() for p in value.split(",") if p.strip()]
                return frozenset(parts)
            if isinstance(value, Sequence):
                parts = [str(p).strip().lower() for p in value if str(p).strip()]
                return frozenset(parts)
        return frozenset()

    # ── helpers ───────────────────────────────────────────────────────────

    def _first_str(self, keys: tuple[str, ...]) -> str:
        for key in keys:
            value = self.metadata.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""


def _to_unix(value: Any) -> float | None:
    """Coerce a metadata timestamp to epoch seconds.

    Accepts numeric epoch seconds (int/float, or numeric string) and RFC-3339
    strings (``Z`` or offset). Returns ``None`` for unparsable values rather
    than raising, so one malformed field cannot crash admission.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        normalised = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalised).timestamp()
    except ValueError:
        return None
