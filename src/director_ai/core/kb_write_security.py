# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - KB write security

"""Access and HMAC helpers for knowledge-base write surfaces."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any

__all__ = [
    "KBWriteAccessError",
    "canonical_kb_payload",
    "check_kb_write_access",
    "parse_hmac_keys",
    "sign_kb_payload",
    "verify_kb_payload_signature",
]


@dataclass(frozen=True)
class KBWriteAccessError(Exception):
    """Structured denial for a knowledge-base write."""

    status_code: int
    detail: str


def _sha256_hex(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def canonical_kb_payload(
    *,
    kind: str,
    tenant_id: str = "",
    doc_id: str = "",
    source: str = "",
    key: str = "",
    text: str = "",
    value: str = "",
    content: bytes = b"",
) -> str:
    """Return the canonical payload covered by a KB write signature."""
    payload: dict[str, str] = {
        "kind": kind,
        "tenant_id": tenant_id,
        "doc_id": doc_id,
        "source": source,
        "key": key,
    }
    if text:
        payload["text_sha256"] = _sha256_hex(text)
    if value:
        payload["value_sha256"] = _sha256_hex(value)
    if content:
        payload["content_sha256"] = _sha256_hex(content)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def parse_hmac_keys(raw: str) -> dict[str, str]:
    """Parse configured HMAC keys from JSON object or comma-separated values."""
    clean = raw.strip()
    if not clean:
        return {}
    try:
        parsed: Any = json.loads(clean)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return {
            str(key): str(value)
            for key, value in parsed.items()
            if str(key) and str(value)
        }

    keys: dict[str, str] = {}
    for index, part in enumerate(clean.split(",")):
        item = part.strip()
        if not item:
            continue
        if "=" in item:
            key_id, value = item.split("=", 1)
            keys[key_id.strip()] = value.strip()
        else:
            keys[f"k{index}"] = item
    return {key_id: value for key_id, value in keys.items() if key_id and value}


def sign_kb_payload(canonical_payload: str, hmac_key: str) -> str:
    """Sign a canonical KB payload with HMAC-SHA256."""
    return hmac.new(
        hmac_key.encode("utf-8"),
        canonical_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_kb_payload_signature(
    canonical_payload: str,
    signature: str,
    hmac_keys: dict[str, str],
    key_id: str = "",
) -> bool:
    """Verify a KB payload signature against one configured HMAC key."""
    clean_signature = signature.removeprefix("sha256=").strip()
    if not clean_signature or not hmac_keys:
        return False
    if key_id:
        selected = hmac_keys.get(key_id)
        if not selected:
            return False
        return hmac.compare_digest(
            clean_signature,
            sign_kb_payload(canonical_payload, selected),
        )
    return any(
        hmac.compare_digest(clean_signature, sign_kb_payload(canonical_payload, key))
        for key in hmac_keys.values()
    )


def check_kb_write_access(
    *,
    require_auth: bool,
    require_tenant_binding: bool,
    authenticated: bool,
    tenant_binding_enforced: bool,
    bound_tenant: str,
    requested_tenant: str,
) -> None:
    """Validate caller rights before a knowledge-base write."""
    if not require_auth:
        return
    if not authenticated:
        raise KBWriteAccessError(403, "Knowledge-base writes require authentication")
    if bound_tenant and requested_tenant and bound_tenant != requested_tenant:
        raise KBWriteAccessError(403, "Credential is not authorised for this tenant")
    if requested_tenant and require_tenant_binding and not tenant_binding_enforced:
        raise KBWriteAccessError(403, "Tenant write requires bound credential")
