# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — tenant listing and knowledge-write routes

"""The /v1/tenants listing and tenant knowledge-write routes.

Split out of the ``create_app`` factory. Handlers read the tenant router from
``request.app.state`` and the write-access controls from
``request.app.state.config``, so ``create_tenants_router`` needs no
construction-time dependencies. The tenant-binding, write-access, and signature
checks are module-level helpers that resolve config from the request.
"""

from __future__ import annotations

from typing import Any

from ..core.kb_write_security import (
    KBWriteAccessError,
    canonical_kb_payload,
    check_kb_write_access,
    parse_hmac_keys,
    verify_kb_payload_signature,
)

try:
    from fastapi import APIRouter, HTTPException, Request

    from .._server_models import (
        StatusResponse,
        TenantFactRequest,
        TenantListResponse,
        TenantVectorFactRequest,
    )

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - server extras absent
    _FASTAPI_AVAILABLE = False


def _enforce_tenant_binding(request: Request, tenant_id: str) -> None:
    """Reject cross-tenant writes for tenant-bound API keys."""
    bound = getattr(request.state, "tenant_id", "")
    if bound and bound != tenant_id:
        raise HTTPException(403, "API key not authorized for this tenant")


def _enforce_kb_write_access(request: Request, tenant_id: str) -> None:
    """Enforce configured knowledge-base write access controls."""
    cfg = request.app.state.config
    try:
        check_kb_write_access(
            require_auth=cfg.knowledge_write_require_auth,
            require_tenant_binding=cfg.knowledge_write_require_tenant_binding,
            authenticated=bool(getattr(request.state, "kb_write_key_ok", False)),
            tenant_binding_enforced=bool(
                getattr(request.state, "kb_tenant_binding_ok", False)
            ),
            bound_tenant=getattr(request.state, "tenant_id", ""),
            requested_tenant=tenant_id,
        )
    except KBWriteAccessError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc


def _kb_signature_metadata(
    request: Request,
    canonical_payload: str,
    signature: str,
    key_id: str,
) -> dict[str, object]:
    """Verify tenant knowledge writes and return signature metadata."""
    cfg = request.app.state.config
    clean_signature = signature.strip()
    clean_key_id = key_id.strip()
    if not clean_signature:
        if cfg.knowledge_write_require_signature:
            raise HTTPException(403, "Knowledge-base write signature required")
        return {}
    if not verify_kb_payload_signature(
        canonical_payload,
        clean_signature,
        parse_hmac_keys(cfg.knowledge_write_hmac_keys),
        clean_key_id,
    ):
        raise HTTPException(403, "Invalid knowledge-base write signature")
    return {
        "kb_signature": clean_signature,
        "kb_signature_key_id": clean_key_id,
        "kb_signature_verified": True,
    }


def create_tenants_router() -> APIRouter:
    """Build the tenant route group (list, add fact, add vector fact)."""
    if not _FASTAPI_AVAILABLE:  # pragma: no cover - guarded by create_app
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install director-ai[server]",
        )

    router = APIRouter()

    @router.get("/v1/tenants", response_model=TenantListResponse)
    async def list_tenants(request: Request) -> dict[str, Any]:
        """List tenants visible to the authenticated caller."""
        tenant_router = request.app.state._state.get("tenant_router")
        if not tenant_router:
            raise HTTPException(404, "Tenant routing not enabled")
        bound = getattr(request.state, "tenant_id", "")
        visible = [bound] if bound else tenant_router.tenant_ids
        return {
            "tenants": [
                {"id": tid, "fact_count": tenant_router.fact_count(tid)}
                for tid in visible
                if tid in tenant_router.tenant_ids
            ],
        }

    @router.post("/v1/tenants/{tenant_id}/facts", response_model=StatusResponse)
    async def add_tenant_fact(
        request: Request, tenant_id: str, req: TenantFactRequest
    ) -> dict[str, Any]:
        """Add a scalar tenant fact after tenant and write checks."""
        tenant_router = request.app.state._state.get("tenant_router")
        if not tenant_router:
            raise HTTPException(404, "Tenant routing not enabled")
        _enforce_tenant_binding(request, tenant_id)
        _enforce_kb_write_access(request, tenant_id)
        _kb_signature_metadata(
            request,
            canonical_kb_payload(
                kind="tenant_fact",
                tenant_id=tenant_id,
                key=req.key,
                value=req.value,
            ),
            req.signature,
            req.signature_key_id,
        )
        tenant_router.add_fact(tenant_id, req.key, req.value)
        return {"status": "ok", "tenant_id": tenant_id, "key": req.key}

    @router.post("/v1/tenants/{tenant_id}/vector-facts", response_model=StatusResponse)
    async def add_tenant_vector_fact(
        request: Request,
        tenant_id: str,
        req: TenantVectorFactRequest,
    ) -> dict[str, Any]:
        """Add a tenant-scoped vector fact to the configured vector store."""
        tenant_router = request.app.state._state.get("tenant_router")
        if not tenant_router:
            raise HTTPException(404, "Tenant routing not enabled")
        _enforce_tenant_binding(request, tenant_id)
        _enforce_kb_write_access(request, tenant_id)
        sig_meta = _kb_signature_metadata(
            request,
            canonical_kb_payload(
                kind="tenant_vector_fact",
                tenant_id=tenant_id,
                key=req.key,
                value=req.value,
            ),
            req.signature,
            req.signature_key_id,
        )
        try:
            store = tenant_router.get_vector_store(
                tenant_id, backend_type=req.backend_type
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, f"Invalid backend_type: {exc}") from exc
        store.add_fact(req.key, req.value, metadata=sig_meta)
        return {
            "status": "ok",
            "tenant_id": tenant_id,
            "key": req.key,
            "backend_type": req.backend_type,
            "count": store.backend.count(),
        }

    return router
