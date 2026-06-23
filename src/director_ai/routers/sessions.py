# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — conversation session routes

"""The /v1/sessions/{session_id} read and delete routes.

Split out of the ``create_app`` factory. Both handlers read the session store
and ownership map from ``request.app.state`` under the shared sessions lock, so
``create_sessions_router`` needs no construction-time dependencies. Access is
gated to the owning API-key identity.
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter, HTTPException, Request

    from .._server_models import DeletedResponse, SessionResponse

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - server extras absent
    _FASTAPI_AVAILABLE = False


def create_sessions_router() -> APIRouter:
    """Build the conversation-session route group (get, delete)."""
    if not _FASTAPI_AVAILABLE:  # pragma: no cover - guarded by create_app
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install director-ai[server]",
        )

    router = APIRouter()

    @router.get("/v1/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(request: Request, session_id: str) -> dict[str, Any]:
        """Return a session only to its owning API-key identity."""
        caller_hash = getattr(request.state, "api_key_hash", "")
        async with request.app.state._state["sessions_lock"]:
            sessions = request.app.state._state["sessions"]
            owners = request.app.state._state["session_owners"]
            if session_id not in sessions:
                raise HTTPException(404, "Session not found")
            owner = owners.get(session_id, "")
            if owner and owner != caller_hash:
                raise HTTPException(404, "Session not found")
            s = sessions[session_id]
        return {
            "session_id": s.session_id,
            "turn_count": len(s),
            "turns": [
                {
                    "prompt": t.prompt,
                    "response": t.response,
                    "score": t.score,
                    "turn_index": t.turn_index,
                }
                for t in s.turns
            ],
        }

    @router.delete("/v1/sessions/{session_id}", response_model=DeletedResponse)
    async def delete_session(request: Request, session_id: str) -> dict[str, Any]:
        """Delete a session owned by the authenticated API-key identity."""
        caller_hash = getattr(request.state, "api_key_hash", "")
        async with request.app.state._state["sessions_lock"]:
            sessions = request.app.state._state["sessions"]
            owners = request.app.state._state["session_owners"]
            if session_id not in sessions:
                raise HTTPException(404, "Session not found")
            owner = owners.get(session_id, "")
            if owner and owner != caller_hash:
                raise HTTPException(404, "Session not found")
            del sessions[session_id]
            owners.pop(session_id, None)
        return {"status": "deleted", "session_id": session_id}

    return router
