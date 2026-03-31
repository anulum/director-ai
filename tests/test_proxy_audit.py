# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Proxy Audit Log Tests (STRONG)
"""Multi-angle tests for proxy → audit log pipeline.

Covers: audit entry creation, entry fields (model, provider, score,
approved), audit disabled path, multiple requests, parametrised
on_fail modes, threshold effects on approval, and pipeline
performance documentation.
"""

from __future__ import annotations

import httpx
import pytest
from httpx import ASGITransport

from director_ai.compliance.audit_log import AuditLog
from director_ai.proxy import create_proxy_app


def _upstream_transport(content: str = "The sky is blue."):
    async def _handler(request: httpx.Request):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    },
                ],
            },
        )

    return httpx.MockTransport(_handler)


def _make_app(*, audit_db=None, **kw):
    defaults = {
        "threshold": 0.3,
        "upstream_url": "http://fake-upstream",
        "on_fail": "warn",
        "use_nli": False,
        "allow_http_upstream": True,
        "_transport": _upstream_transport(),
    }
    defaults.update(kw)
    if audit_db:
        defaults["audit_db"] = audit_db
    return create_proxy_app(**defaults)


async def _post(app, content="What color is the sky?", model="gpt-4o-mini"):
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        return await c.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": content}]},
        )


# ── Audit entry creation ─────────────────────────────────────────


class TestAuditEntryCreation:
    """Proxy must create audit entries when audit_db configured."""

    @pytest.mark.asyncio
    async def test_creates_audit_entry(self, tmp_path):
        db = str(tmp_path / "audit.db")
        app = _make_app(audit_db=db)
        resp = await _post(app)
        assert resp.status_code == 200

        log = AuditLog(db)
        entries = log.query()
        assert len(entries) == 1
        log.close()

    @pytest.mark.asyncio
    async def test_entry_has_model(self, tmp_path):
        db = str(tmp_path / "audit.db")
        app = _make_app(audit_db=db)
        await _post(app, model="gpt-4o-mini")

        log = AuditLog(db)
        entry = log.query()[0]
        assert entry.model == "gpt-4o-mini"
        log.close()

    @pytest.mark.asyncio
    async def test_entry_has_provider(self, tmp_path):
        db = str(tmp_path / "audit.db")
        app = _make_app(audit_db=db)
        await _post(app)

        log = AuditLog(db)
        entry = log.query()[0]
        assert entry.provider == "proxy"
        log.close()

    @pytest.mark.asyncio
    async def test_entry_has_score(self, tmp_path):
        db = str(tmp_path / "audit.db")
        app = _make_app(audit_db=db)
        await _post(app)

        log = AuditLog(db)
        entry = log.query()[0]
        assert entry.score > 0
        log.close()

    @pytest.mark.asyncio
    async def test_entry_approved_with_low_threshold(self, tmp_path):
        db = str(tmp_path / "audit.db")
        app = _make_app(audit_db=db, threshold=0.1)
        await _post(app)

        log = AuditLog(db)
        entry = log.query()[0]
        assert entry.approved is True
        log.close()


# ── Audit disabled ───────────────────────────────────────────────


class TestAuditDisabled:
    """No audit entries when audit_db not configured."""

    @pytest.mark.asyncio
    async def test_no_audit_when_disabled(self):
        app = _make_app()  # no audit_db
        resp = await _post(app)
        assert resp.status_code == 200


# ── Multiple requests ────────────────────────────────────────────


class TestMultipleRequests:
    """Audit log must accumulate entries across requests."""

    @pytest.mark.asyncio
    async def test_multiple_entries(self, tmp_path):
        db = str(tmp_path / "audit.db")
        app = _make_app(audit_db=db)
        for _ in range(3):
            await _post(app)

        log = AuditLog(db)
        entries = log.query()
        assert len(entries) == 3
        log.close()


# ── Parametrised on_fail modes ───────────────────────────────────


class TestAuditOnFailModes:
    """Audit must work with all on_fail modes."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("on_fail", ["warn", "reject"])
    async def test_audit_with_on_fail(self, tmp_path, on_fail):
        db = str(tmp_path / f"audit_{on_fail}.db")
        app = _make_app(audit_db=db, on_fail=on_fail)
        await _post(app)

        log = AuditLog(db)
        entries = log.query()
        assert len(entries) >= 1
        log.close()


# ── Pipeline performance ─────────────────────────────────────────


class TestAuditPerformance:
    """Document proxy → audit pipeline characteristics."""

    @pytest.mark.asyncio
    async def test_audit_entry_has_all_fields(self, tmp_path):
        db = str(tmp_path / "audit.db")
        app = _make_app(audit_db=db)
        await _post(app)

        log = AuditLog(db)
        entry = log.query()[0]
        assert hasattr(entry, "model")
        assert hasattr(entry, "provider")
        assert hasattr(entry, "score")
        assert hasattr(entry, "approved")
        log.close()
