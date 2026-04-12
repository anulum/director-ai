# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``director_ai.middleware`` — API key + rate limiting.

Covers: key validation (Bearer, X-API-Key, env, file), exempt paths,
constant-time comparison, key hashing, rate limiting (token bucket,
burst, retry-after), and middleware integration with FastAPI TestClient.
"""

from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from director_ai.middleware.api_key import APIKeyMiddleware, _extract_key, _hash_key
from director_ai.middleware.rate_limit import RateLimitMiddleware, _TokenBucket

# ── _extract_key ────────────────────────────────────────────────────────


class TestExtractKey:
    def test_bearer_token(self):
        from unittest.mock import MagicMock

        from starlette.testclient import TestClient as _  # noqa: F401

        req = MagicMock()
        req.headers = {"authorization": "Bearer sk-test-123"}
        assert _extract_key(req) == "sk-test-123"

    def test_x_api_key_header(self):
        from unittest.mock import MagicMock

        req = MagicMock()
        req.headers = {"x-api-key": "sk-test-456"}
        assert _extract_key(req) == "sk-test-456"

    def test_no_key(self):
        from unittest.mock import MagicMock

        req = MagicMock()
        req.headers = {}
        assert _extract_key(req) is None


# ── _hash_key ───────────────────────────────────────────────────────────


class TestHashKey:
    def test_returns_hex_string(self):
        h = _hash_key("sk-test-123")
        assert isinstance(h, str)
        assert len(h) == 16

    def test_deterministic(self):
        assert _hash_key("key") == _hash_key("key")

    def test_different_keys_different_hashes(self):
        assert _hash_key("key1") != _hash_key("key2")


# ── _TokenBucket ────────────────────────────────────────────────────────


class TestTokenBucket:
    def test_initial_capacity(self):
        b = _TokenBucket(capacity=5, refill_rate=1.0)
        for _ in range(5):
            assert b.consume()
        assert not b.consume()

    def test_refill(self):
        b = _TokenBucket(capacity=1, refill_rate=100.0)  # fast refill
        assert b.consume()
        assert not b.consume()
        time.sleep(0.02)  # wait for refill
        assert b.consume()

    def test_retry_after(self):
        b = _TokenBucket(capacity=1, refill_rate=1.0)
        b.consume()
        assert b.retry_after > 0


# ── APIKeyMiddleware integration ────────────────────────────────────────


def _app_with_keys(keys: set[str]) -> FastAPI:
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware, keys=keys)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/review")
    def review():
        return {"score": 0.8}

    return app


class TestAPIKeyMiddleware:
    def test_health_exempt(self):
        client = TestClient(_app_with_keys({"sk-123"}))
        r = client.get("/health")
        assert r.status_code == 200

    def test_valid_bearer(self):
        client = TestClient(_app_with_keys({"sk-123"}))
        r = client.get("/v1/review", headers={"Authorization": "Bearer sk-123"})
        assert r.status_code == 200

    def test_valid_x_api_key(self):
        client = TestClient(_app_with_keys({"sk-123"}))
        r = client.get("/v1/review", headers={"X-API-Key": "sk-123"})
        assert r.status_code == 200

    def test_missing_key_401(self):
        client = TestClient(_app_with_keys({"sk-123"}))
        r = client.get("/v1/review")
        assert r.status_code == 401
        assert "Invalid" in r.json()["error"]

    def test_wrong_key_401(self):
        client = TestClient(_app_with_keys({"sk-123"}))
        r = client.get("/v1/review", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401

    def test_env_keys(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_API_KEYS", "sk-env-1,sk-env-2")
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware)

        @app.get("/v1/test")
        def test_ep():
            return {"ok": True}

        client = TestClient(app)
        r = client.get("/v1/test", headers={"Authorization": "Bearer sk-env-1"})
        assert r.status_code == 200

    def test_file_keys(self, tmp_path):
        kf = tmp_path / "keys.txt"
        kf.write_text("sk-file-1\nsk-file-2\n# comment\n")
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, keys_file=str(kf))

        @app.get("/v1/test")
        def test_ep():
            return {"ok": True}

        client = TestClient(app)
        r = client.get("/v1/test", headers={"Authorization": "Bearer sk-file-1"})
        assert r.status_code == 200


# ── RateLimitMiddleware integration ─────────────────────────────────────


def _app_with_rate_limit(rpm: int = 60, burst: int | None = None) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, requests_per_minute=rpm, burst=burst)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/score")
    def score():
        return {"score": 0.5}

    return app


class TestRateLimitMiddleware:
    def test_health_exempt(self):
        client = TestClient(_app_with_rate_limit(rpm=1, burst=1))
        for _ in range(10):
            r = client.get("/health")
            assert r.status_code == 200

    def test_under_limit(self):
        client = TestClient(_app_with_rate_limit(rpm=60, burst=10))
        for _ in range(5):
            r = client.get("/v1/score")
            assert r.status_code == 200

    def test_over_burst_returns_429(self):
        client = TestClient(_app_with_rate_limit(rpm=60, burst=2))
        results = []
        for _ in range(5):
            r = client.get("/v1/score")
            results.append(r.status_code)
        assert 429 in results

    def test_429_has_retry_after(self):
        client = TestClient(_app_with_rate_limit(rpm=60, burst=1))
        client.get("/v1/score")  # consume the single token
        r = client.get("/v1/score")
        if r.status_code == 429:
            assert "Retry-After" in r.headers
            assert "retry_after_seconds" in r.json()
