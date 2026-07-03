# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for Redis enterprise retrieval and cache wiring."""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("redis", reason="redis client required for Redis integration tests")

import redis

import director_ai.enterprise.redis as redis_module
from director_ai.core.config import DirectorConfig
from director_ai.core.config_builders import build_store
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _free_local_port() -> int:
    """Reserve and return an available loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _redis_client(redis_url: str) -> Any:
    """Return a decode-responses Redis client for the ephemeral server."""
    return redis.Redis.from_url(redis_url, decode_responses=True)


def _wait_for_redis(redis_url: str) -> None:
    """Block until the ephemeral Redis server answers PING."""
    deadline = time.monotonic() + 5.0
    client = _redis_client(redis_url)
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            if client.ping():
                return
        except BaseException as exc:  # Redis raises concrete optional classes.
            last_error = exc
        time.sleep(0.05)
    raise RuntimeError("ephemeral Redis server did not start") from last_error


def _stop_redis_process(process: subprocess.Popen[bytes], redis_url: str) -> None:
    """Stop the ephemeral Redis process without hiding a failed shutdown path."""
    try:
        _redis_client(redis_url).shutdown(nosave=True)
    except redis.RedisError:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)
        return

    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


@pytest.fixture()
def redis_url(tmp_path: Path) -> Iterator[str]:
    """Start a real local Redis process and yield its connection URL."""
    server = shutil.which("redis-server")
    if server is None:
        pytest.fail("redis-server binary is required for real Redis integration tests")

    port = _free_local_port()
    redis_dir = tmp_path / "redis"
    redis_dir.mkdir()
    command = [
        server,
        "--bind",
        "127.0.0.1",
        "--port",
        str(port),
        "--dir",
        str(redis_dir),
        "--save",
        "",
        "--appendonly",
        "no",
        "--protected-mode",
        "yes",
        "--loglevel",
        "warning",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"redis://127.0.0.1:{port}/0"
    try:
        _wait_for_redis(url)
        yield url
    finally:
        _stop_redis_process(process, url)


def test_redis_enterprise_unit_guard_declares_this_real_surface_companion() -> None:
    """The unit guard manifest must point at this production-surface companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_redis_enterprise.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_redis_enterprise_real_surface.py" in reason


def test_real_redis_ground_truth_store_preserves_tenant_boundaries(
    redis_url: str,
) -> None:
    """RedisGroundTruthStore should retrieve only facts for the requested tenant."""
    store = redis_module.RedisGroundTruthStore(
        redis_url=redis_url,
        prefix="dai:test:facts:",
    )
    store.add(
        "refund policy",
        "Tenant alpha refunds require signed approval evidence.",
        tenant_id="tenant-alpha",
    )
    store.add(
        "refund policy",
        "Tenant beta refunds require finance approval evidence.",
        tenant_id="tenant-beta",
    )
    store.add_many(
        {
            "rollback policy": "Tenant alpha rollback requires operator attestation.",
            "audit packet": "Tenant alpha audit packets retain source hashes.",
        },
        tenant_id="tenant-alpha",
    )

    alpha_context = store.retrieve_context(
        "refund rollback audit policy",
        tenant_id="tenant-alpha",
    )
    beta_context = store.retrieve_context("refund policy", tenant_id="tenant-beta")
    alpha_top_one = store.retrieve_context(
        "refund rollback audit policy",
        top_k=1,
        tenant_id="tenant-alpha",
    )

    assert store.count(tenant_id="tenant-alpha") == 3
    assert store.count(tenant_id="tenant-beta") == 1
    assert alpha_context is not None
    assert "Tenant alpha refunds" in alpha_context
    assert "Tenant alpha rollback" in alpha_context
    assert "Tenant beta" not in alpha_context
    assert beta_context == "Tenant beta refunds require finance approval evidence."
    assert alpha_top_one == "Tenant alpha refunds require signed approval evidence."


def test_real_redis_score_cache_enforces_ttl_generation_and_prefix_clear(
    redis_url: str,
) -> None:
    """RedisScoreCache should persist score entries through Redis semantics."""
    cache = redis_module.RedisScoreCache(
        redis_url=redis_url,
        prefix="dai:test:cache:",
        ttl_seconds=1,
    )

    cache.put(
        "What is the refund policy?",
        "Tenant alpha refunds require signed approval evidence.",
        0.87,
        0.11,
        0.22,
        tenant_id="tenant-alpha",
        scope="store-rev-1",
    )

    hit = cache.get(
        "What is the refund policy?",
        "Tenant alpha refunds require signed approval evidence.",
        tenant_id="tenant-alpha",
        scope="store-rev-1",
    )
    other_tenant = cache.get(
        "What is the refund policy?",
        "Tenant alpha refunds require signed approval evidence.",
        tenant_id="tenant-beta",
        scope="store-rev-1",
    )

    assert hit is not None
    assert hit.score == pytest.approx(0.87)
    assert hit.h_logical == pytest.approx(0.11)
    assert hit.h_factual == pytest.approx(0.22)
    assert other_tenant is None
    assert cache.hits == 1
    assert cache.misses == 1
    assert cache.size == 1

    cache.invalidate()
    stale = cache.get(
        "What is the refund policy?",
        "Tenant alpha refunds require signed approval evidence.",
        tenant_id="tenant-alpha",
        scope="store-rev-1",
    )

    assert stale is None
    assert cache.size == 0

    cache.put("q1", "p1", 0.5, 0.0, 0.0)
    cache.put("q2", "p2", 0.6, 0.0, 0.0)
    assert cache.size == 2
    cache.clear()
    assert cache.size == 0
    assert cache.hits == 0
    assert cache.misses == 0


def test_config_build_store_uses_real_redis_backend(redis_url: str) -> None:
    """DirectorConfig store construction should open the Redis enterprise store."""
    store = build_store(
        DirectorConfig(
            redis_url=redis_url,
            redis_prefix="dai:test:builder:",
        ),
    )

    assert isinstance(store, redis_module.RedisGroundTruthStore)
    store.add("evidence policy", "Redis-backed evidence remains tenant scoped.")
    context = store.retrieve_context("evidence policy")

    assert context == "Redis-backed evidence remains tenant scoped."
