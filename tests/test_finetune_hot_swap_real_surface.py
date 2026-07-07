# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - live fine-tuned-scorer hot-swap real-surface tests
"""Real-surface coverage for live fine-tuned-scorer hot-swap (BUG-2).

The fine-tune ``/activate`` endpoint used to only flip a flag while its docstring
claimed to activate the model. These tests exercise the real swap plumbing: the
sync bundle rebuild, the serialised async activator, and the HTTP route swapping
the live scorer that the review path reads on the mounted server app.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="fastapi required for hot-swap server tests")

from fastapi.testclient import TestClient

from director_ai import server
from director_ai.core.config import DirectorConfig
from director_ai.core.runtime.batch import BatchProcessor


def _fast_state() -> dict[str, Any]:
    """Build a real scoring bundle in the fast (heuristic, no-NLI) profile."""
    cfg = DirectorConfig.from_profile("fast")
    store = cfg.build_store()
    scorer = cfg.build_scorer(store=store)
    agent = server._build_coherence_agent(cfg, scorer, store)
    batch = BatchProcessor(agent, max_concurrency=cfg.batch_max_concurrency)
    return {
        "config": cfg,
        "store": store,
        "scorer": scorer,
        "agent": agent,
        "batch": batch,
        "review_queue": None,
        "scorer_swap_lock": asyncio.Lock(),
    }


def test_swap_scorer_replaces_bundle_and_reuses_store() -> None:
    """``_swap_scorer`` swaps scorer/agent/batch, keeps the store, tracks config."""
    state = _fast_state()
    old_scorer = state["scorer"]
    old_agent = state["agent"]
    old_batch = state["batch"]
    old_store = state["store"]

    server._swap_scorer(state, "acme/fine-tuned-nli")

    assert state["scorer"] is not old_scorer
    assert state["agent"] is not old_agent
    assert state["batch"] is not old_batch
    # The ground-truth store is reused so runtime-added facts survive the swap.
    assert state["store"] is old_store
    assert state["config"].nli_model == "acme/fine-tuned-nli"
    # The swapped-in scorer is a working CoherenceScorer, not a raising stub.
    approved, score = state["scorer"].review("What is 2+2?", "2+2 is 4.")
    assert isinstance(approved, bool)
    assert score is not None


def test_swapped_bundle_agent_wraps_the_new_scorer() -> None:
    """The rebuilt agent must wrap the freshly built scorer, not the old one."""
    state = _fast_state()
    server._swap_scorer(state, "acme/fine-tuned-nli")
    # The agent holds its scorer privately; it must be the swapped-in instance so
    # the /process path (which reads the agent) also serves the new model.
    assert state["agent"].scorer is state["scorer"]


def test_activate_scorer_serialises_concurrent_activations() -> None:
    """Concurrent ``_activate_scorer`` calls are serialised by the swap lock."""
    state = _fast_state()
    old_scorer = state["scorer"]

    async def _run() -> None:
        await asyncio.gather(
            server._activate_scorer(state, "acme/model-a"),
            server._activate_scorer(state, "acme/model-b"),
        )

    asyncio.run(_run())

    # Both activations completed without interleaving corruption; the surviving
    # config is exactly one of the two requested models and the bundle is fresh.
    assert state["config"].nli_model in {"acme/model-a", "acme/model-b"}
    assert state["scorer"] is not old_scorer
    assert state["agent"].scorer is state["scorer"]


def test_activate_scorer_rebuilds_running_review_queue() -> None:
    """A live review queue is rebuilt around the new scorer and the old drained."""
    from director_ai.core.runtime.review_queue import ReviewQueue

    state = _fast_state()
    old_scorer = state["scorer"]

    async def _run() -> None:
        old_queue = ReviewQueue(
            state["scorer"],
            max_batch=state["config"].review_queue_max_batch,
            flush_timeout_ms=state["config"].review_queue_flush_timeout_ms,
        )
        await old_queue.start()
        state["review_queue"] = old_queue

        await server._activate_scorer(state, "acme/fine-tuned-nli")

        # The queue was rebuilt around the swapped-in scorer and replaced.
        assert state["review_queue"] is not old_queue
        assert state["scorer"] is not old_scorer
        await state["review_queue"].stop()

    asyncio.run(_run())


def _finetune_store_from_app(app: Any) -> Any:
    """Discover the mounted fine-tune router's in-memory job store.

    There is no public API to seed a *completed* job (that would need real
    training), so a real-surface activation test reaches the router's own store
    through the route closures. The store is found by class name (not a private
    import) so the router's genuine object is exercised, not a fake.
    ``include_router`` wraps the mount in an ``_IncludedRouter`` whose sub-routes
    live on ``original_router`` rather than in ``app.routes``, so descend there.
    """

    def _leaf_routes(container: Any) -> list[Any]:
        original = getattr(container, "original_router", None)
        return list(original.routes) if original is not None else [container]

    for route in app.routes:
        for leaf in _leaf_routes(route):
            endpoint = getattr(leaf, "endpoint", None)
            closure = getattr(endpoint, "__closure__", None)
            if not closure:
                continue
            for cell in closure:
                try:
                    obj = cell.cell_contents
                except ValueError:
                    continue
                if type(obj).__name__ == "_JobStore":
                    return obj
    raise AssertionError("fine-tune job store not found on the app")


def test_activate_route_hot_swaps_live_scorer_over_real_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /activate rebuilds the live scorer that the review route reads."""
    monkeypatch.setenv("DIRECTOR_FORCE_CPU", "1")
    config = DirectorConfig.from_profile("fast")
    config.mode = "general"
    config.hybrid_retrieval = False
    config.reranker_enabled = False
    app = server.create_app(config)

    with TestClient(app) as client:
        store = _finetune_store_from_app(app)
        job = store.create({"epochs": 1})
        job.state = "completed"
        job.model_path = "acme/fine-tuned-nli"

        original_scorer = app.state._state["scorer"]

        resp = client.post(f"/v1/finetune/{job.job_id}/activate")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["hot_swapped"] is True
        assert body["activated"] is True
        assert body["model_path"] == "acme/fine-tuned-nli"
        assert "hot-swapped" in body["detail"]

        # The live scorer the review route reads has actually been replaced,
        # and the active config reflects the newly served model.
        assert app.state._state["scorer"] is not original_scorer
        assert app.state._state["config"].nli_model == "acme/fine-tuned-nli"

        # The review route still serves — now through the swapped-in scorer.
        review = client.post(
            "/v1/review",
            json={"prompt": "What is 2+2?", "response": "2+2 is 4."},
        )
        assert review.status_code == 200, review.text


def test_activate_route_standalone_marks_active_without_hot_swap(
    tmp_path: Any,
) -> None:
    """Mounted without a server scorer, /activate stays honest (no hot-swap)."""
    from fastapi import FastAPI

    from director_ai.finetune_api import create_finetune_router

    app = FastAPI()
    app.include_router(
        create_finetune_router(models_dir=tmp_path / "models"),
        prefix="/v1/finetune",
    )
    client = TestClient(app)

    store = _finetune_store_from_app(app)
    job = store.create({"epochs": 1})
    job.state = "completed"
    job.model_path = str(tmp_path / "models" / job.job_id)

    resp = client.post(f"/v1/finetune/{job.job_id}/activate")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["hot_swapped"] is False
    assert body["activated"] is True
    # Honest guidance: no live scorer to swap, so it points at the restart path.
    assert "nli_model" in body["detail"]
    assert "restart" in body["detail"]
