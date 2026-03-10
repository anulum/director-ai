# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Fine-tuning API Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json

import pytest

from director_ai.finetune_api import (
    FinetuneJob,
    _JobStore,
    _MAX_CONCURRENT_JOBS,
    create_finetune_router,
)


class TestJobStore:
    def test_create_and_get(self):
        store = _JobStore()
        job = store.create({"epochs": 3})
        assert job.state == "pending"
        assert job.job_id
        fetched = store.get(job.job_id)
        assert fetched is job

    def test_get_nonexistent(self):
        store = _JobStore()
        assert store.get("does-not-exist") is None

    def test_list_all(self):
        store = _JobStore()
        store.create({"epochs": 1})
        store.create({"epochs": 2})
        jobs = store.list_all()
        assert len(jobs) == 2

    def test_delete(self):
        store = _JobStore()
        job = store.create({"epochs": 3})
        assert store.delete(job.job_id)
        assert store.get(job.job_id) is None

    def test_delete_nonexistent(self):
        store = _JobStore()
        assert not store.delete("nope")

    def test_concurrent_job_limit(self):
        store = _JobStore()
        for i in range(_MAX_CONCURRENT_JOBS):
            job = store.create({"epochs": 1})
            job.state = "training"
        with pytest.raises(ValueError, match="Too many"):
            store.create({"epochs": 1})


class TestFinetuneJob:
    def test_defaults(self):
        job = FinetuneJob(job_id="test-123")
        assert job.state == "pending"
        assert job.progress == 0.0
        assert job.activated is False
        assert job.error == ""

    def test_state_transitions(self):
        job = FinetuneJob(job_id="test-456")
        job.state = "training"
        job.progress = 0.5
        job.state = "completed"
        job.progress = 1.0
        assert job.state == "completed"
        assert job.progress == 1.0


class TestCreateRouter:
    def test_router_creates(self, tmp_path):
        router = create_finetune_router(models_dir=tmp_path / "models")
        assert router is not None
        routes = [r.path for r in router.routes]
        assert "/validate" in routes
        assert "/start" in routes
        assert "/{job_id}" in routes
        assert "/{job_id}/result" in routes
        assert "/{job_id}/activate" in routes
        assert "/{job_id}/rollback" in routes
        assert "/" in routes

    def test_models_dir_created(self, tmp_path):
        models_dir = tmp_path / "new_models"
        create_finetune_router(models_dir=models_dir)
        assert models_dir.exists()
        assert (models_dir / "_uploads").exists()


class TestRouterEndpoints:
    """Integration tests using FastAPI TestClient."""

    @pytest.fixture
    def client(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        return TestClient(app)

    def _make_jsonl_bytes(self, n_pos=300, n_neg=300):
        rows = []
        for i in range(n_pos):
            rows.append({"premise": f"Fact {i}.", "hypothesis": f"Claim {i}.", "label": 1})
        for i in range(n_neg):
            rows.append({"premise": f"Source {i}.", "hypothesis": f"Wrong {i}.", "label": 0})
        return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")

    def test_validate_valid_data(self, client):
        data = self._make_jsonl_bytes()
        resp = client.post(
            "/v1/finetune/validate",
            files={"file": ("train.jsonl", data, "application/jsonl")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_valid"]
        assert body["total_samples"] == 600
        assert body["estimated_cost_usd"] > 0

    def test_validate_invalid_data(self, client):
        data = b'{"premise": "a"}\n{"hypothesis": "b"}\n'
        resp = client.post(
            "/v1/finetune/validate",
            files={"file": ("bad.jsonl", data, "application/jsonl")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert not body["is_valid"]
        assert len(body["errors"]) > 0

    def test_start_rejects_bad_data(self, client):
        data = b'not json\n'
        resp = client.post(
            "/v1/finetune/start",
            files={"file": ("bad.jsonl", data, "application/jsonl")},
        )
        assert resp.status_code == 422

    def test_list_models_empty(self, client):
        resp = client.get("/v1/finetune/")
        assert resp.status_code == 200
        assert resp.json()["models"] == []

    def test_get_nonexistent_job(self, client):
        resp = client.get("/v1/finetune/nonexistent")
        assert resp.status_code == 404

    def test_activate_nonexistent_job(self, client):
        resp = client.post("/v1/finetune/nonexistent/activate")
        assert resp.status_code == 404

    def test_rollback_nonexistent_job(self, client):
        resp = client.post("/v1/finetune/nonexistent/rollback")
        assert resp.status_code == 404

    def test_delete_nonexistent_job(self, client):
        resp = client.delete("/v1/finetune/nonexistent")
        assert resp.status_code == 404

    def test_result_nonexistent_job(self, client):
        resp = client.get("/v1/finetune/nonexistent/result")
        assert resp.status_code == 404


class TestRouterSuccessPaths:
    """Test activate/rollback/delete on real (mocked-completed) jobs."""

    @pytest.fixture
    def client_with_job(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from director_ai.finetune_api import _JobStore

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        # Inject a completed job directly into the store
        # Access the store via the router's closure
        store = None
        for route in router.routes:
            if hasattr(route, "endpoint") and route.endpoint.__name__ == "list_models":
                store = route.endpoint.__code__.co_freevars
                break

        # Create job via store directly — we need to access the store from the closure
        # Instead, just POST valid data and manually complete the job
        return client

    def test_activate_completed_job(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        # Manually create a completed job via the internal store
        from director_ai.finetune_api import FinetuneJob, _JobStore
        import types

        # Get the store from the router closure
        for route in router.routes:
            if hasattr(route, "endpoint"):
                fn = route.endpoint
                if hasattr(fn, "__closure__") and fn.__closure__:
                    for cell in fn.__closure__:
                        try:
                            obj = cell.cell_contents
                            if isinstance(obj, _JobStore):
                                store = obj
                                break
                        except ValueError:
                            pass

        job = store.create({"epochs": 1})
        job.state = "completed"
        job.model_path = str(tmp_path / "models" / job.job_id)

        resp = client.post(f"/v1/finetune/{job.job_id}/activate")
        assert resp.status_code == 200
        assert resp.json()["activated"] is True

    def test_rollback_activated_job(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from director_ai.finetune_api import _JobStore

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        store = None
        for route in router.routes:
            if hasattr(route, "endpoint") and hasattr(route.endpoint, "__closure__") and route.endpoint.__closure__:
                for cell in route.endpoint.__closure__:
                    try:
                        obj = cell.cell_contents
                        if isinstance(obj, _JobStore):
                            store = obj
                            break
                    except ValueError:
                        pass
                if store:
                    break

        job = store.create({"epochs": 1})
        job.state = "completed"
        job.activated = True

        resp = client.post(f"/v1/finetune/{job.job_id}/rollback")
        assert resp.status_code == 200
        assert resp.json()["activated"] is False

    def test_delete_completed_job(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from director_ai.finetune_api import _JobStore

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        store = None
        for route in router.routes:
            if hasattr(route, "endpoint") and hasattr(route.endpoint, "__closure__") and route.endpoint.__closure__:
                for cell in route.endpoint.__closure__:
                    try:
                        obj = cell.cell_contents
                        if isinstance(obj, _JobStore):
                            store = obj
                            break
                    except ValueError:
                        pass
                if store:
                    break

        job = store.create({"epochs": 1})
        job.state = "completed"

        resp = client.delete(f"/v1/finetune/{job.job_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        resp2 = client.get(f"/v1/finetune/{job.job_id}")
        assert resp2.status_code == 404

    def test_delete_activated_blocked(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from director_ai.finetune_api import _JobStore

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        store = None
        for route in router.routes:
            if hasattr(route, "endpoint") and hasattr(route.endpoint, "__closure__") and route.endpoint.__closure__:
                for cell in route.endpoint.__closure__:
                    try:
                        obj = cell.cell_contents
                        if isinstance(obj, _JobStore):
                            store = obj
                            break
                    except ValueError:
                        pass
                if store:
                    break

        job = store.create({"epochs": 1})
        job.state = "completed"
        job.activated = True

        resp = client.delete(f"/v1/finetune/{job.job_id}")
        assert resp.status_code == 409

    def test_result_completed_job(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from director_ai.finetune_api import _JobStore

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        store = None
        for route in router.routes:
            if hasattr(route, "endpoint") and hasattr(route.endpoint, "__closure__") and route.endpoint.__closure__:
                for cell in route.endpoint.__closure__:
                    try:
                        obj = cell.cell_contents
                        if isinstance(obj, _JobStore):
                            store = obj
                            break
                    except ValueError:
                        pass
                if store:
                    break

        job = store.create({"epochs": 1})
        job.state = "completed"
        job.metrics = {"eval_balanced_accuracy": 0.85}
        job.regression_report = {"recommendation": "deploy"}

        resp = client.get(f"/v1/finetune/{job.job_id}/result")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == "completed"
        assert body["metrics"]["eval_balanced_accuracy"] == 0.85
        assert body["regression_report"]["recommendation"] == "deploy"


class TestRouterIsolation:
    def test_separate_routers_have_separate_stores(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        r1 = create_finetune_router(models_dir=tmp_path / "m1")
        r2 = create_finetune_router(models_dir=tmp_path / "m2")
        app.include_router(r1, prefix="/v1/ft1")
        app.include_router(r2, prefix="/v1/ft2")
        client = TestClient(app)

        resp1 = client.get("/v1/ft1/")
        resp2 = client.get("/v1/ft2/")
        assert resp1.json()["models"] == []
        assert resp2.json()["models"] == []


class TestBenchmarkJsonlRobust:
    def test_malformed_json_skipped(self, tmp_path):
        from director_ai.core.finetune_benchmark import _load_benchmark_jsonl

        f = tmp_path / "bench.jsonl"
        f.write_text(
            '{"premise":"a","hypothesis":"b","label":1}\n'
            'not json at all\n'
            '{"premise":"c","hypothesis":"d","label":0}\n',
            encoding="utf-8",
        )
        rows = _load_benchmark_jsonl(f)
        assert len(rows) == 2
