# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Fine-tuning API Tests

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi", reason="fastapi required for finetune API tests")

from director_ai.finetune_api import (
    _MAX_CONCURRENT_JOBS,
    FinetuneJob,
    _JobStore,
    _run_training_worker,
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
        for _i in range(_MAX_CONCURRENT_JOBS):
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
            rows.append(
                {"premise": f"Fact {i}.", "hypothesis": f"Claim {i}.", "label": 1},
            )
        for i in range(n_neg):
            rows.append(
                {"premise": f"Source {i}.", "hypothesis": f"Wrong {i}.", "label": 0},
            )
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
        data = b"not json\n"
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

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        return TestClient(app)

    def test_activate_completed_job(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        # Manually create a completed job via the internal store

        from director_ai.finetune_api import _JobStore

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
            if (
                hasattr(route, "endpoint")
                and hasattr(route.endpoint, "__closure__")
                and route.endpoint.__closure__
            ):
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
            if (
                hasattr(route, "endpoint")
                and hasattr(route.endpoint, "__closure__")
                and route.endpoint.__closure__
            ):
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
            if (
                hasattr(route, "endpoint")
                and hasattr(route.endpoint, "__closure__")
                and route.endpoint.__closure__
            ):
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

    def test_activate_training_returns_409(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from director_ai.finetune_api import _JobStore

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        store = None
        for route in router.routes:
            if (
                hasattr(route, "endpoint")
                and hasattr(route.endpoint, "__closure__")
                and route.endpoint.__closure__
            ):
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
        job.state = "training"

        resp = client.post(f"/v1/finetune/{job.job_id}/activate")
        assert resp.status_code == 409

    def test_delete_cleans_model_directory(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from director_ai.finetune_api import _JobStore

        models_dir = tmp_path / "models"
        app = FastAPI()
        router = create_finetune_router(models_dir=models_dir)
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        store = None
        for route in router.routes:
            if (
                hasattr(route, "endpoint")
                and hasattr(route.endpoint, "__closure__")
                and route.endpoint.__closure__
            ):
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

        model_dir = models_dir / "test-model-dir"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

        job = store.create({"epochs": 1})
        job.state = "completed"
        job.model_path = str(model_dir)

        resp = client.delete(f"/v1/finetune/{job.job_id}")
        assert resp.status_code == 200
        assert not model_dir.exists()

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
            if (
                hasattr(route, "endpoint")
                and hasattr(route.endpoint, "__closure__")
                and route.endpoint.__closure__
            ):
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


class TestRouterStartEndpoint:
    """Test the /start endpoint with mocked training worker."""

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
            rows.append(
                {"premise": f"Fact {i}.", "hypothesis": f"Claim {i}.", "label": 1},
            )
        for i in range(n_neg):
            rows.append(
                {"premise": f"Source {i}.", "hypothesis": f"Wrong {i}.", "label": 0},
            )
        return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")

    @patch("director_ai.finetune_api._run_training_worker")
    def test_start_valid_data(self, mock_worker, client):
        data = self._make_jsonl_bytes()
        resp = client.post(
            "/v1/finetune/start",
            files={"file": ("train.jsonl", data, "application/jsonl")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        assert body["total_samples"] == 600
        assert body["estimated_time_min"] > 0
        assert mock_worker.called

    @patch("director_ai.finetune_api._run_training_worker")
    def test_start_returns_409_on_result_while_training(self, mock_worker, client):
        data = self._make_jsonl_bytes()
        resp = client.post(
            "/v1/finetune/start",
            files={"file": ("train.jsonl", data, "application/jsonl")},
        )
        job_id = resp.json()["job_id"]
        result_resp = client.get(f"/v1/finetune/{job_id}/result")
        assert result_resp.status_code == 409

    @patch("director_ai.finetune_api._run_training_worker")
    def test_start_get_status(self, mock_worker, client):
        data = self._make_jsonl_bytes()
        resp = client.post(
            "/v1/finetune/start",
            files={"file": ("train.jsonl", data, "application/jsonl")},
        )
        job_id = resp.json()["job_id"]
        status = client.get(f"/v1/finetune/{job_id}")
        assert status.status_code == 200
        assert status.json()["job_id"] == job_id

    @patch("director_ai.finetune_api._run_training_worker")
    def test_start_shows_in_list(self, mock_worker, client):
        data = self._make_jsonl_bytes()
        client.post(
            "/v1/finetune/start",
            files={"file": ("train.jsonl", data, "application/jsonl")},
        )
        listing = client.get("/v1/finetune/")
        assert len(listing.json()["models"]) == 1

    @patch("director_ai.finetune_api._run_training_worker")
    def test_start_concurrent_limit_429(self, mock_worker, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        # Fill up the concurrent limit
        store = None
        for route in router.routes:
            if (
                hasattr(route, "endpoint")
                and hasattr(route.endpoint, "__closure__")
                and route.endpoint.__closure__
            ):
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

        for _i in range(_MAX_CONCURRENT_JOBS):
            job = store.create({"epochs": 1})
            job.state = "training"

        data = self._make_jsonl_bytes()
        resp = client.post(
            "/v1/finetune/start",
            files={"file": ("train.jsonl", data, "application/jsonl")},
        )
        assert resp.status_code == 429


class TestBenchmarkJsonlRobust:
    def test_malformed_json_skipped(self, tmp_path):
        from director_ai.core.finetune_benchmark import _load_benchmark_jsonl

        f = tmp_path / "bench.jsonl"
        f.write_text(
            '{"premise":"a","hypothesis":"b","label":1}\n'
            "not json at all\n"
            '{"premise":"c","hypothesis":"d","label":0}\n',
            encoding="utf-8",
        )
        rows = _load_benchmark_jsonl(f)
        assert len(rows) == 2


def _make_jsonl_file(path, n_pos=60, n_neg=60):
    rows = []
    for i in range(n_pos):
        rows.append({"premise": f"Fact {i}.", "hypothesis": f"Claim {i}.", "label": 1})
    for i in range(n_neg):
        rows.append(
            {"premise": f"Source {i}.", "hypothesis": f"Wrong {i}.", "label": 0},
        )
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


class TestTrainingWorkerDirect:
    """Test _run_training_worker synchronously with mocked finetune_nli."""

    @patch("director_ai.core.finetune.finetune_nli")
    def test_worker_completes(self, mock_ft, tmp_path):
        from director_ai.core.finetune import FinetuneResult

        mock_ft.return_value = FinetuneResult(
            output_dir=str(tmp_path / "models" / "test-ok"),
            epochs_completed=1,
            train_samples=108,
            eval_samples=12,
            best_balanced_accuracy=0.85,
            final_loss=0.3,
            eval_metrics={"eval_balanced_accuracy": 0.85},
        )

        data_path = tmp_path / "upload.jsonl"
        _make_jsonl_file(data_path)

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        job = FinetuneJob(
            job_id="test-ok",
            config={
                "epochs": 1,
                "batch_size": 4,
                "auto_benchmark": False,
                "auto_onnx_export": False,
            },
        )

        _run_training_worker(job, data_path, models_dir)

        assert job.state == "completed"
        assert job.progress == 1.0
        assert job.model_path == str(tmp_path / "models" / "test-ok")
        assert job.completed_at > 0
        assert not data_path.exists()

    @patch(
        "director_ai.core.finetune.finetune_nli",
        side_effect=ValueError("No valid samples"),
    )
    def test_worker_handles_training_error(self, mock_ft, tmp_path):
        data_path = tmp_path / "data.jsonl"
        _make_jsonl_file(data_path, 10, 10)

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        job = FinetuneJob(
            job_id="test-fail",
            config={"epochs": 1, "batch_size": 4},
        )

        _run_training_worker(job, data_path, models_dir)

        assert job.state == "failed"
        assert "No valid samples" in job.error
        assert not data_path.exists()

    @patch("director_ai.core.finetune.finetune_nli")
    def test_worker_splits_data_90_10(self, mock_ft, tmp_path):
        from director_ai.core.finetune import FinetuneResult

        mock_ft.return_value = FinetuneResult(output_dir=str(tmp_path / "m" / "j1"))

        data_path = tmp_path / "data.jsonl"
        _make_jsonl_file(data_path, 50, 50)

        models_dir = tmp_path / "m"
        models_dir.mkdir()

        job = FinetuneJob(job_id="j1", config={"epochs": 1, "batch_size": 4})
        _run_training_worker(job, data_path, models_dir)

        assert job.state == "completed"
        call_args = mock_ft.call_args
        train_path = call_args[0][0]
        eval_path = call_args[1]["eval_path"]
        assert train_path.endswith("_train.jsonl")
        assert eval_path.endswith("_eval.jsonl")

    @patch(
        "director_ai.core.finetune.finetune_nli",
        side_effect=RuntimeError("GPU OOM"),
    )
    def test_worker_cleans_up_on_exception(self, mock_ft, tmp_path):
        data_path = tmp_path / "data.jsonl"
        _make_jsonl_file(data_path)

        models_dir = tmp_path / "m"
        models_dir.mkdir()

        job = FinetuneJob(job_id="j2", config={"epochs": 1, "batch_size": 4})
        _run_training_worker(job, data_path, models_dir)

        assert job.state == "failed"
        assert "GPU OOM" in job.error
        assert not data_path.exists()
