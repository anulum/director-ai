# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Fine-tuning Job Store Persistence Tests

"""Real-surface tests for the persistent fine-tuning job stores (BUG-2).

Exercises the SQLite write-through path: restart survival, interrupted-job
recovery, cross-instance (multi-worker) visibility, the cross-process
concurrency cap, tenant isolation of the managed ledger, and the documented
in-memory fallbacks.
"""

from __future__ import annotations

import time

import pytest

from director_ai.finetune_jobs import (
    _MAX_CONCURRENT_JOBS,
    FinetuneJob,
    ManagedTrainingRecord,
    _JobStore,
    _ManagedJobStore,
)


def _db(tmp_path):
    return tmp_path / "finetune_jobs.sqlite3"


def _managed_record(job_id: str, tenant: str = "acme") -> ManagedTrainingRecord:
    return ManagedTrainingRecord(
        job_id=job_id,
        backend="vertex",
        state="submitted",
        tenant_id=tenant,
        dry_run=True,
        submitted_at=time.time(),
        display_name="test-run",
        output_uri="gs://bucket/out",
    )


class TestJobStorePersistence:
    def test_completed_job_survives_restart(self, tmp_path):
        store = _JobStore(_db(tmp_path))
        job = store.create({"epochs": 3})
        job.state = "completed"
        job.model_path = "/models/x"
        job.metrics = {"balanced_accuracy": 0.91}
        store.save(job)

        reopened = _JobStore(_db(tmp_path))  # simulated restart
        loaded = reopened.get(job.job_id)

        assert loaded is not None
        assert loaded.state == "completed"
        assert loaded.model_path == "/models/x"
        assert loaded.metrics == {"balanced_accuracy": 0.91}

    def test_activation_designation_survives_restart(self, tmp_path):
        store = _JobStore(_db(tmp_path))
        job = store.create({})
        job.state = "completed"
        job.activated = True
        store.save(job)

        reopened = _JobStore(_db(tmp_path))
        loaded = reopened.get(job.job_id)

        assert loaded is not None
        assert loaded.activated is True

    def test_interrupted_job_fails_closed_on_restart(self, tmp_path):
        store = _JobStore(_db(tmp_path))
        job = store.create({})
        job.state = "training"
        store.save(job)

        reopened = _JobStore(_db(tmp_path))  # its worker thread is gone
        loaded = reopened.get(job.job_id)

        assert loaded is not None
        assert loaded.state == "failed"
        assert "interrupted by restart" in loaded.error

    def test_recovered_jobs_free_the_concurrency_cap(self, tmp_path):
        store = _JobStore(_db(tmp_path))
        for _ in range(_MAX_CONCURRENT_JOBS):
            job = store.create({})
            job.state = "training"
            store.save(job)
        with pytest.raises(ValueError, match="Too many concurrent jobs"):
            store.create({})

        reopened = _JobStore(_db(tmp_path))
        # The ghost "training" jobs were failed on open, so capacity is back.
        assert reopened.create({}).state == "pending"

    def test_cross_instance_visibility(self, tmp_path):
        # Two live stores on one database = two server workers.
        worker_a = _JobStore(_db(tmp_path))
        worker_b = _JobStore(_db(tmp_path))
        job = worker_a.create({"epochs": 1})

        seen = worker_b.get(job.job_id)
        assert seen is not None
        assert seen.job_id == job.job_id
        assert any(j.job_id == job.job_id for j in worker_b.list_all())

    def test_concurrency_cap_counts_across_instances(self, tmp_path):
        worker_a = _JobStore(_db(tmp_path))
        worker_b = _JobStore(_db(tmp_path))
        for _ in range(_MAX_CONCURRENT_JOBS):
            job = worker_a.create({})
            job.state = "training"
            worker_a.save(job)

        with pytest.raises(ValueError, match="Too many concurrent jobs"):
            worker_b.create({})

    def test_delete_removes_the_persisted_row(self, tmp_path):
        store = _JobStore(_db(tmp_path))
        job = store.create({})
        store.save(job)

        assert store.delete(job.job_id) is True
        assert store.delete(job.job_id) is False
        reopened = _JobStore(_db(tmp_path))
        assert reopened.get(job.job_id) is None

    def test_delete_of_another_workers_job(self, tmp_path):
        worker_a = _JobStore(_db(tmp_path))
        worker_b = _JobStore(_db(tmp_path))
        job = worker_a.create({})

        # B never held the object in memory; the row alone must satisfy delete.
        assert worker_b.delete(job.job_id) is True
        reopened = _JobStore(_db(tmp_path))
        assert reopened.get(job.job_id) is None


class TestJobStoreFallbacks:
    def test_no_path_is_ephemeral(self):
        store = _JobStore()
        assert store.persistent is False
        job = store.create({"epochs": 3})
        assert store.get(job.job_id) is job
        assert store.delete(job.job_id) is True

    def test_unwritable_path_falls_back_to_memory(self, tmp_path):
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("occupied")
        store = _JobStore(blocker / "jobs.sqlite3")  # cannot be created

        assert store.persistent is False
        job = store.create({})
        assert store.get(job.job_id) is job

    def test_ephemeral_jobs_do_not_survive_a_new_store(self):
        first = _JobStore()
        job = first.create({})
        second = _JobStore()
        assert second.get(job.job_id) is None


class TestManagedJobStorePersistence:
    def test_record_survives_restart_and_keeps_tenant_isolation(self, tmp_path):
        store = _ManagedJobStore(_db(tmp_path))
        store.add(_managed_record("mj-1", tenant="acme"))

        reopened = _ManagedJobStore(_db(tmp_path))
        assert reopened.get("acme", "mj-1") is not None
        assert reopened.get("other", "mj-1") is None

    def test_update_state_is_written_through(self, tmp_path):
        store = _ManagedJobStore(_db(tmp_path))
        store.add(_managed_record("mj-2"))
        updated = store.update_state("acme", "mj-2", "running", error="")
        assert updated is not None

        reopened = _ManagedJobStore(_db(tmp_path))
        loaded = reopened.get("acme", "mj-2")
        assert loaded is not None
        assert loaded.state == "running"

    def test_update_state_rejects_foreign_tenant(self, tmp_path):
        store = _ManagedJobStore(_db(tmp_path))
        store.add(_managed_record("mj-3", tenant="acme"))
        assert store.update_state("mallory", "mj-3", "cancelled") is None

        reopened = _ManagedJobStore(_db(tmp_path))
        loaded = reopened.get("acme", "mj-3")
        assert loaded is not None
        assert loaded.state == "submitted"

    def test_list_for_tenant_merges_database_and_live_records(self, tmp_path):
        seeder = _ManagedJobStore(_db(tmp_path))
        seeder.add(_managed_record("mj-old", tenant="acme"))

        live = _ManagedJobStore(_db(tmp_path))
        live.add(_managed_record("mj-new", tenant="acme"))
        live.add(_managed_record("mj-foreign", tenant="other"))

        ids = {r.job_id for r in live.list_for_tenant("acme")}
        assert ids == {"mj-old", "mj-new"}

    def test_no_path_is_ephemeral(self):
        store = _ManagedJobStore()
        assert store.persistent is False
        store.add(_managed_record("mj-eph"))
        assert store.get("acme", "mj-eph") is not None
        assert _ManagedJobStore().get("acme", "mj-eph") is None


class TestRouterPersistenceWiring:
    def test_router_serves_jobs_persisted_before_it_started(self, tmp_path):
        # Real API surface: a completed job written by a previous process is
        # visible through a freshly created router (restart survival, BUG-2).
        pytest.importorskip("fastapi", reason="fastapi required for router test")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from director_ai.finetune_api import _JOB_DB_FILENAME, create_finetune_router

        previous = _JobStore(tmp_path / _JOB_DB_FILENAME)
        job = previous.create({"epochs": 2})
        job.state = "completed"
        job.model_path = str(tmp_path / job.job_id)
        job.activated = True
        previous.save(job)

        app = FastAPI()
        app.include_router(
            create_finetune_router(models_dir=tmp_path), prefix="/v1/finetune"
        )
        client = TestClient(app)

        status = client.get(f"/v1/finetune/{job.job_id}")
        assert status.status_code == 200
        assert status.json()["state"] == "completed"

        models = client.get("/v1/finetune/").json()["models"]
        mine = [m for m in models if m["job_id"] == job.job_id]
        assert len(mine) == 1
        assert mine[0]["activated"] is True

    def test_activation_via_api_survives_a_router_restart(self, tmp_path):
        pytest.importorskip("fastapi", reason="fastapi required for router test")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from director_ai.finetune_api import _JOB_DB_FILENAME, create_finetune_router

        seeder = _JobStore(tmp_path / _JOB_DB_FILENAME)
        job = seeder.create({})
        job.state = "completed"
        job.model_path = str(tmp_path / job.job_id)
        seeder.save(job)

        def _client() -> TestClient:
            app = FastAPI()
            app.include_router(
                create_finetune_router(models_dir=tmp_path), prefix="/v1/finetune"
            )
            return TestClient(app)

        first = _client()
        response = first.post(f"/v1/finetune/{job.job_id}/activate")
        assert response.status_code == 200
        assert response.json()["activated"] is True

        second = _client()  # simulated restart: new router, same models_dir
        models = second.get("/v1/finetune/").json()["models"]
        mine = [m for m in models if m["job_id"] == job.job_id]
        assert mine and mine[0]["activated"] is True
        # Deletion protection derives from the persisted designation.
        assert second.delete(f"/v1/finetune/{job.job_id}").status_code == 409


class TestJobRecordShapes:
    def test_finetune_job_round_trips_all_fields(self, tmp_path):
        store = _JobStore(_db(tmp_path))
        job = store.create({"batch_size": 8})
        job.state = "completed"
        job.progress = 1.0
        job.current_step = 10
        job.total_steps = 10
        job.validation_report = {"total_samples": 100}
        job.metrics = {"f1": 0.9}
        job.regression_report = {"passed": True}
        job.model_path = "/m"
        job.error = ""
        job.completed_at = 123.0
        store.save(job)

        loaded = _JobStore(_db(tmp_path)).get(job.job_id)
        assert loaded == job

    def test_job_ids_are_unique_hex(self, tmp_path):
        store = _JobStore(_db(tmp_path))
        ids = {store.create({}).job_id for _ in range(3)}
        assert len(ids) == 3
        assert all(len(i) == 32 for i in ids)


def test_reexports_through_finetune_api():
    """The historical import surface stays intact after the module split."""
    from director_ai import finetune_api

    assert finetune_api.FinetuneJob is FinetuneJob
    assert finetune_api.ManagedTrainingRecord is ManagedTrainingRecord
    assert finetune_api._JobStore is _JobStore
    assert finetune_api._ManagedJobStore is _ManagedJobStore
    assert finetune_api._MAX_CONCURRENT_JOBS == _MAX_CONCURRENT_JOBS
