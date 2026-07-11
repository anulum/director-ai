# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — managed-training lane contract tests

"""Contract tests for the managed-training endpoint module."""

from __future__ import annotations

import importlib.util
import sys
import types
from unittest.mock import patch

import pytest

import director_ai._finetune_managed as managed_module
import director_ai.finetune_api as finetune_api_module
from director_ai.finetune_jobs import ManagedTrainingRecord


def test_module_import_without_fastapi_skips_route_models():
    fake_schemas = types.ModuleType("director_ai._finetune_schemas")
    fake_schemas._FASTAPI_AVAILABLE = False

    spec = importlib.util.spec_from_file_location(
        "director_ai._finetune_managed_no_fastapi",
        managed_module.__file__,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"fastapi": None, "director_ai._finetune_schemas": fake_schemas},
    ):
        spec.loader.exec_module(module)

    assert module._FASTAPI_AVAILABLE is False
    # The serialiser has no FastAPI dependency and must survive the degrade.
    assert callable(module._managed_record_to_dict)


pytestmark = pytest.mark.skipif(
    not managed_module._FASTAPI_AVAILABLE,
    reason="fastapi not installed",
)

_MANAGED_PATHS = {
    "/managed/submit",
    "/managed/jobs",
    "/managed/status",
    "/managed/cancel",
    "/managed/models",
    "/managed/benchmark-models",
}


def test_facade_reexports_managed_helpers():
    assert (
        finetune_api_module._managed_record_to_dict
        is managed_module._managed_record_to_dict
    )
    assert (
        finetune_api_module.register_managed_routes
        is managed_module.register_managed_routes
    )


def test_register_managed_routes_registers_the_full_lane():
    from fastapi import APIRouter

    router = APIRouter()
    managed_module.register_managed_routes(
        router,
        managed_store=object(),  # endpoints resolve the store lazily per request
        tenant_from_request=lambda request: "",
    )
    assert {route.path for route in router.routes} == _MANAGED_PATHS


def test_managed_record_round_trips_every_field():
    record = ManagedTrainingRecord(
        job_id="mj-1",
        backend="vertex",
        state="running",
        tenant_id="tenant-a",
        dry_run=False,
        submitted_at=123.5,
        display_name="run",
        output_uri="gs://bucket/out",
        console_uri="https://console.example/mj-1",
        error="",
    )
    payload = managed_module._managed_record_to_dict(record)
    assert payload == {
        "job_id": "mj-1",
        "backend": "vertex",
        "state": "running",
        "tenant_id": "tenant-a",
        "dry_run": False,
        "submitted_at": 123.5,
        "display_name": "run",
        "output_uri": "gs://bucket/out",
        "console_uri": "https://console.example/mj-1",
        "error": "",
    }
