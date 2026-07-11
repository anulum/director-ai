# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — fine-tuning schema contract tests

"""Contract tests for the fine-tuning API request/response schemas."""

from __future__ import annotations

import pytest

import director_ai._finetune_schemas as schemas_module
import director_ai.finetune_api as finetune_api_module

pytestmark = pytest.mark.skipif(
    not schemas_module._FASTAPI_AVAILABLE,
    reason="pydantic not installed",
)


def test_facade_reexports_schema_classes():
    for name in ("ValidateRequest", "StartRequest", "JobStatus", "ModelInfo"):
        assert getattr(finetune_api_module, name) is getattr(schemas_module, name)


def test_validate_request_defaults_and_bounds():
    req = schemas_module.ValidateRequest()
    assert (req.epochs, req.batch_size) == (3, 16)
    with pytest.raises(ValueError):
        schemas_module.ValidateRequest(epochs=0)
    with pytest.raises(ValueError):
        schemas_module.ValidateRequest(batch_size=129)


def test_start_request_defaults_pin_the_local_training_contract():
    req = schemas_module.StartRequest()
    assert req.base_model == "factcg-deberta-v3-large"
    assert req.allow_experimental_model is False
    assert req.learning_rate == pytest.approx(2e-5)
    assert req.auto_benchmark is True
    assert req.auto_onnx_export is False
    with pytest.raises(ValueError):
        schemas_module.StartRequest(general_data_ratio=0.6)


def test_managed_lookup_request_requires_job_id():
    with pytest.raises(ValueError):
        schemas_module.ManagedTrainingLookupRequest(job_id="")
    req = schemas_module.ManagedTrainingLookupRequest(job_id="job-1")
    assert (req.backend, req.job_id) == ("vertex", "job-1")


def test_managed_training_request_bounds():
    kwargs = {
        "dataset_uri": "gs://bucket/data.jsonl",
        "output_uri": "gs://bucket/out",
        "container_image_uri": "gcr.io/x/y:z",
    }
    req = schemas_module.ManagedTrainingRequest(**kwargs)
    assert req.dry_run is True
    assert req.region == "us-central1"
    with pytest.raises(ValueError):
        schemas_module.ManagedTrainingRequest(**kwargs, accelerator_count=9)
    with pytest.raises(ValueError):
        schemas_module.ManagedTrainingRequest(**kwargs, timeout_minutes=0)
