# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — caller-facing refusal integration tests
"""Exercise refusal mappings through real HTTP routes and production backends."""

from __future__ import annotations

import io
import json
import sqlite3
import zipfile
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from director_ai.core.config import DirectorConfig
from director_ai.finetune_api import FinetuneJob
from director_ai.server import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Start the actual server with offline retrieval and multimodal backends."""
    monkeypatch.setenv("DIRECTOR_AI_ENABLE_EXPERIMENTAL_HOOKS", "1")
    config = DirectorConfig(
        mode="grounded",
        hybrid_retrieval=False,
        reranker_enabled=False,
        use_nli=False,
        scorer_backend="lite",
        tenant_routing=True,
        sanitize_inputs=False,
        knowledge_write_require_tenant_binding=False,
        finetune_models_dir=str(tmp_path / "models"),
        api_keys=["operator-test-key"],
        finetune_operator_api_keys=["operator-test-key"],
        multimodal_enabled_modalities=("image", "audio", "video"),
    )
    with TestClient(
        create_app(config),
        headers={"X-Tenant-ID": "acme", "X-API-Key": "operator-test-key"},
    ) as http:
        yield http


@pytest.mark.parametrize(
    ("filename", "content", "detail"),
    [
        ("broken.pdf", b"not a PDF", "invalid PDF document"),
        ("broken.docx", b"not a ZIP", "invalid DOCX document"),
    ],
)
def test_document_upload_retains_authored_refusals(
    client: TestClient,
    filename: str,
    content: bytes,
    detail: str,
) -> None:
    """Malformed documents receive deliberate fixed parser refusals over HTTP."""
    response = client.post(
        "/v1/knowledge/upload",
        files={"file": (filename, content, "application/octet-stream")},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": detail}


def test_document_upload_maps_native_archive_key_error(client: TestClient) -> None:
    """A structurally incomplete DOCX never returns the ZIP reader's key name."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as doc:
        doc.writestr("ordinary.txt", "Incomplete document")
    response = client.post(
        "/v1/knowledge/upload",
        files={"file": ("incomplete.docx", archive.getvalue(), "application/zip")},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid document"}
    assert "[Content_Types].xml" not in response.text
    assert "archive" not in response.text


def test_tenant_backend_maps_real_factory_value_error(client: TestClient) -> None:
    """The real tenant backend factory cannot echo invalid backend input."""
    response = client.post(
        "/v1/tenants/acme/vector-facts",
        json={
            "key": "policy",
            "value": "Refunds within 30 days",
            "backend_type": "unknown-input",
        },
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid backend_type"}
    assert "unknown-input" not in response.text


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ({"modality": "image"}, "Invalid multimodal request"),
        ({"modality": "audio"}, "audio modality requires transcript_text"),
        ({"modality": "video"}, "video modality requires frame_similarities"),
        (
            {"modality": "video", "frame_similarities": [1.1]},
            "score must be finite and in [0, 1]",
        ),
    ],
)
def test_multimodal_real_backends_distinguish_native_and_authored_errors(
    client: TestClient,
    body: dict[str, object],
    detail: str,
) -> None:
    """Production adapters preserve explicit refusals and replace native errors."""
    response = client.post(
        "/v1/multimodal/check",
        json={"claim_text": "A red car", "media_ref": "media:car", **body},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": detail}
    assert "image_bytes" not in response.text


def test_batch_malformed_request_keeps_framework_validation(client: TestClient) -> None:
    """The intentional Pydantic 422 contract remains available for typed inputs."""
    response = client.post("/v1/batch", json={"prompts": [None]})
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "string_type"
    mismatch = client.post(
        "/v1/batch",
        json={"task": "review", "prompts": ["Hello"], "responses": []},
    )
    assert mismatch.status_code == 422
    assert mismatch.json() == {
        "detail": "review requires equal prompts (1) and responses (0)"
    }


def test_managed_training_retains_refusal_and_maps_real_backend_failure(
    client: TestClient,
) -> None:
    """A live portable submission fails safely without provisioning compute."""
    body = {
        "backend": "portable",
        "dry_run": False,
        "dataset_uri": "s3://tests/train",
        "output_uri": "s3://tests/output",
        "container_image_uri": "ghcr.io/anulum/director-ai-trainer:3.21.0",
    }
    response = client.post("/v1/finetune/managed/submit", json=body)
    assert response.status_code == 502
    assert response.json() == {"detail": "Training backend submission failed"}
    assert "orchestrator" not in response.text
    refusal = client.post(
        "/v1/finetune/managed/submit",
        json={**body, "dataset_uri": ""},
    )
    assert refusal.status_code == 422
    assert refusal.json() == {"detail": "dataset_uri is required"}


def test_benchmark_malformed_dataset_maps_native_attribute_error(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """Actual malformed JSONL produces a fixed per-model failure sentence."""
    job_id = "a" * 32
    model = tmp_path / "models" / "candidate"
    model.mkdir()
    job = FinetuneJob(
        job_id=job_id,
        state="completed",
        config={"base_model": "factcg-deberta-v3-large"},
        model_path=str(model),
    )
    with sqlite3.connect(tmp_path / "models" / "finetune_jobs.sqlite3") as db:
        db.execute(
            "INSERT INTO finetune_jobs VALUES (?, ?, ?)",
            (job_id, job.state, json.dumps(asdict(job))),
        )
    dataset_dir = tmp_path / "models" / "_benchmark_datasets"
    dataset_dir.mkdir()
    dataset_id = "b" * 32
    dataset = dataset_dir / f"{dataset_id}.jsonl"
    dataset.write_text(json.dumps(["not a training row"]) + "\n", encoding="utf-8")
    response = client.post(
        "/v1/finetune/managed/benchmark-models",
        json={
            "model_jobs": {"factcg-deberta-v3-large": job_id},
            "general_dataset_id": dataset_id,
        },
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["error"] == "Model benchmark failed"
    assert result["recommendation"] == "reject"
    assert "attribute" not in response.text
    assert "get" not in result["error"]
    refusal = client.post(
        "/v1/finetune/managed/benchmark-models",
        json={"model_jobs": {}, "general_dataset_id": dataset_id},
    )
    assert refusal.status_code == 422
    assert refusal.json() == {
        "detail": "Benchmark requires between one and eight local jobs"
    }


def test_local_training_cap_preserves_authored_refusal_and_cleans_upload(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """The actual persistent job cap refuses a valid upload without a worker."""
    with sqlite3.connect(tmp_path / "models" / "finetune_jobs.sqlite3") as db:
        for index in range(4):
            job = FinetuneJob(job_id=f"occupied-{index}", state="training")
            db.execute(
                "INSERT INTO finetune_jobs (job_id, state, payload) VALUES (?, ?, ?)",
                (job.job_id, job.state, json.dumps(asdict(job))),
            )
    rows = [
        {
            "premise": f"Evidence {index}",
            "hypothesis": f"Claim {index}",
            "label": index % 2,
        }
        for index in range(500)
    ]
    body = "\n".join(json.dumps(row) for row in rows).encode()
    response = client.post(
        "/v1/finetune/start",
        files={"file": ("valid.jsonl", body, "application/x-ndjson")},
    )
    assert response.status_code == 429
    assert response.json() == {"detail": "Too many concurrent jobs (4/4)"}
    assert list((tmp_path / "models" / "_uploads").iterdir()) == []


def test_batch_preserves_real_execution_bound_refusal(client: TestClient) -> None:
    """The actual batch processor keeps its authored execution-bound refusal."""
    from director_ai.core.runtime.batch import BatchProcessor

    assert isinstance(client.app, FastAPI)
    processor = client.app.state._state["batch"]
    assert isinstance(processor, BatchProcessor)
    processor.max_concurrency = 0
    response = client.post(
        "/v1/batch", json={"task": "process", "prompts": ["ordinary prompt"]}
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "max_concurrency must be >= 1, got 0"}


def test_document_upload_maps_malformed_docx_xml(client: TestClient) -> None:
    """The real DOCX XML parser keeps malformed content in the 422 contract."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as doc:
        doc.writestr("[Content_Types].xml", b"ordinary invalid XML")
    response = client.post(
        "/v1/knowledge/upload",
        files={"file": ("malformed.docx", archive.getvalue(), "application/zip")},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid DOCX document"}
    assert "Start tag" not in response.text
    assert "line 1" not in response.text


@pytest.mark.parametrize("text", [b"", b"Ordinary PDF document"])
def test_document_upload_reads_actual_pdf_pages(
    client: TestClient,
    text: bytes,
) -> None:
    """Actual PDF pages extract text and preserve the authored empty refusal."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    if text:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 10 10 Td (" + text + b") Tj ET")
        page.replace_contents(stream.flate_encode())
    document = io.BytesIO()
    writer.write(document)
    response = client.post(
        "/v1/knowledge/upload",
        files={"file": ("ordinary.pdf", document.getvalue(), "application/pdf")},
    )
    if text:
        assert response.status_code == 201
        assert response.json()["chunk_count"] == 1
    else:
        assert response.status_code == 422
        assert response.json() == {"detail": "Parsed file contains no text"}


def test_document_upload_refuses_malformed_pdf_page_content(client: TestClient) -> None:
    """The real PDF content reader maps an incomplete text string to 422."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT (incomplete text string")
    page.replace_contents(stream.flate_encode())
    document = io.BytesIO()
    writer.write(document)
    response = client.post(
        "/v1/knowledge/upload",
        files={"file": ("malformed-page.pdf", document.getvalue(), "application/pdf")},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid PDF document"}
    assert "ended" not in response.text
    assert "stream" not in response.text


def test_document_upload_requires_a_registry_tenant(client: TestClient) -> None:
    """An unbound upload receives a fixed refusal before document registration."""
    response = client.post(
        "/v1/knowledge/upload",
        headers={"X-Tenant-ID": ""},
        files={"file": ("ordinary.txt", b"Ordinary document", "text/plain")},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Document uploads require a tenant"}
