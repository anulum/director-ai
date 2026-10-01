# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Managed benchmark artifact access

"""Resolve operator-selected job and dataset IDs to service-owned artifacts."""

from __future__ import annotations

import hmac
import json
import logging
import re
import uuid
from collections.abc import Callable, Coroutine
from pathlib import Path

from fastapi import HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from .finetune_jobs import _JobStore
from .server_support import _extract_request_api_key

logger = logging.getLogger("DirectorAI.FinetuneAPI")

_ID = re.compile(r"[0-9a-f]{32}")
_MAX_DATASET_BYTES = 10 * 1024 * 1024


class ManagedRoute(APIRoute):
    """Keep malformed managed requests free of echoed input and native errors."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[None, None, Response]]:
        """Map request validation and unexpected faults to fixed HTTP failures.

        Returns
        -------
        Callable
            Asynchronous handler retaining deliberate HTTP refusals.
        """
        handler = super().get_route_handler()

        async def safe_handler(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as exc:
                raise HTTPException(422, "Invalid managed training request") from exc
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("Managed training request failed")
                raise HTTPException(500, "Managed training request failed") from exc

        return safe_handler


class BenchmarkArtifactAccess:
    """Confine benchmark inputs to local completed jobs and uploaded datasets.

    Parameters
    ----------
    models_dir : Path
        Server-controlled model root, also containing the dataset directory.
    jobs : _JobStore
        Persistent local training job ledger.
    operator_keys : tuple of str
        Existing API keys explicitly allowed to operate managed routes.
        An empty tuple disables those routes, including standalone routers.
    """

    def __init__(
        self, models_dir: Path, jobs: _JobStore, operator_keys: tuple[str, ...]
    ) -> None:
        self.root = models_dir.resolve()
        self.datasets = self.root / "_benchmark_datasets"
        self.jobs = jobs
        self.operator_keys = operator_keys

    def require_operator(self, request: Request) -> None:
        """Require an API key explicitly granted managed-training access.

        Parameters
        ----------
        request : Request
            HTTP request carrying the normal API-key authentication header.

        Raises
        ------
        HTTPException
            403 if the key is absent, unrecognised or operator access is disabled.
        """
        provided = _extract_request_api_key(request)
        allowed = False
        for key in self.operator_keys:
            allowed |= bool(key) and hmac.compare_digest(
                provided.encode(), key.encode()
            )
        if not allowed:
            raise HTTPException(403, "Managed training requires an operator API key")

    def model(self, alias: str, job_id: str) -> Path:
        """Resolve a completed local job artifact, refusing foreign sources.

        Parameters
        ----------
        alias : str
            Registry alias matching the job's recorded base model.
        job_id : str
            Opaque identifier of a completed local training job.

        Returns
        -------
        Path
            Existing model directory confined to the server model root.

        Raises
        ------
        HTTPException
            422 for unknown, incomplete, mismatched or unconfined artifacts.
        """
        job = self.jobs.get(job_id) if _ID.fullmatch(job_id) else None
        if (
            job is None
            or job.state != "completed"
            or not job.model_path
            or job.config.get("base_model") != alias
        ):
            raise HTTPException(
                422, "Model must identify a completed local training job"
            )
        try:
            artifact = Path(job.model_path).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise HTTPException(422, "Model artifact is unavailable") from exc
        if not artifact.is_relative_to(self.root) or not artifact.is_dir():
            raise HTTPException(
                422, "Model artifact is outside the training model root"
            )
        return artifact

    def dataset(self, dataset_id: str) -> Path:
        """Resolve an uploaded dataset ID without accepting filesystem paths.

        Parameters
        ----------
        dataset_id : str
            Opaque identifier returned by the dataset upload route.

        Returns
        -------
        Path
            Existing JSONL file confined to the model root.

        Raises
        ------
        HTTPException
            422 for invalid or unconfined identifiers; 404 for absent data.
        """
        if not _ID.fullmatch(dataset_id):
            raise HTTPException(422, "Invalid benchmark dataset ID")
        try:
            path = (self.datasets / f"{dataset_id}.jsonl").resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise HTTPException(404, "Benchmark dataset not found") from exc
        if not path.is_relative_to(self.root) or not path.is_file():
            raise HTTPException(
                422, "Benchmark dataset is outside the training model root"
            )
        return path

    async def upload(self, file: UploadFile) -> dict[str, str | int]:
        """Store at most 10 MiB of valid labelled JSONL.

        Parameters
        ----------
        file : UploadFile
            UTF-8 JSONL with nonempty premise/hypothesis strings and integer labels.

        Returns
        -------
        dict
            Opaque dataset identifier and number of validated samples.

        Raises
        ------
        HTTPException
            413 for excess bytes; 422 for invalid rows or an unconfined data root.
        OSError
            Storage failure, mapped to a fixed failure by the managed route.
        """
        data = bytearray()
        while chunk := await file.read(64 * 1024):
            data.extend(chunk)
            if len(data) > _MAX_DATASET_BYTES:
                raise HTTPException(413, "Benchmark dataset exceeds 10 MiB")
        try:
            rows = [
                json.loads(line)
                for line in data.decode("utf-8").splitlines()
                if line.strip()
            ]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(422, "Invalid benchmark JSONL dataset") from exc
        if not rows or any(
            not isinstance(row, dict)
            or not isinstance(row.get("premise"), str)
            or not row["premise"].strip()
            or not isinstance(row.get("hypothesis"), str)
            or not row["hypothesis"].strip()
            or type(row.get("label")) is not int
            or row["label"] not in (0, 1)
            for row in rows
        ):
            raise HTTPException(422, "Invalid benchmark JSONL dataset")
        self.datasets.mkdir(parents=True, exist_ok=True)
        directory = self.datasets.resolve(strict=True)
        if not directory.is_relative_to(self.root):
            raise HTTPException(
                422, "Benchmark dataset is outside the training model root"
            )
        dataset_id = uuid.uuid4().hex
        path = directory / f"{dataset_id}.jsonl"
        with path.open("xb") as stream:
            stream.write(data)
        return {"dataset_id": dataset_id, "samples": len(rows)}
