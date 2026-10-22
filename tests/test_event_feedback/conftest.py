# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Event feedback protocol tests
"""Provide declared protocol controls, never model-execution or memory-admission proof."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TypeAlias

import pytest

from director_ai.agentic.event_feedback.codec import Object, canonical, decode
from director_ai.agentic.event_feedback.contract import payload_digest
from director_ai.agentic.event_feedback.custody import EvidenceRoot
from director_ai.agentic.event_feedback.originals import Originals
from director_ai.agentic.event_feedback.transport import HttpEndpoint

Headers: TypeAlias = list[tuple[str, str]]


@dataclass
class ProtocolControl:
    """Record actual loopback HTTP requests and return declared protocol controls."""

    body: bytes = b"{}"
    status: int = 200
    headers: Headers = field(default_factory=list)
    disconnect: bool = False
    requests: list[tuple[str, str, bytes, str | None]] = field(default_factory=list)
    after_write_directory: Path | None = None
    collision_name: str = "response.raw"
    endpoint: HttpEndpoint | None = None
    routes: dict[str, tuple[int, bytes, Headers]] = field(default_factory=dict)
    acknowledge_writes: bool = False
    receipt_collision: Path | None = None
    postwrite_relocate: Path | None = None
    omit_content_length: bool = False
    body_delay: float = 0.0


@contextmanager
def protocol_control(control: ProtocolControl) -> Iterator[HttpEndpoint]:
    """Run a real HTTP transport with explicitly controlled response bytes."""

    class Handler(BaseHTTPRequestHandler):
        """Serve controlled protocol responses without producing semantic evidence."""

        def do_GET(self) -> None:
            """Serve the selected read control over an actual socket."""
            self.do_POST()

        def do_POST(self) -> None:
            """Capture original request bytes before the selected response."""
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            control.requests.append(
                (self.command, self.path, raw, self.headers.get("Authorization"))
            )
            if control.after_write_directory is not None:
                for path in control.after_write_directory.glob("attempts/*"):
                    (path / control.collision_name).mkdir()
            if control.disconnect:
                self.close_connection = True
                return
            if self.command == "POST" and control.postwrite_relocate is not None:
                original = control.postwrite_relocate
                original.rename(original.with_name(original.name + "-moved"))
                original.mkdir(mode=0o700)
            if self.command == "POST" and control.receipt_collision is not None:
                control.receipt_collision.mkdir()
            status, body, headers = control.routes.get(
                self.path, (control.status, control.body, control.headers)
            )
            if (
                control.acknowledge_writes
                and self.command == "POST"
                and self.path.endswith("/correctness")
            ):
                status, body, headers = 200, canonical(acknowledged(decode(raw))), []
            self.send_response(status)
            for name, value in headers:
                self.send_header(name, value)
            if not control.omit_content_length:
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if control.body_delay:
                self.wfile.flush()
                threading.Event().wait(control.body_delay)
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, format: str, *args: object) -> None:
            """Keep request paths and credential-bearing headers out of logs."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    control.endpoint = HttpEndpoint(f"http://127.0.0.1:{server.server_port}")
    try:
        yield control.endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def request_body() -> Object:
    """Return a declared schema example without claiming external evidence."""
    return {
        "original_query": "Does this exact source state the selected sentence?",
        "label_target": "policy_answer_support",
        "was_correct": True,
        "verifier_by": "verifier",
        "feedback_submission_id": "submission",
        "evidence_ref": {
            "kind": "policy_answer_support",
            "evidence_id": "review",
            "evidence_sha256": "a" * 64,
            "adjudicator_id": "verifier",
            "procedure_id": "source-grounded-agent-review",
            "procedure_version": "1",
            "supplied_snippet_sha256": ["b" * 64],
        },
    }


def acknowledged(request: Object) -> Object:
    """Bind a declared protocol receipt to every original request member."""
    result: Object = {
        "contract_version": "source-bound-recall-v3",
        "recall_event_id": "event",
        "payload_sha256": payload_digest("event", request),
        "evidence_sha256": request["evidence_ref"]["evidence_sha256"]
        if isinstance(request["evidence_ref"], dict)
        else None,
        "outcome_id": "outcome",
        "recording": "recorded",
        "calibration_eligible": False,
        "recorded_at": "2026-10-04T19:00:00+00:00",
    }
    result.update({k: v for k, v in request.items() if k != "evidence_ref"})
    return result


@pytest.fixture
def originals() -> Originals:
    """Supply source-bound protocol data for structural producer tests."""
    raw = b"Original sources are retained."
    sha = hashlib.sha256(raw).hexdigest()
    snapshot: Object = {
        "source_id": "source",
        "source_version": "version",
        "content_sha256": sha,
    }
    response: Object = {"results": [{"snippet": raw.decode()}]}
    provenance: Object = {
        "producer_id": "producer",
        "actor_type": "ai_project",
        "principal": {
            "principal_id": "verifier",
            "tenant_id": "tenant",
            "project_id": "project",
        },
        "adjudicator_id": "verifier",
        "procedure_id": "source-grounded-agent-review",
        "procedure_version": "1",
        "provider": "local/llama.cpp",
        "model": "qwen-model",
    }
    event: Object = {
        "kind": "served_recall_v3",
        "contract_version": "source-bound-recall-v3",
        "recall_event_id": "event",
        "original_query": "Question",
        "querying_by": "reader",
        "found": True,
        "recorded_at": "2026-10-04T19:00:00Z",
        "source_snapshot_status": "unqualified",
        "principal": {
            "principal_id": "verifier",
            "tenant_id": "tenant",
            "project_id": "project",
            "querying_by": "reader",
            "verifier_by": "verifier",
            "source_ids": ["source"],
        },
        "served_response": response,
    }
    inputs: Object = {
        "event_encoding": "sorted-compact-utf8-json-v1",
        "response_encoding": "sorted-compact-utf8-json-v1",
        "recall_event_id": "event",
        "original_query": "Question",
        "served_event_sha256": hashlib.sha256(canonical(event)).hexdigest(),
        "served_response_sha256": hashlib.sha256(canonical(response)).hexdigest(),
        "candidate_id": "candidate",
        "candidate_text": raw.decode(),
        "source_snapshots": [snapshot],
        "supplied_snippets": [
            {
                "source_snapshot": snapshot,
                "source_start": 0,
                "source_end": len(raw),
                "text": raw.decode(),
                "sha256": sha,
                "served": {
                    "kind": "result",
                    "item": 0,
                    "field": "snippet",
                    "start": 0,
                    "end": len(raw),
                },
            }
        ],
    }
    return Originals(inputs, provenance, event, raw)


def provider_response(verdict: str = "SUPPORTED") -> bytes:
    """Create a declared response control, not a genuine model judgement."""
    decision: Object = {
        "decision": verdict,
        "reason_code": "protocol_control",
        "claim_checks": [
            {
                "claim": "Source retained",
                "support": verdict,
                "snippet_indexes": [0] if verdict == "SUPPORTED" else [],
                "explanation": "Structural protocol control only",
            }
        ],
    }
    return canonical(
        {
            "model": "qwen-model",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": canonical(decision).decode()},
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }
    )


def registered_root(path: Path) -> EvidenceRoot:
    """Provision one actual private operator fixture and retain its inode identity."""
    path.mkdir(mode=0o700, exist_ok=True)
    info = path.stat()
    return EvidenceRoot(path, info.st_dev, info.st_ino)
