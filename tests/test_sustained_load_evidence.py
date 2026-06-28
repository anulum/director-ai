# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — sustained load evidence tests

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

from pytest import MonkeyPatch

from benchmarks import sustained_load_evidence as evidence
from director_ai.core.runtime.async_streaming import AsyncStreamingKernel
from director_ai.core.runtime.streaming import TokenEvent


def test_percentile_handles_empty_and_singleton_inputs() -> None:
    """Verify sustained-load percentile evidence handles small samples."""

    assert evidence._percentile([], 0.5) == 0.0
    assert evidence._percentile([7.0], 0.95) == 7.0


def test_git_commit_falls_back_when_git_is_unavailable(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify sustained-load evidence handles missing and failing git clients."""

    module = cast(Any, evidence)
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)
    assert module._git_commit() == "unknown"

    monkeypatch.setattr(module.shutil, "which", lambda _name: "git")

    def raise_subprocess(*_args: object, **_kwargs: object) -> None:
        raise module.subprocess.SubprocessError()

    monkeypatch.setattr(module.subprocess, "run", raise_subprocess)
    assert module._git_commit() == "unknown"

    class Completed:
        stdout = "abc123\n"

    def complete_subprocess(*_args: object, **_kwargs: object) -> Completed:
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", complete_subprocess)
    assert module._git_commit() == "abc123"


def test_run_one_stream_records_foreign_stream_contamination() -> None:
    """Verify stream contamination telemetry records foreign prefixes."""

    class ContaminatingKernel:
        async def stream_tokens(
            self,
            _token_source: Callable[[], AsyncIterator[str]],
            score: Callable[[str], Any],
        ) -> AsyncIterator[TokenEvent]:
            await score("stream=1;foreign-token")
            yield TokenEvent(
                token="stream=0;token=0000;",
                index=0,
                coherence=0.95,
                timestamp=0.0,
            )

    result = asyncio.run(
        evidence._run_one_stream(
            cast(AsyncStreamingKernel, ContaminatingKernel()),
            0,
            stream_count=2,
            tokens_per_stream=1,
        )
    )

    assert result["contamination_count"] == 1
    assert result["contamination_samples"] == ["stream=1;foreign-token"]


def test_async_ordering_probe_reports_clean_concurrent_streams() -> None:
    """Verify async load evidence preserves stream ordering under concurrency."""

    packet = evidence.run_async_ordering_probe(streams=4, tokens_per_stream=8)

    assert packet["passed"] is True
    assert packet["streams"] == 4
    assert packet["tokens_per_stream"] == 8
    assert packet["total_events"] == 32
    assert packet["failed_streams"] == 0
    assert packet["events_per_second"] > 0


def test_async_ordering_probe_validates_positive_dimensions() -> None:
    """Verify sustained async evidence rejects invalid dimensions."""

    for kwargs in (
        {"streams": 0, "tokens_per_stream": 1},
        {"streams": 1, "tokens_per_stream": 0},
    ):
        try:
            evidence.run_async_ordering_probe(**kwargs)
        except ValueError as exc:
            assert ">=" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


def test_tenant_poisoning_probe_blocks_same_key_cross_tenant_contamination() -> None:
    """Verify tenant poisoning evidence blocks cross-tenant contamination."""

    packet = evidence.run_tenant_poisoning_probe(cases=4)

    assert packet["passed"] is True
    assert packet["cases"] == 4
    assert packet["writes"] == 8
    assert packet["queries"] == 12
    assert packet["failed_cases"] == 0


def test_tenant_poisoning_probe_validates_case_count() -> None:
    """Verify tenant poisoning evidence rejects empty case sets."""

    try:
        evidence.run_tenant_poisoning_probe(cases=0)
    except ValueError as exc:
        assert "cases" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_tenant_poisoning_probe_records_failure_samples(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify tenant poisoning evidence records failing isolation samples."""

    class Backend:
        def count(self) -> int:
            return 1

    class Chunk:
        text = "Tenant B leaked payload"

    class LeakyStore:
        backend = Backend()

        def __init__(self, *, backend: object) -> None:
            del backend

        def add_fact(self, _key: str, _text: str, *, tenant_id: str) -> None:
            del tenant_id

        def retrieve_context(
            self,
            _query: str,
            *,
            top_k: int,
            tenant_id: str,
        ) -> str:
            del top_k, tenant_id
            return ""

        def retrieve_context_with_chunks(
            self,
            _query: str,
            *,
            top_k: int,
            tenant_id: str,
        ) -> list[Chunk]:
            del top_k, tenant_id
            return [Chunk()]

    module = cast(Any, evidence)
    monkeypatch.setattr(module, "VectorGroundTruthStore", LeakyStore)

    packet = evidence.run_tenant_poisoning_probe(cases=1)

    assert packet["passed"] is False
    assert packet["failed_cases"] == 1
    assert packet["failure_samples"][0]["chunk_a"] == "Tenant B leaked payload"


def test_tenant_poisoning_probe_default_scale_is_not_rank_saturation() -> None:
    """Verify the default tenant poisoning scale is not rank saturated."""

    packet = evidence.run_tenant_poisoning_probe()

    assert packet["passed"] is True
    assert packet["cases"] == 64
    assert packet["failed_cases"] == 0


def test_sustained_load_evidence_payload_has_acceptance_summary(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify the R17 packet records acceptance checks and release limits."""

    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")

    packet = evidence.run_sustained_load_evidence(
        streams=2,
        tokens_per_stream=4,
        tenant_cases=2,
    )

    assert packet["benchmark"] == "sustained_load_evidence"
    assert packet["git_commit"] == "abc123"
    assert packet["acceptance"] == {
        "passed": True,
        "async_ordering": True,
        "tenant_poisoning": True,
        "limits": {
            "local_only": True,
            "staging_or_production_telemetry_included": False,
            "external_operator_signoff_included": False,
        },
    }
    assert packet["probes"]["async_ordering"]["total_events"] == 8
    assert packet["probes"]["tenant_poisoning"]["cases"] == 2


def test_main_writes_requested_output_path(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify the R17 CLI writes the requested evidence artifact."""

    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")
    output = tmp_path / "evidence.json"

    exit_code = evidence.main(
        [
            "--streams",
            "2",
            "--tokens-per-stream",
            "4",
            "--tenant-cases",
            "2",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["acceptance"]["passed"] is True


def test_main_uses_default_results_path(monkeypatch: MonkeyPatch) -> None:
    """Verify the R17 CLI saves to the default benchmark results path."""

    saved: list[str] = []

    def save_results(payload: object, filename: str) -> None:
        saved.append(filename)

    monkeypatch.setattr(evidence, "save_results", save_results)
    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")

    assert (
        evidence.main(
            [
                "--streams",
                "2",
                "--tokens-per-stream",
                "4",
                "--tenant-cases",
                "2",
            ]
        )
        == 0
    )
    assert len(saved) == 1
    assert saved[0].startswith("sustained_load_evidence_")
