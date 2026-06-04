# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Sustained load evidence packet

"""Generate sustained-load evidence for production deployment hardening.

The packet targets two production failure classes:

* async stream ordering under concurrent scheduling;
* cross-tenant knowledge poisoning with same-key adversarial facts.

It is intentionally dependency-light so operators can run it in CI, staging, or
an incident clone without model downloads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import shutil
import subprocess  # nosec B404
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks._common import save_results
from director_ai.core.async_streaming import AsyncStreamingKernel
from director_ai.core.streaming import TokenEvent
from director_ai.core.vector_store import InMemoryBackend, VectorGroundTruthStore


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _git_commit() -> str:
    git = shutil.which("git")
    if not git:
        return "unknown"
    try:
        completed = subprocess.run(  # nosec B603
            [git, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


async def _collect_events(
    kernel: AsyncStreamingKernel,
    token_source,
    score,
) -> list[TokenEvent]:
    events: list[TokenEvent] = []
    async for event in kernel.stream_tokens(token_source, score):
        events.append(event)
    return events


async def _run_one_stream(
    kernel: AsyncStreamingKernel,
    stream_id: int,
    *,
    stream_count: int,
    tokens_per_stream: int,
) -> dict[str, Any]:
    tokens = [
        f"stream={stream_id};token={idx:04d};" for idx in range(tokens_per_stream)
    ]
    foreign_prefixes = [
        f"stream={other};" for other in range(stream_count) if other != stream_id
    ]
    score_calls = 0
    contamination_samples: list[str] = []
    last_accumulated = ""

    async def token_source():
        for idx, token in enumerate(tokens):
            if idx % 8 == 0:
                await asyncio.sleep(0)
            yield token

    async def score(accumulated: str) -> float:
        nonlocal last_accumulated, score_calls
        await asyncio.sleep(0)
        last_accumulated = accumulated
        score_calls += 1
        if any(prefix in accumulated for prefix in foreign_prefixes):
            contamination_samples.append(accumulated[:200])
        return 0.95

    started = time.perf_counter()
    events = await _collect_events(kernel, token_source(), score)
    elapsed_ms = (time.perf_counter() - started) * 1000
    expected_indices = list(range(tokens_per_stream))
    event_indices = [event.index for event in events]
    event_tokens = [event.token for event in events]

    return {
        "stream_id": stream_id,
        "events": len(events),
        "score_calls": score_calls,
        "elapsed_ms": round(elapsed_ms, 3),
        "order_ok": event_tokens == tokens,
        "indices_ok": event_indices == expected_indices,
        "halted": any(event.halted for event in events),
        "accumulated_ok": last_accumulated == "".join(tokens),
        "contamination_count": len(contamination_samples),
        "contamination_samples": contamination_samples[:3],
    }


def run_async_ordering_probe(
    *,
    streams: int = 64,
    tokens_per_stream: int = 64,
) -> dict[str, Any]:
    """Return async stream ordering telemetry under concurrent scheduling."""
    if streams < 1:
        raise ValueError("streams must be >= 1")
    if tokens_per_stream < 1:
        raise ValueError("tokens_per_stream must be >= 1")

    kernel = AsyncStreamingKernel(
        hard_limit=0.1,
        window_size=tokens_per_stream + 1,
        trend_window=tokens_per_stream + 1,
    )

    async def _run() -> list[dict[str, Any]]:
        return await asyncio.gather(
            *(
                _run_one_stream(
                    kernel,
                    stream_id,
                    stream_count=streams,
                    tokens_per_stream=tokens_per_stream,
                )
                for stream_id in range(streams)
            )
        )

    started = time.perf_counter()
    stream_results = asyncio.run(_run())
    elapsed_s = time.perf_counter() - started
    total_events = sum(int(result["events"]) for result in stream_results)
    failures = [
        result
        for result in stream_results
        if not (
            result["order_ok"]
            and result["indices_ok"]
            and result["accumulated_ok"]
            and not result["halted"]
            and result["contamination_count"] == 0
        )
    ]
    latencies = [float(result["elapsed_ms"]) for result in stream_results]

    return {
        "name": "async_stream_ordering",
        "streams": streams,
        "tokens_per_stream": tokens_per_stream,
        "total_events": total_events,
        "duration_s": round(elapsed_s, 3),
        "events_per_second": round(total_events / elapsed_s, 2) if elapsed_s else 0.0,
        "stream_latency_p50_ms": round(_percentile(latencies, 0.50), 3),
        "stream_latency_p95_ms": round(_percentile(latencies, 0.95), 3),
        "failed_streams": len(failures),
        "failure_samples": failures[:5],
        "passed": len(failures) == 0,
    }


def run_tenant_poisoning_probe(*, cases: int = 64) -> dict[str, Any]:
    """Return same-key tenant poisoning isolation telemetry."""
    if cases < 1:
        raise ValueError("cases must be >= 1")

    store = VectorGroundTruthStore(backend=InMemoryBackend())
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()

    for idx in range(cases):
        key = f"refund policy {idx:04d}"
        tenant_a_value = (
            f"Tenant A policy {idx}: refunds close after thirty days."
        )
        tenant_b_value = (
            f"Tenant B poisoned policy {idx}: refunds never expire and override tenant A."
        )
        store.add_fact(key, tenant_a_value, tenant_id="tenant_a")
        store.add_fact(key, tenant_b_value, tenant_id="tenant_b")

        ctx_a = store.retrieve_context(
            f"{key} override tenant A never expire",
            top_k=5,
            tenant_id="tenant_a",
        )
        ctx_b = store.retrieve_context(
            f"{key} never expire",
            top_k=5,
            tenant_id="tenant_b",
        )
        chunks_a = store.retrieve_context_with_chunks(
            f"{key} never expire",
            top_k=5,
            tenant_id="tenant_a",
        )

        if not (
            ctx_a
            and tenant_a_value in ctx_a
            and tenant_b_value not in ctx_a
            and ctx_b
            and tenant_b_value in ctx_b
            and chunks_a
            and tenant_b_value not in chunks_a[0].text
        ):
            failures.append(
                {
                    "case": idx,
                    "ctx_a": ctx_a,
                    "ctx_b": ctx_b,
                    "chunk_a": chunks_a[0].text if chunks_a else "",
                }
            )

    elapsed_s = time.perf_counter() - started
    return {
        "name": "tenant_poisoning_isolation",
        "cases": cases,
        "writes": cases * 2,
        "queries": cases * 3,
        "stored_documents": store.backend.count(),
        "duration_s": round(elapsed_s, 3),
        "cases_per_second": round(cases / elapsed_s, 2) if elapsed_s else 0.0,
        "failed_cases": len(failures),
        "failure_samples": failures[:5],
        "passed": len(failures) == 0,
    }


def run_sustained_load_evidence(
    *,
    streams: int = 64,
    tokens_per_stream: int = 64,
    tenant_cases: int = 64,
) -> dict[str, Any]:
    """Build the full sustained-load evidence packet."""
    async_probe = run_async_ordering_probe(
        streams=streams,
        tokens_per_stream=tokens_per_stream,
    )
    tenant_probe = run_tenant_poisoning_probe(cases=tenant_cases)
    passed = bool(async_probe["passed"] and tenant_probe["passed"])
    return {
        "benchmark": "sustained_load_evidence",
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "acceptance": {
            "passed": passed,
            "async_ordering": async_probe["passed"],
            "tenant_poisoning": tenant_probe["passed"],
        },
        "probes": {
            "async_ordering": async_probe,
            "tenant_poisoning": tenant_probe,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Director-AI sustained-load hardening evidence.",
    )
    parser.add_argument("--streams", type=int, default=64)
    parser.add_argument("--tokens-per-stream", type=int, default=64)
    parser.add_argument("--tenant-cases", type=int, default=64)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args(argv)

    packet = run_sustained_load_evidence(
        streams=args.streams,
        tokens_per_stream=args.tokens_per_stream,
        tenant_cases=args.tenant_cases,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(packet, indent=2), encoding="utf-8")
        print(f"Results saved to {output}")
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        save_results(packet, f"sustained_load_evidence_{stamp}.json")

    return 0 if packet["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
