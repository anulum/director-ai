# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — latency SLO qualification CLI

"""Installed CLI for deployment-specific latency SLO qualification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from director_ai.core.observability.latency_slo import (
    LatencySLOConfig,
    run_latency_slo,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="director-ai latency-slo",
        description=(
            "Qualify one live /v1/review deployment at a declared operating point. "
            "This is not a universal hard real-time guarantee."
        ),
    )
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--requests", type=int, default=100, dest="request_count")
    parser.add_argument("--warmup", type=int, default=10, dest="warmup_count")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-ms", type=float, default=5_000.0)
    parser.add_argument("--target-p95-ms", type=float, default=500.0)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--tenant-id", default="")
    parser.add_argument(
        "--workload", type=Path, help="JSONL file with prompt and response text pairs"
    )
    parser.add_argument(
        "--api-key-env",
        default="DIRECTOR_API_KEY",
        help="environment variable holding the API key (never written to evidence)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("latency_slo_evidence.json"),
    )
    return parser


def _load_workload(path: Path) -> tuple[tuple[str, str], ...]:
    """Read ordered review pairs without exposing private input in errors."""
    pairs = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"invalid workload JSON at line {number}") from None
            if not isinstance(row, dict) or not (
                isinstance(prompt := row.get("prompt"), str)
                and isinstance(response := row.get("response"), str)
            ):
                raise ValueError(
                    f"workload line {number} requires prompt and response strings"
                )
            pairs.append((prompt, response))
    return tuple(pairs)


def _cmd_latency_slo(args: list[str]) -> None:
    """Measure the configured deployment and save its integrity packet."""
    parser = _parser()
    namespace = parser.parse_args(args)
    api_key = os.environ.get(namespace.api_key_env, "")
    try:
        workload = None
        if namespace.workload is not None:
            workload = _load_workload(namespace.workload)
            if namespace.output.exists() and namespace.workload.samefile(
                namespace.output
            ):
                raise ValueError("output must differ from the workload file")
        config = LatencySLOConfig(
            server_url=namespace.server,
            request_count=namespace.request_count,
            warmup_count=namespace.warmup_count,
            concurrency=namespace.concurrency,
            timeout_ms=namespace.timeout_ms,
            target_p95_ms=namespace.target_p95_ms,
            max_error_rate=namespace.max_error_rate,
            tenant_id=namespace.tenant_id,
            api_key=api_key,
            workload=workload,
        )
    except (OSError, UnicodeError):
        parser.error("cannot read the workload as UTF-8 JSONL")
    except ValueError as exc:
        parser.error(str(exc))

    packet = run_latency_slo(config)
    namespace.output.parent.mkdir(parents=True, exist_ok=True)
    namespace.output.write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    content = packet["content"]
    measurements = content["measurements"]
    qualification = content["qualification"]
    status = "QUALIFIED" if qualification["passed"] else "NOT QUALIFIED"
    print(f"Latency SLO: {status}")
    print(
        "  p95: "
        f"{measurements['latency_p95_ms']} ms "
        f"(target <= {content['targets']['p95_ms_lte']} ms)"
    )
    print(
        "  errors: "
        f"{measurements['error_rate']:.2%} "
        f"(target <= {content['targets']['error_rate_lte']:.2%})"
    )
    print(f"  evidence: {namespace.output}")
    if not qualification["passed"]:
        print(f"  reasons: {', '.join(qualification['failure_reasons'])}")
        raise SystemExit(2)
