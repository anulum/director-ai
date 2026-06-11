# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — edge and mobile runtime evidence packet

"""Generate local R14 evidence for edge and mobile runtime readiness."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess  # nosec B404
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks._common import save_results
from director_ai.core.edge import build_edge_runtime_readiness


def _git_commit(repo: Path) -> str:
    git = shutil.which("git")
    if not git:
        return "unknown"
    try:
        completed = subprocess.run(  # nosec B603
            [git, "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def run_edge_mobile_evidence(
    *,
    repo_root: str | Path | None = None,
    quantised_model_path: str | Path | None = None,
    browser_smoke_evidence: str | Path | None = None,
    mobile_smoke_evidence: str | Path | None = None,
) -> dict[str, Any]:
    """Return the complete local R14 edge/mobile evidence packet."""

    repo = Path(repo_root) if repo_root is not None else Path(__file__).parents[1]
    repo = repo.resolve()
    profile = build_edge_runtime_readiness(
        repo,
        target_id="browser-worker",
        quantised_model_path=quantised_model_path,
        browser_smoke_evidence=browser_smoke_evidence,
        mobile_smoke_evidence=mobile_smoke_evidence,
    )
    checks = profile.check_map()
    serialised = json.dumps(profile.to_dict(), sort_keys=True)
    tenant_safe_serialisation = not _raw_paths_leaked(
        serialised,
        quantised_model_path,
        browser_smoke_evidence,
        mobile_smoke_evidence,
    )
    passed = bool(
        profile.ready_for_local_trial
        and checks["wasm_release_plan"].passed
        and checks["wasm_source_contract"].passed
        and checks["quantised_nli_contract"].passed
        and checks["rust_kernel_source_contract"].passed
        and tenant_safe_serialisation
    )
    return {
        "schema_version": "director-ai.edge-mobile-evidence.v1",
        "benchmark": "edge_mobile_evidence",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(repo),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "acceptance": {
            "passed": passed,
            "checks": {
                "browser_worker_local_trial_ready": profile.ready_for_local_trial,
                "wasm_release_plan": checks["wasm_release_plan"].passed,
                "wasm_source_contract": checks["wasm_source_contract"].passed,
                "quantised_nli_contract": checks["quantised_nli_contract"].passed,
                "rust_kernel_source_contract": checks[
                    "rust_kernel_source_contract"
                ].passed,
                "edge_deployment_docs": checks["edge_deployment_docs"].passed,
                "latency_benchmark_surface": checks["latency_benchmark_surface"].passed,
                "tenant_safe_serialisation": tenant_safe_serialisation,
            },
            "limits": dict(profile.limits),
        },
        "profiles": {
            "browser-worker": profile.to_dict(),
        },
    }


def _raw_paths_leaked(
    serialised: str,
    *paths: str | Path | None,
) -> bool:
    """Return whether any caller-supplied raw absolute path reached JSON."""

    for path in paths:
        if path is None:
            continue
        raw = str(path)
        if Path(path).is_absolute() and raw in serialised:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Director-AI edge/mobile runtime evidence packet.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root. Defaults to the parent of benchmarks/.",
    )
    parser.add_argument(
        "--quantised-model-path",
        type=Path,
        default=None,
        help="Optional quantised ONNX model artefact path.",
    )
    parser.add_argument(
        "--browser-smoke-evidence",
        type=Path,
        default=None,
        help="Optional browser or Web Worker smoke evidence path.",
    )
    parser.add_argument(
        "--mobile-smoke-evidence",
        type=Path,
        default=None,
        help="Optional mobile or embedded-device smoke evidence path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to benchmarks/results/.",
    )
    args = parser.parse_args(argv)

    payload = run_edge_mobile_evidence(
        repo_root=args.repo_root,
        quantised_model_path=args.quantised_model_path,
        browser_smoke_evidence=args.browser_smoke_evidence,
        mobile_smoke_evidence=args.mobile_smoke_evidence,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results saved to {args.output}")
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        save_results(payload, f"edge_mobile_evidence_{stamp}.json")
    print(json.dumps(payload["acceptance"], indent=2))
    return 0 if payload["acceptance"]["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
