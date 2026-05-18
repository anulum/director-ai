# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Vertex model package campaign runner

"""Run every Vertex-eligible per-model benchmark package stage sequentially."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.model_benchmark_packages import (
    ModelBenchmarkPackage,
    ModelEvidenceItem,
    load_package_manifest,
)
from director_ai.core.scoring.nli import _resolve_model_source


@dataclass(frozen=True)
class CampaignRunItem:
    """One concrete model/stage execution in the Vertex campaign."""

    model_alias: str
    runtime_model: str
    stage_id: str
    evidence_id: str
    command: tuple[str, ...]
    output_dir: Path


@dataclass(frozen=True)
class StageResult:
    """Execution and upload outcome for one model/stage item."""

    model_alias: str
    stage_id: str
    evidence_id: str
    returncode: int
    elapsed_seconds: float
    output_dir: str
    uploaded_files: tuple[str, ...]
    error: str = ""


def build_run_items(
    *,
    output_root: Path,
    model_aliases: set[str] | None = None,
    stage_ids: set[str] | None = None,
) -> tuple[CampaignRunItem, ...]:
    """Expand the package manifest into isolated Vertex run items."""

    manifest = load_package_manifest()
    items: list[CampaignRunItem] = []
    for package in manifest.packages:
        if model_aliases is not None and package.model_alias not in model_aliases:
            continue
        for evidence in package.evidence:
            if not evidence.vertex_allowed:
                continue
            if stage_ids is not None and evidence.stage_id not in stage_ids:
                continue
            items.append(_build_run_item(package, evidence, output_root=output_root))
    return tuple(items)


def _build_run_item(
    package: ModelBenchmarkPackage,
    evidence: ModelEvidenceItem,
    *,
    output_root: Path,
) -> CampaignRunItem:
    output_dir = output_root / package.model_alias / evidence.stage_id
    result_path = output_dir / _result_filename(evidence.stage_id)
    model_var = "${DIRECTOR_RESOLVED_NLI_MODEL}"
    commands: dict[str, tuple[str, ...]] = {
        "aggrefact_anchor_vertex": (
            sys.executable,
            "-m",
            "benchmarks.aggrefact_eval",
            "--model",
            model_var,
            "--threshold",
            "0.46",
            "--save-scores",
            str(result_path),
        ),
        "ragtruth_vertex": (
            sys.executable,
            "benchmarks/run_ragtruth_freshqa.py",
        ),
        "halueval_vertex": (
            sys.executable,
            "-m",
            "benchmarks.e2e_eval",
            "--nli",
            "--nli-model",
            model_var,
            "--max-samples",
            "100",
            "--threshold",
            "0.35",
            "--soft-limit",
            "0.45",
            "--output-json",
            str(result_path),
        ),
        "financebench_vertex": (
            sys.executable,
            "-m",
            "benchmarks.finance_eval",
            "--dataset",
            "financebench",
            "--nli",
            "--model",
            model_var,
        ),
        "legal_contractnli_vertex": (
            sys.executable,
            "-m",
            "benchmarks.legal_eval",
            "--dataset",
            "all",
            "--nli",
            "--model",
            model_var,
        ),
        "medical_mednli_pubmedqa_vertex": (
            sys.executable,
            "-m",
            "benchmarks.medical_eval",
            "--dataset",
            "all",
            "--nli",
            "--model",
            model_var,
        ),
        "patronus_halubench_wire": (
            sys.executable,
            "-m",
            "benchmarks.patronus_halubench_eval",
            "--model",
            model_var,
            "--output-json",
            str(result_path),
        ),
    }
    return CampaignRunItem(
        model_alias=package.model_alias,
        runtime_model=package.runtime_model,
        stage_id=evidence.stage_id,
        evidence_id=evidence.evidence_id,
        command=commands[evidence.stage_id],
        output_dir=output_dir,
    )


def _result_filename(stage_id: str) -> str:
    return {
        "aggrefact_anchor_vertex": "aggrefact_scores.json",
        "ragtruth_vertex": "ragtruth_nli_results.json",
        "halueval_vertex": "e2e_guardrail.json",
        "financebench_vertex": "finance_eval.json",
        "legal_contractnli_vertex": "legal_eval.json",
        "medical_mednli_pubmedqa_vertex": "medical_eval.json",
        "patronus_halubench_wire": "patronus_halubench_eval.json",
    }[stage_id]


def run_campaign(
    *,
    output_root: Path,
    bucket_uri: str,
    prefix: str,
    min_free_gb: float,
    model_aliases: set[str] | None = None,
    stage_ids: set[str] | None = None,
) -> tuple[StageResult, ...]:
    """Run and upload every selected model package stage."""

    output_root.mkdir(parents=True, exist_ok=True)
    _require_free_disk(output_root, min_free_gb)
    items = build_run_items(
        output_root=output_root,
        model_aliases=model_aliases,
        stage_ids=stage_ids,
    )
    results: list[StageResult] = []
    for item in items:
        result = _run_one(item, output_root=output_root)
        uploaded = _upload_tree(
            bucket_uri=bucket_uri,
            prefix=f"{prefix}/{item.model_alias}/{item.stage_id}",
            root=item.output_dir,
        )
        results.append(
            StageResult(
                model_alias=result.model_alias,
                stage_id=result.stage_id,
                evidence_id=result.evidence_id,
                returncode=result.returncode,
                elapsed_seconds=result.elapsed_seconds,
                output_dir=result.output_dir,
                uploaded_files=uploaded,
                error=result.error,
            )
        )
        _require_free_disk(output_root, min_free_gb)
    summary_path = output_root / "campaign_summary.json"
    summary_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    _upload_tree(bucket_uri=bucket_uri, prefix=prefix, root=output_root)
    return tuple(results)


def _run_one(item: CampaignRunItem, *, output_root: Path) -> StageResult:
    item.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["DIRECTOR_SCORER_MODEL"] = item.model_alias
    env["DIRECTOR_NLI_MODEL"] = item.runtime_model
    env["DIRECTOR_RESOLVED_NLI_MODEL"] = _resolve_model_source(item.runtime_model)
    env.setdefault("HF_HOME", "/workspace/cache/huggingface")
    env.setdefault("HF_DATASETS_CACHE", "/workspace/cache/hf-datasets")
    env.setdefault("TRANSFORMERS_CACHE", "/workspace/cache/transformers")
    command = tuple(
        env["DIRECTOR_RESOLVED_NLI_MODEL"]
        if part == "${DIRECTOR_RESOLVED_NLI_MODEL}"
        else part
        for part in item.command
    )
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        env=env,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    _move_stage_outputs(item.stage_id, item.output_dir)
    metadata = {
        "model_alias": item.model_alias,
        "runtime_model": item.runtime_model,
        "resolved_model": env["DIRECTOR_RESOLVED_NLI_MODEL"],
        "stage_id": item.stage_id,
        "evidence_id": item.evidence_id,
        "command": list(command),
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "disk_free_gb": _free_gb(output_root),
    }
    (item.output_dir / "stage_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return StageResult(
        model_alias=item.model_alias,
        stage_id=item.stage_id,
        evidence_id=item.evidence_id,
        returncode=completed.returncode,
        elapsed_seconds=elapsed,
        output_dir=str(item.output_dir),
        uploaded_files=(),
        error="" if completed.returncode == 0 else f"returncode={completed.returncode}",
    )


def _move_stage_outputs(stage_id: str, output_dir: Path) -> None:
    result_name = _result_filename(stage_id)
    candidates = [
        Path("benchmarks/results") / result_name,
        Path("benchmarks/results/ragtruth_nli_results.json"),
        Path("benchmarks/results/freshqa_nli_results.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            target = output_dir / candidate.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidate), target)


def _upload_tree(*, bucket_uri: str, prefix: str, root: Path) -> tuple[str, ...]:
    from google.cloud import storage

    bucket_name = bucket_uri.removeprefix("gs://").strip("/")
    clean_prefix = prefix.strip("/")
    bucket = storage.Client().bucket(bucket_name)
    uploaded: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        blob_name = f"{clean_prefix}/{rel}"
        bucket.blob(blob_name).upload_from_filename(str(path))
        uploaded.append(f"gs://{bucket_name}/{blob_name}")
    return tuple(uploaded)


def _free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1024**3


def _require_free_disk(path: Path, min_free_gb: float) -> None:
    free = _free_gb(path)
    if free < min_free_gb:
        raise RuntimeError(
            f"free disk below threshold: {free:.1f} GiB < {min_free_gb:.1f} GiB",
        )


def _parse_csv(raw: str) -> set[str] | None:
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or None


def _summarise(results: Iterable[StageResult]) -> int:
    failures = [result for result in results if result.returncode != 0]
    print(json.dumps([asdict(result) for result in results], indent=2))
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/output"))
    parser.add_argument("--bucket", default=os.environ.get("DIRECTOR_BENCH_BUCKET", ""))
    parser.add_argument("--prefix", default=os.environ.get("DIRECTOR_BENCH_PREFIX", ""))
    parser.add_argument("--model-aliases", default="")
    parser.add_argument("--stage-ids", default="")
    parser.add_argument("--min-free-gb", type=float, default=25.0)
    args = parser.parse_args(argv)

    if not args.bucket.startswith("gs://"):
        raise SystemExit("--bucket or DIRECTOR_BENCH_BUCKET must be a gs:// URI")
    if not args.prefix.strip():
        raise SystemExit("--prefix or DIRECTOR_BENCH_PREFIX is required")
    results = run_campaign(
        output_root=args.output_root,
        bucket_uri=args.bucket,
        prefix=args.prefix,
        min_free_gb=args.min_free_gb,
        model_aliases=_parse_csv(args.model_aliases),
        stage_ids=_parse_csv(args.stage_ids),
    )
    return _summarise(results)


if __name__ == "__main__":
    raise SystemExit(main())
