# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vertex model-choice benchmark entrypoint

"""Run managed-training model-choice benchmarks inside Vertex AI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from google.cloud import storage

from director_ai.core.training.finetune_benchmark import benchmark_model_candidates

_SKIP_FILENAMES = {
    "optimizer.pt",
    "rng_state.pth",
    "scaler.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
}


def _parse_model_specs(raw: str) -> dict[str, str]:
    """Parse JSON or ``alias=uri;alias=uri`` model specifications."""

    raw = raw.strip()
    if not raw:
        raise ValueError("DIRECTOR_MODEL_BENCHMARK_MODELS is required")
    if raw.startswith("{"):
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("model JSON must be an object")
        return {str(k): str(v) for k, v in parsed.items()}

    models: dict[str, str] = {}
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        alias, sep, uri = item.partition("=")
        if not sep or not alias.strip() or not uri.strip():
            raise ValueError(f"invalid model specification: {item!r}")
        models[alias.strip()] = uri.strip()
    if not models:
        raise ValueError("no model specifications parsed")
    return models


def _split_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"not a gs:// URI: {uri}")
    path = uri[5:]
    bucket, sep, blob = path.partition("/")
    if not sep or not bucket or not blob:
        raise ValueError(f"incomplete gs:// URI: {uri}")
    return bucket, blob.strip("/")


def _download_file(client: storage.Client, uri: str, dest: Path) -> Path:
    bucket_name, blob_name = _split_gs_uri(uri)
    dest.parent.mkdir(parents=True, exist_ok=True)
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(dest))
    return dest


def _should_skip_artifact(rel: str) -> bool:
    parts = Path(rel).parts
    if any(part.startswith("checkpoint-") for part in parts):
        return True
    return Path(rel).name in _SKIP_FILENAMES


def _download_artifact(client: storage.Client, uri: str, dest: Path) -> Path:
    """Download a model artefact prefix, excluding trainer-only checkpoints."""

    if not uri.startswith("gs://"):
        return Path(uri)

    bucket_name, prefix = _split_gs_uri(uri)
    bucket = client.bucket(bucket_name)
    prefix = prefix.rstrip("/")
    count = 0
    for blob in client.list_blobs(bucket_name, prefix=f"{prefix}/"):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(prefix) + 1 :]
        if not rel or _should_skip_artifact(rel):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        bucket.blob(blob.name).download_to_filename(str(target))
        count += 1
        print(f"downloaded gs://{bucket_name}/{blob.name} -> {target}")
    if count == 0:
        raise FileNotFoundError(f"no model files downloaded from {uri}")
    return dest


def _write_summary(report: object, output_dir: Path) -> None:
    data = report.to_dict()
    lines = [
        "# Vertex model-choice benchmark",
        "",
        f"General data: `{data.get('general_path', '')}`",
        f"Domain data: `{data.get('eval_path', '')}`",
        f"Best model: `{data.get('best_model_alias') or 'none'}`",
        "",
        "| Model | General BA | Domain BA | Regression pp | Recommendation |",
        "|---|---:|---:|---:|---|",
    ]
    for result in data.get("results", []):
        lines.append(
            "| {alias} | {general:.3f} | {domain:.3f} | {regression:+.1f} | {rec} |".format(
                alias=result.get("alias", result.get("requested_model", "")),
                general=float(result.get("general_accuracy") or 0.0),
                domain=float(result.get("domain_accuracy") or 0.0),
                regression=float(result.get("regression_pp") or 0.0),
                rec=result.get("recommendation", ""),
            )
        )
    (output_dir / "model_benchmark_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/workspace/output")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path("/workspace/model-benchmark-inputs")
    work_dir.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    models = _parse_model_specs(os.environ.get("DIRECTOR_MODEL_BENCHMARK_MODELS", ""))
    local_models = {
        alias: str(_download_artifact(client, uri, work_dir / "models" / alias))
        for alias, uri in models.items()
    }

    general_uri = os.environ.get("DIRECTOR_MODEL_BENCHMARK_GENERAL_URI", "").strip()
    eval_uri = os.environ.get("DIRECTOR_MODEL_BENCHMARK_EVAL_URI", "").strip()
    general_path = (
        str(_download_file(client, general_uri, work_dir / "general.jsonl"))
        if general_uri.startswith("gs://")
        else general_uri or None
    )
    eval_path = (
        str(_download_file(client, eval_uri, work_dir / "eval.jsonl"))
        if eval_uri.startswith("gs://")
        else eval_uri or None
    )

    batch_size_raw = os.environ.get("DIRECTOR_MODEL_BENCHMARK_BATCH_SIZE", "").strip()
    report = benchmark_model_candidates(
        local_models,
        general_path=general_path,
        eval_path=eval_path,
        batch_size=int(batch_size_raw) if batch_size_raw else None,
        allow_experimental=True,
    )
    (output_dir / "model_benchmark_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_summary(report, output_dir)
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
