# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - per-model benchmark package planner

"""Build and validate per-model benchmark evidence packages."""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.credibility_campaign import load_campaign
from director_ai.core.scoring.model_choices import (
    ScorerModelChoice,
    list_scorer_model_choices,
)

PACKAGE_MANIFEST_PATH = Path(__file__).with_name("model_benchmark_packages.toml")
SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class RequiredPackageStage:
    """One evidence stage every public model package must contain."""

    stage_id: str
    title: str
    source_stage_id: str
    vertex_allowed: bool
    required_for_public_claim: bool


@dataclass(frozen=True)
class ModelEvidenceItem:
    """One model-specific evidence item."""

    evidence_id: str
    model_alias: str
    stage_id: str
    title: str
    status: str
    command: str
    vertex_allowed: bool
    required_for_public_claim: bool


@dataclass(frozen=True)
class ModelBenchmarkPackage:
    """Evidence package for one selectable runtime scorer model."""

    model_alias: str
    model_id: str
    runtime_model: str
    status: str
    public_claim: str
    evidence: tuple[ModelEvidenceItem, ...]


@dataclass(frozen=True)
class ModelPackageManifest:
    """Per-model package manifest expanded with runtime model choices."""

    schema_version: str
    package_id: str
    description: str
    default_status: str
    vertex_project: str
    vertex_region: str
    result_prefix_template: str
    required_stages: tuple[RequiredPackageStage, ...]
    packages: tuple[ModelBenchmarkPackage, ...]


@dataclass(frozen=True)
class NextModelPackageWork:
    """Next missing model evidence item."""

    model_alias: str
    stage_id: str
    evidence_id: str
    command: str
    vertex_allowed: bool


def load_package_manifest(path: Path = PACKAGE_MANIFEST_PATH) -> ModelPackageManifest:
    """Load and expand the per-model benchmark package manifest."""

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    required_stages = tuple(
        RequiredPackageStage(
            stage_id=stage["stage_id"],
            title=stage["title"],
            source_stage_id=stage.get("source_stage_id", ""),
            vertex_allowed=bool(stage.get("vertex_allowed", False)),
            required_for_public_claim=bool(stage.get("required_for_public_claim", True)),
        )
        for stage in raw["required_stages"]
    )
    choices = {
        choice.alias: choice for choice in list_scorer_model_choices(include_domain_only=True)
    }
    campaign = load_campaign()
    campaign_stages = {stage.stage_id: stage for stage in campaign.stages}
    packages = []
    for package_raw in raw["packages"]:
        alias = package_raw["model_alias"]
        choice = choices[alias]
        evidence = tuple(
            _build_evidence_item(
                model=choice,
                package_status=package_raw.get("status", raw["default_status"]),
                required_stage=required_stage,
                campaign_stages=campaign_stages,
            )
            for required_stage in required_stages
        )
        packages.append(
            ModelBenchmarkPackage(
                model_alias=alias,
                model_id=choice.model_id,
                runtime_model=choice.runtime_model,
                status=package_raw.get("status", raw["default_status"]),
                public_claim=package_raw["public_claim"],
                evidence=evidence,
            )
        )
    return ModelPackageManifest(
        schema_version=raw["schema_version"],
        package_id=raw["package_id"],
        description=raw["description"],
        default_status=raw["default_status"],
        vertex_project=raw["vertex_project"],
        vertex_region=raw["vertex_region"],
        result_prefix_template=raw["result_prefix_template"],
        required_stages=required_stages,
        packages=tuple(packages),
    )


def validate_package_manifest(
    manifest: ModelPackageManifest,
    *,
    root: Path,
) -> list[str]:
    """Return structural findings for the package manifest."""

    findings: list[str] = []
    if manifest.schema_version != SCHEMA_VERSION:
        findings.append(f"unsupported schema_version={manifest.schema_version!r}")
    if not manifest.result_prefix_template.startswith("gs://"):
        findings.append("result_prefix_template must be a gs:// URI")

    stable_aliases = {choice.alias for choice in list_scorer_model_choices()}
    package_aliases = {package.model_alias for package in manifest.packages}
    missing_aliases = stable_aliases - package_aliases
    if missing_aliases:
        findings.append(f"stable scorer aliases missing packages: {sorted(missing_aliases)}")

    stage_ids = {stage.stage_id for stage in manifest.required_stages}
    if len(stage_ids) != len(manifest.required_stages):
        findings.append("duplicate required stage ids")

    campaign = load_campaign()
    campaign_stage_ids = {stage.stage_id for stage in campaign.stages}
    for required_stage in manifest.required_stages:
        if (
            required_stage.source_stage_id
            and required_stage.source_stage_id not in campaign_stage_ids
        ):
            findings.append(
                f"{required_stage.stage_id}: source stage missing from campaign"
            )

    for package in manifest.packages:
        if not package.public_claim.strip():
            findings.append(f"{package.model_alias}: missing public claim")
        if not package.runtime_model:
            findings.append(f"{package.model_alias}: missing runtime model")
        evidence_stage_ids = {item.stage_id for item in package.evidence}
        if evidence_stage_ids != stage_ids:
            findings.append(f"{package.model_alias}: evidence does not match stages")
        for item in package.evidence:
            if not item.command.strip():
                findings.append(f"{item.evidence_id}: missing command")
            if item.stage_id == "model_choice_general_gate":
                continue
            if item.vertex_allowed and "DIRECTOR_SCORER_MODEL=" not in item.command:
                findings.append(f"{item.evidence_id}: command does not set model")

    readme = root / "README.md"
    if not readme.exists():
        findings.append("README.md missing")
    return findings


def next_model_package_work(
    manifest: ModelPackageManifest,
    *,
    completed_evidence_ids: set[str],
) -> NextModelPackageWork | None:
    """Return the first missing evidence item in manifest order."""

    for package in manifest.packages:
        for item in package.evidence:
            if item.evidence_id not in completed_evidence_ids and item.vertex_allowed:
                return NextModelPackageWork(
                    model_alias=package.model_alias,
                    stage_id=item.stage_id,
                    evidence_id=item.evidence_id,
                    command=item.command,
                    vertex_allowed=item.vertex_allowed,
                )
    return None


def package_manifest_to_dict(manifest: ModelPackageManifest) -> dict:
    """Serialise the manifest to JSON-safe containers."""

    return {
        "model_benchmark_packages": asdict(manifest),
    }


def _build_evidence_item(
    *,
    model: ScorerModelChoice,
    package_status: str,
    required_stage: RequiredPackageStage,
    campaign_stages: dict[str, object],
) -> ModelEvidenceItem:
    if required_stage.stage_id == "model_choice_general_gate":
        command = (
            "director-ai train benchmark-models "
            f"--model {model.alias}={model.runtime_model} "
            "--general-uri gs://gotm-director-ai-training/labels/sweeps/20260428/"
            "managed_eval_1000_20260428.jsonl"
        )
    else:
        campaign_stage = campaign_stages[required_stage.source_stage_id]
        stage_command = campaign_stage.command
        command = (
            f"DIRECTOR_SCORER_MODEL={model.alias} "
            f"DIRECTOR_NLI_MODEL={model.runtime_model} "
            f"{stage_command}"
        )
    return ModelEvidenceItem(
        evidence_id=f"{model.alias}:{required_stage.stage_id}",
        model_alias=model.alias,
        stage_id=required_stage.stage_id,
        title=required_stage.title,
        status="complete" if required_stage.stage_id == "model_choice_general_gate" else package_status,
        command=command,
        vertex_allowed=required_stage.vertex_allowed,
        required_for_public_claim=required_stage.required_for_public_claim,
    )


def _parse_completed(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and plan per-model benchmark packages.",
    )
    parser.add_argument("--manifest", type=Path, default=PACKAGE_MANIFEST_PATH)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--completed", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_package_manifest(args.manifest)
    findings = validate_package_manifest(manifest, root=args.root)
    next_work = next_model_package_work(
        manifest,
        completed_evidence_ids=_parse_completed(args.completed),
    )
    if args.json:
        print(
            json.dumps(
                {
                    **package_manifest_to_dict(manifest),
                    "findings": findings,
                    "next_work": asdict(next_work) if next_work else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"package_id: {manifest.package_id}")
        print(f"findings: {len(findings)}")
        for finding in findings:
            print(f"- {finding}")
        if next_work:
            print("next work:")
            print(f"- {next_work.evidence_id} [{next_work.command}]")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
