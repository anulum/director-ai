# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - credibility campaign planner

"""Read and validate the Vertex credibility benchmark campaign manifest."""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

CAMPAIGN_PATH = Path(__file__).with_name("vertex_credibility_campaign.toml")
SCHEMA_VERSION = "1.0.0"
_TOKEN_SENTINELS = ("<accepted-access-token>", "<hf-token>", "<token>")


@dataclass(frozen=True)
class VertexDefaults:
    """Default Vertex AI target for credibility benchmark jobs."""

    project: str
    region: str
    bucket: str
    machine_type: str
    accelerator: str
    accelerator_count: int
    image: str
    run_prefix_template: str


@dataclass(frozen=True)
class CampaignStage:
    """One ordered benchmark or preparation stage."""

    stage_id: str
    title: str
    purpose: str
    dataset_ids: tuple[str, ...]
    runner: str
    command: str
    expected_outputs: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    requires_gated_access: bool
    requires_predictions: bool
    vertex_allowed: bool
    depends_on: tuple[str, ...]
    claim_boundary: str


@dataclass(frozen=True)
class CredibilityCampaign:
    """Versioned benchmark campaign definition."""

    schema_version: str
    campaign_id: str
    description: str
    vertex: VertexDefaults
    stages: tuple[CampaignStage, ...]


def load_campaign(path: Path = CAMPAIGN_PATH) -> CredibilityCampaign:
    """Load a credibility campaign TOML file into typed records."""

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    vertex = VertexDefaults(**raw["vertex"])
    stages = tuple(
        CampaignStage(
            stage_id=stage["stage_id"],
            title=stage["title"],
            purpose=stage["purpose"],
            dataset_ids=tuple(stage.get("dataset_ids", ())),
            runner=stage["runner"],
            command=_redact_command(stage["command"]),
            expected_outputs=tuple(stage.get("expected_outputs", ())),
            primary_metrics=tuple(stage.get("primary_metrics", ())),
            requires_gated_access=bool(stage.get("requires_gated_access", False)),
            requires_predictions=bool(stage.get("requires_predictions", False)),
            vertex_allowed=bool(stage.get("vertex_allowed", False)),
            depends_on=tuple(stage.get("depends_on", ())),
            claim_boundary=stage["claim_boundary"],
        )
        for stage in raw["stages"]
    )
    return CredibilityCampaign(
        schema_version=raw["schema_version"],
        campaign_id=raw["campaign_id"],
        description=raw["description"],
        vertex=vertex,
        stages=stages,
    )


def validate_campaign(campaign: CredibilityCampaign, *, root: Path) -> list[str]:
    """Return structural findings for the campaign manifest.

    The validator intentionally does not require result files to exist:
    those are produced by the campaign itself. It does require runners
    and local manifests to exist so a stage cannot point at a placeholder.
    """

    findings: list[str] = []
    if campaign.schema_version != SCHEMA_VERSION:
        findings.append(f"unsupported schema_version={campaign.schema_version!r}")

    seen: set[str] = set()
    for stage in campaign.stages:
        if stage.stage_id in seen:
            findings.append(f"duplicate stage_id={stage.stage_id}")
        seen.add(stage.stage_id)

        if not stage.title.strip():
            findings.append(f"{stage.stage_id}: missing title")
        if not stage.purpose.strip():
            findings.append(f"{stage.stage_id}: missing purpose")
        if not stage.command.strip():
            findings.append(f"{stage.stage_id}: missing command")
        if not stage.primary_metrics:
            findings.append(f"{stage.stage_id}: missing primary_metrics")
        if not stage.claim_boundary.strip():
            findings.append(f"{stage.stage_id}: missing claim_boundary")
        if any(sentinel in stage.command for sentinel in _TOKEN_SENTINELS):
            findings.append(f"{stage.stage_id}: command contains literal token sentinel")

        runner_path = root / stage.runner
        if not runner_path.exists():
            findings.append(f"{stage.stage_id}: runner missing: {stage.runner}")
        for dependency in stage.depends_on:
            if dependency not in seen:
                findings.append(
                    f"{stage.stage_id}: dependency must precede stage: {dependency}"
                )

    return findings


def next_runnable_stages(
    campaign: CredibilityCampaign,
    *,
    completed_stage_ids: Iterable[str],
    include_gated: bool = False,
) -> tuple[CampaignStage, ...]:
    """Return the next dependency-ready stages.

    Gated stages are excluded by default because access approval and
    prediction provenance are human/account state, not code state.
    """

    completed = set(completed_stage_ids)
    ready: list[CampaignStage] = []
    for stage in campaign.stages:
        if stage.stage_id in completed:
            continue
        if not set(stage.depends_on) <= completed:
            continue
        if (stage.requires_gated_access or stage.requires_predictions) and not include_gated:
            continue
        ready.append(stage)
    return tuple(ready)


def campaign_to_dict(campaign: CredibilityCampaign) -> dict:
    """Serialise a campaign to plain JSON-compatible containers."""

    return asdict(campaign)


def _redact_command(command: str) -> str:
    redacted = command
    for sentinel in _TOKEN_SENTINELS:
        redacted = redacted.replace(sentinel, "HF_TOKEN")
    return redacted


def _parse_completed(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and plan the Director-AI credibility campaign.",
    )
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_PATH)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--completed", default="")
    parser.add_argument("--include-gated", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    campaign = load_campaign(args.manifest)
    findings = validate_campaign(campaign, root=args.root)
    ready = next_runnable_stages(
        campaign,
        completed_stage_ids=_parse_completed(args.completed),
        include_gated=args.include_gated,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "campaign": campaign_to_dict(campaign),
                    "findings": findings,
                    "next_runnable_stage_ids": [stage.stage_id for stage in ready],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"campaign: {campaign.campaign_id}")
        print(f"findings: {len(findings)}")
        for finding in findings:
            print(f"- {finding}")
        print("next runnable:")
        for stage in ready:
            target = "vertex" if stage.vertex_allowed else "local"
            print(f"- {stage.stage_id} [{target}] {stage.command}")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
