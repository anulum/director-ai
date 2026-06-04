#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Gate release claims against local roadmap evidence packets.

The individual benchmark packets are useful local evidence. They are not enough
for a customer release when they still declare local-only limits, missing smoke
evidence, missing operator sign-off, or absent external runs. This tool makes
that distinction mechanical.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "director-ai.local-release-evidence-gate.v1"
GateMode = Literal["local", "release"]


@dataclass(frozen=True)
class RequiredEvidencePacket:
    """One roadmap evidence packet required for local/release claims."""

    roadmap_id: str
    benchmark: str
    description: str

    @property
    def pattern(self) -> str:
        """Return the benchmark result filename pattern."""

        return f"{self.benchmark}_*.json"


@dataclass(frozen=True)
class EvidencePacketStatus:
    """Gate status for one evidence packet."""

    roadmap_id: str
    benchmark: str
    description: str
    path: str
    local_ready: bool
    release_ready: bool
    local_blockers: tuple[dict[str, str], ...]
    release_blockers: tuple[dict[str, str], ...]

    @property
    def blockers(self) -> tuple[dict[str, str], ...]:
        """Return every blocker for compatibility with report consumers."""

        return self.local_blockers + self.release_blockers

    def to_dict(self) -> dict[str, Any]:
        """Serialise the packet status."""

        return {
            "roadmap_id": self.roadmap_id,
            "benchmark": self.benchmark,
            "description": self.description,
            "path": self.path,
            "local_ready": self.local_ready,
            "release_ready": self.release_ready,
            "local_blockers": [dict(blocker) for blocker in self.local_blockers],
            "release_blockers": [dict(blocker) for blocker in self.release_blockers],
            "blockers": [dict(blocker) for blocker in self.blockers],
        }


@dataclass(frozen=True)
class ReleaseEvidenceGate:
    """Aggregate local/release evidence gate result."""

    schema_version: str
    mode: GateMode
    ready: bool
    local_ready: bool
    release_ready: bool
    packets: tuple[EvidencePacketStatus, ...]

    @property
    def blockers(self) -> tuple[dict[str, str], ...]:
        """Return blockers that apply to the selected gate mode."""

        if self.mode == "local":
            return tuple(
                blocker
                for packet in self.packets
                for blocker in packet.local_blockers
            )
        return tuple(blocker for packet in self.packets for blocker in packet.blockers)

    @property
    def release_blockers(self) -> tuple[dict[str, str], ...]:
        """Return all blockers that still prevent a release claim."""

        return tuple(
            blocker for packet in self.packets for blocker in packet.release_blockers
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the gate result."""

        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "ready": self.ready,
            "local_ready": self.local_ready,
            "release_ready": self.release_ready,
            "packets": [packet.to_dict() for packet in self.packets],
            "blockers": [dict(blocker) for blocker in self.blockers],
            "release_blockers": [
                dict(blocker) for blocker in self.release_blockers
            ],
        }

    def to_markdown(self) -> str:
        """Return a compact operator-readable report."""

        rows = [
            "| Roadmap | Benchmark | Local | Release | Packet |",
            "|---|---|---:|---:|---|",
        ]
        for packet in self.packets:
            rows.append(
                f"| {packet.roadmap_id} | {packet.benchmark} | "
                f"{str(packet.local_ready).lower()} | "
                f"{str(packet.release_ready).lower()} | `{packet.path}` |"
            )
        mode_blocker_lines = [
            f"- {item['roadmap_id']}:{item['code']} — {item['message']}"
            for item in self.blockers
        ]
        if not mode_blocker_lines:
            mode_blocker_lines = ["- none"]
        release_blocker_lines = [
            f"- {item['roadmap_id']}:{item['code']} — {item['message']}"
            for item in self.release_blockers
        ]
        if not release_blocker_lines:
            release_blocker_lines = ["- none"]
        return "\n".join(
            [
                "# Local Release Evidence Gate",
                "",
                f"mode: {self.mode}",
                f"ready: {str(self.ready).lower()}",
                f"local_ready: {str(self.local_ready).lower()}",
                f"release_ready: {str(self.release_ready).lower()}",
                "",
                *rows,
                "",
                "## Mode Blockers",
                "",
                *mode_blocker_lines,
                "",
                "## Release Blockers",
                "",
                *release_blocker_lines,
                "",
            ]
        )


REQUIRED_PACKETS: tuple[RequiredEvidencePacket, ...] = (
    RequiredEvidencePacket("R9", "provenance_evidence", "KB provenance lineage"),
    RequiredEvidencePacket("R10", "conformal_routing_evidence", "Conformal routing"),
    RequiredEvidencePacket("R11", "trajectory_rollback_evidence", "Trajectory rollback"),
    RequiredEvidencePacket("R12", "multimodal_temporal_evidence", "Multimodal temporal"),
    RequiredEvidencePacket("R13", "federated_privacy_evidence", "Federated privacy"),
    RequiredEvidencePacket("R14", "edge_mobile_evidence", "Edge/mobile runtime"),
    RequiredEvidencePacket("R15", "auto_redteam_defence_evidence", "Auto-redteam loop"),
    RequiredEvidencePacket("R16", "formal_symbolic_evidence", "Formal symbolic"),
    RequiredEvidencePacket("R17", "sustained_load_evidence", "Async tenant hardening"),
)


def evaluate_release_evidence(
    root: str | Path,
    *,
    mode: GateMode = "local",
    required_packets: tuple[RequiredEvidencePacket, ...] = REQUIRED_PACKETS,
) -> ReleaseEvidenceGate:
    """Evaluate local or release readiness for roadmap evidence packets."""

    repo = Path(root).resolve()
    statuses = tuple(_evaluate_packet(repo, packet) for packet in required_packets)
    local_ready = all(packet.local_ready for packet in statuses)
    release_ready = all(packet.release_ready for packet in statuses)
    ready = local_ready if mode == "local" else release_ready
    return ReleaseEvidenceGate(
        schema_version=SCHEMA_VERSION,
        mode=mode,
        ready=ready,
        local_ready=local_ready,
        release_ready=release_ready,
        packets=statuses,
    )


def _evaluate_packet(
    repo: Path,
    required: RequiredEvidencePacket,
) -> EvidencePacketStatus:
    path = _latest_packet(repo, required)
    if path is None:
        blocker = _blocker(
            required,
            "evidence_packet_missing",
            f"No {required.pattern} packet found under benchmarks/results",
        )
        return EvidencePacketStatus(
            roadmap_id=required.roadmap_id,
            benchmark=required.benchmark,
            description=required.description,
            path="",
            local_ready=False,
            release_ready=False,
            local_blockers=(blocker,),
            release_blockers=(),
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        blocker = _blocker(
            required,
            "evidence_packet_invalid_json",
            f"Packet is not valid JSON: {exc.msg}",
        )
        return EvidencePacketStatus(
            roadmap_id=required.roadmap_id,
            benchmark=required.benchmark,
            description=required.description,
            path=_rel(repo, path),
            local_ready=False,
            release_ready=False,
            local_blockers=(blocker,),
            release_blockers=(),
        )
    acceptance = payload.get("acceptance", {})
    local_blockers = []
    acceptance_passed = bool(acceptance.get("passed") is True)
    local_ready = acceptance_passed
    if not local_ready:
        local_blockers.append(
            _blocker(
                required,
                "evidence_packet_not_passing",
                "Packet acceptance.passed is not true",
            )
        )
    release_blockers = tuple(_release_blockers(required, payload))
    return EvidencePacketStatus(
        roadmap_id=required.roadmap_id,
        benchmark=required.benchmark,
        description=required.description,
        path=_rel(repo, path),
        local_ready=local_ready,
        release_ready=local_ready and not release_blockers,
        local_blockers=tuple(local_blockers),
        release_blockers=release_blockers,
    )


def _latest_packet(repo: Path, required: RequiredEvidencePacket) -> Path | None:
    result_dir = repo / "benchmarks" / "results"
    candidates = sorted(result_dir.glob(required.pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _release_blockers(
    required: RequiredEvidencePacket,
    payload: dict[str, Any],
) -> list[dict[str, str]]:
    acceptance = payload.get("acceptance", {})
    limits = acceptance.get("limits", {})
    blockers: list[dict[str, str]] = []
    if limits.get("local_only") is True:
        blockers.append(
            _blocker(
                required,
                "local_only_evidence",
                "Packet is explicitly local-only evidence",
            )
        )
    for key, value in sorted(limits.items()):
        if key == "local_only":
            continue
        if key.endswith("_included") and value is False:
            blockers.append(
                _blocker(
                    required,
                    f"missing_{key}",
                    f"Release evidence limit {key} is false",
                )
            )
    if required.benchmark == "edge_mobile_evidence":
        profile = payload.get("profiles", {}).get("browser-worker", {})
        if profile.get("ready_for_release") is not True:
            blockers.append(
                _blocker(
                    required,
                    "edge_runtime_not_release_ready",
                    "Edge/mobile profile ready_for_release is not true",
                )
            )
    return blockers


def _blocker(
    required: RequiredEvidencePacket,
    code: str,
    message: str,
) -> dict[str, str]:
    return {
        "roadmap_id": required.roadmap_id,
        "benchmark": required.benchmark,
        "code": code,
        "severity": "error",
        "message": message,
    }


def _rel(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo).as_posix()
    except ValueError:
        return "external path not serialised"


def main(argv: list[str] | None = None) -> int:
    """Run the evidence gate from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("local", "release"), default="local")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report")
    args = parser.parse_args(argv)

    result = evaluate_release_evidence(args.root, mode=args.mode)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(result.to_markdown())
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
