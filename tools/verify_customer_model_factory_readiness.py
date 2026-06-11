# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory readiness gate

"""Verify internal enterprise evidence required before factory promotion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RequiredEvidence:
    """One enterprise-trust evidence file required for readiness."""

    control_id: str
    debt_id: str
    title: str
    path: str


@dataclass(frozen=True)
class ReadinessResult:
    """Customer Model Factory readiness gate result."""

    ready: bool
    present_paths: tuple[str, ...]
    missing_paths: tuple[str, ...]
    blocking_debt_ids: tuple[str, ...]

    def to_markdown(self) -> str:
        """Return a compact human-readable readiness report."""

        rows = []
        for item in REQUIRED_EVIDENCE:
            status = "present" if item.path in self.present_paths else "missing"
            rows.append(
                f"| {item.control_id} | {item.debt_id} | {status} | `{item.path}` |"
            )
        return "\n".join(
            [
                "# Customer Model Factory Readiness",
                "",
                f"ready: {str(self.ready).lower()}",
                "",
                "| Control | Trust Debt | Status | Evidence |",
                "|---|---|---|---|",
                *rows,
                "",
            ]
        )


REQUIRED_EVIDENCE: tuple[RequiredEvidence, ...] = (
    RequiredEvidence(
        control_id="TRUST-THREAT-001",
        debt_id="TRUST-DEBT-0002",
        title="Customer Model Factory threat model",
        path=(
            "docs/internal/threat_models/"
            "director_ai_enterprise_customer_model_factory_2026-05-18.md"
        ),
    ),
    RequiredEvidence(
        control_id="TRUST-DATA-001",
        debt_id="TRUST-DEBT-0003",
        title="Customer Model Factory data-flow and data-lineage evidence",
        path=(
            "docs/internal/data_flow/"
            "director_ai_customer_model_factory_data_flow_2026-05-18.md"
        ),
    ),
)


def evaluate_readiness(root: Path) -> ReadinessResult:
    """Evaluate whether required internal enterprise evidence exists."""

    present = []
    missing = []
    blocking_debt = []
    for item in REQUIRED_EVIDENCE:
        if (root / item.path).is_file():
            present.append(item.path)
        else:
            missing.append(item.path)
            blocking_debt.append(item.debt_id)
    return ReadinessResult(
        ready=not missing,
        present_paths=tuple(present),
        missing_paths=tuple(missing),
        blocking_debt_ids=tuple(blocking_debt),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the readiness gate from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    result = evaluate_readiness(args.root)
    markdown = result.to_markdown()
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
