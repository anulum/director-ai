# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory readiness tests

from __future__ import annotations

from pathlib import Path

from tools.verify_customer_model_factory_readiness import (
    REQUIRED_EVIDENCE,
    evaluate_readiness,
)


def test_readiness_fails_when_enterprise_evidence_is_missing(tmp_path: Path):
    result = evaluate_readiness(tmp_path)

    assert result.ready is False
    assert result.missing_paths == tuple(item.path for item in REQUIRED_EVIDENCE)
    assert result.blocking_debt_ids == ("TRUST-DEBT-0002", "TRUST-DEBT-0003")


def test_readiness_passes_when_required_enterprise_evidence_exists(tmp_path: Path):
    for item in REQUIRED_EVIDENCE:
        path = tmp_path / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# {item.title}\n\nEvidence placeholder for test.\n", encoding="utf-8"
        )

    result = evaluate_readiness(tmp_path)

    assert result.ready is True
    assert result.missing_paths == ()
    assert result.blocking_debt_ids == ()


def test_readiness_report_lists_control_ids_and_paths(tmp_path: Path):
    first = REQUIRED_EVIDENCE[0]
    path = tmp_path / first.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# present\n", encoding="utf-8")

    result = evaluate_readiness(tmp_path)
    report = result.to_markdown()

    assert "Customer Model Factory Readiness" in report
    assert "TRUST-THREAT-001" in report
    assert "TRUST-DATA-001" in report
    assert REQUIRED_EVIDENCE[1].path in report
    assert "ready: false" in report
