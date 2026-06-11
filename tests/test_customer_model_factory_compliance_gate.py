# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory compliance gate tests

from __future__ import annotations

import shutil
from pathlib import Path

from tools.verify_customer_model_factory_compliance import (
    CONTROL_MATRIX,
    evaluate_compliance,
)

ROOT = Path(__file__).resolve().parents[1]


def test_compliance_gate_covers_every_customer_model_factory_phase_artifact():
    result = evaluate_compliance(ROOT)

    assert result.findings == ()
    assert {control.module for control in CONTROL_MATRIX} == {
        "sector_extension",
        "benchmark_selection",
        "dataset_contract",
        "deployment_manifest",
        "evidence_pack",
        "monitoring_manifest",
        "release_gate",
        "risk_register",
        "runtime_package",
        "training_manifest",
    }
    assert result.checked_controls == len(CONTROL_MATRIX)


def test_compliance_gate_reports_missing_test_schema_and_docs(tmp_path: Path):
    for relative_path in (
        "src/director_ai/core/customer_model_factory/dataset_contract.py",
        "src/director_ai/core/customer_model_factory/__init__.py",
        "tests/test_customer_model_factory_dataset_contract.py",
        "schemas/customer-model-factory-trace.schema.json",
        "docs-site/guide/customer-model-factory.md",
        "docs-site/api/customer-model-factory.md",
    ):
        source = ROOT / relative_path
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    (tmp_path / "tests/test_customer_model_factory_dataset_contract.py").unlink()
    (tmp_path / "schemas/customer-model-factory-trace.schema.json").unlink()
    (tmp_path / "docs-site/api/customer-model-factory.md").write_text(
        "# API\n\n",
        encoding="utf-8",
    )

    result = evaluate_compliance(tmp_path)

    assert "dataset_contract:missing_test" in result.findings
    assert (
        "dataset_contract:missing_schema:schemas/customer-model-factory-trace.schema.json"
        in result.findings
    )
    assert "dataset_contract:missing_api_doc" in result.findings


def test_compliance_gate_rejects_undocumented_public_symbols(tmp_path: Path):
    module_path = (
        tmp_path / "src/director_ai/core/customer_model_factory/dataset_contract.py"
    )
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        "def undocumented_public_function():\n    return None\n",
        encoding="utf-8",
    )

    result = evaluate_compliance(tmp_path)

    assert (
        "dataset_contract:missing_public_docstring:undocumented_public_function"
        in result.findings
    )
