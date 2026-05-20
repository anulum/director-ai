# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory docs freeze verifier

"""Verify Customer Model Factory public docs, schemas, examples, and tools align."""

from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED_PATHS = (
    "README.md",
    "docs-site/guide/customer-model-factory.md",
    "docs-site/api/customer-model-factory.md",
    "examples/customer_model_factory_runtime.py",
    "examples/customer_model_factory_rest_payload.py",
    "tools/assemble_customer_model_factory_release.py",
    "tools/export_customer_model_factory_evidence_pack.py",
    "tools/generate_customer_model_factory_fixture.py",
    "tools/verify_public_sector_boundary.py",
    "tools/verify_customer_model_factory_compliance.py",
    "schemas/customer-model-factory-trace.schema.json",
    "schemas/customer-model-factory-training-manifest.schema.json",
    "schemas/customer-model-factory-selection.schema.json",
    "schemas/customer-model-factory-deployment.schema.json",
    "schemas/customer-model-factory-sector-metadata.schema.json",
    "schemas/customer-model-factory-evidence-pack.schema.json",
    "schemas/customer-model-factory-runtime-package.schema.json",
    "schemas/customer-model-factory-monitoring.schema.json",
    "schemas/customer-model-factory-risk-register.schema.json",
    "schemas/customer-model-factory-release-gate.schema.json",
)

README_TOKENS = (
    "Customer Model Factory",
    "examples/customer_model_factory_runtime.py",
    "examples/customer_model_factory_rest_payload.py",
    "schemas/customer-model-factory-runtime-package.schema.json",
    "zero silent unsafe passes",
    "Customer-specific accuracy claims require package-specific benchmark evidence",
)

GUIDE_TOKENS = (
    "Dataset validation",
    "Training manifest",
    "Benchmark selection",
    "Deployment manifest",
    "Sector-extension boundary",
    "Evidence pack",
    "Runtime package",
    "Monitoring manifest",
    "Risk register",
    "Release gate",
    "tools/generate_customer_model_factory_fixture.py",
    "tools/assemble_customer_model_factory_release.py",
    "tools/verify_customer_model_factory_compliance.py",
    "tools/verify_customer_model_factory_docs_freeze.py",
)

API_MODULE_TOKENS = (
    "director_ai.core.customer_model_factory.dataset_contract",
    "director_ai.core.customer_model_factory.training_manifest",
    "director_ai.core.customer_model_factory.benchmark_selection",
    "director_ai.core.customer_model_factory.deployment_manifest",
    "director_ai.core.customer_model_factory.sector_extension",
    "director_ai.core.customer_model_factory.evidence_pack",
    "director_ai.core.customer_model_factory.runtime_package",
    "director_ai.core.customer_model_factory.monitoring_manifest",
    "director_ai.core.customer_model_factory.risk_register",
    "director_ai.core.customer_model_factory.release_gate",
)


def main(argv: list[str] | None = None) -> int:
    """Run the docs freeze verifier."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    findings = evaluate_docs_freeze(args.root)
    for finding in findings:
        print(finding)
    return 0 if not findings else 1


def evaluate_docs_freeze(root: Path) -> tuple[str, ...]:
    """Return docs-freeze findings for missing public surface pieces."""

    findings: list[str] = []
    for relative_path in REQUIRED_PATHS:
        if not (root / relative_path).is_file():
            findings.append(f"missing:{relative_path}")
    readme_path = root / "README.md"
    if readme_path.is_file():
        readme = readme_path.read_text(encoding="utf-8")
        for token in README_TOKENS:
            if token not in readme:
                findings.append(f"readme_missing:{token}")
        if "100%" + " accuracy" in readme:
            findings.append("readme_unscoped_accuracy_claim")
    guide_path = root / "docs-site/guide/customer-model-factory.md"
    if guide_path.is_file():
        guide = guide_path.read_text(encoding="utf-8")
        for token in GUIDE_TOKENS:
            if token not in guide:
                findings.append(f"guide_missing:{token}")
    api_path = root / "docs-site/api/customer-model-factory.md"
    if api_path.is_file():
        api_page = api_path.read_text(encoding="utf-8")
        for token in API_MODULE_TOKENS:
            if token not in api_page:
                findings.append(f"api_missing:{token}")
    return tuple(findings)


if __name__ == "__main__":
    raise SystemExit(main())
