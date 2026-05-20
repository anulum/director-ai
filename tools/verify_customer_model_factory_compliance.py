# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory compliance verifier

"""Verify Customer Model Factory implementation, test, schema, and docs parity."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CustomerFactoryControl:
    """One required implementation control for the Customer Model Factory."""

    module: str
    test_path: str
    schema_paths: tuple[str, ...]
    guide_tokens: tuple[str, ...]
    public_api_doc: bool = True


@dataclass(frozen=True)
class CustomerFactoryComplianceResult:
    """Result returned by the Customer Model Factory compliance verifier."""

    checked_controls: int
    findings: tuple[str, ...]

    @property
    def ready(self) -> bool:
        """Return true when no compliance findings remain."""

        return not self.findings


CONTROL_MATRIX = (
    CustomerFactoryControl(
        module="dataset_contract",
        test_path="tests/test_customer_model_factory_dataset_contract.py",
        schema_paths=("schemas/customer-model-factory-trace.schema.json",),
        guide_tokens=("Dataset validation",),
    ),
    CustomerFactoryControl(
        module="training_manifest",
        test_path="tests/test_customer_model_factory_training_manifest.py",
        schema_paths=("schemas/customer-model-factory-training-manifest.schema.json",),
        guide_tokens=("Training manifest",),
    ),
    CustomerFactoryControl(
        module="benchmark_selection",
        test_path="tests/test_customer_model_factory_benchmark_selection.py",
        schema_paths=("schemas/customer-model-factory-selection.schema.json",),
        guide_tokens=("Benchmark selection",),
    ),
    CustomerFactoryControl(
        module="deployment_manifest",
        test_path="tests/test_customer_model_factory_deployment_manifest.py",
        schema_paths=("schemas/customer-model-factory-deployment.schema.json",),
        guide_tokens=("Deployment manifest",),
    ),
    CustomerFactoryControl(
        module="sector_extension",
        test_path="tests/test_customer_model_factory_sector_extension.py",
        schema_paths=("schemas/customer-model-factory-sector-metadata.schema.json",),
        guide_tokens=("Sector-extension boundary",),
    ),
    CustomerFactoryControl(
        module="evidence_pack",
        test_path="tests/test_customer_model_factory_evidence_pack.py",
        schema_paths=("schemas/customer-model-factory-evidence-pack.schema.json",),
        guide_tokens=("Evidence pack",),
    ),
    CustomerFactoryControl(
        module="runtime_package",
        test_path="tests/test_customer_model_factory_runtime_package.py",
        schema_paths=("schemas/customer-model-factory-runtime-package.schema.json",),
        guide_tokens=("Runtime package",),
    ),
    CustomerFactoryControl(
        module="monitoring_manifest",
        test_path="tests/test_customer_model_factory_monitoring.py",
        schema_paths=("schemas/customer-model-factory-monitoring.schema.json",),
        guide_tokens=("Monitoring manifest",),
    ),
    CustomerFactoryControl(
        module="risk_register",
        test_path="tests/test_customer_model_factory_risk_register.py",
        schema_paths=("schemas/customer-model-factory-risk-register.schema.json",),
        guide_tokens=("Risk register",),
    ),
    CustomerFactoryControl(
        module="release_gate",
        test_path="tests/test_customer_model_factory_release_gate.py",
        schema_paths=("schemas/customer-model-factory-release-gate.schema.json",),
        guide_tokens=("Release gate",),
    ),
)


def main(argv: list[str] | None = None) -> int:
    """Run the Customer Model Factory compliance verifier."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    result = evaluate_compliance(args.root)
    for finding in result.findings:
        print(finding)
    return 0 if result.ready else 1


def evaluate_compliance(root: Path) -> CustomerFactoryComplianceResult:
    """Return compliance findings for the Customer Model Factory control matrix."""

    findings: list[str] = []
    package_init = root / "src/director_ai/core/customer_model_factory/__init__.py"
    exported_symbols = _read_exported_symbols(package_init)
    guide = _read_optional(root / "docs-site/guide/customer-model-factory.md")
    api_page = _read_optional(root / "docs-site/api/customer-model-factory.md")

    for control in CONTROL_MATRIX:
        module_path = (
            root / f"src/director_ai/core/customer_model_factory/{control.module}.py"
        )
        if not module_path.is_file():
            findings.append(f"{control.module}:missing_module")
            continue
        if not (root / control.test_path).is_file():
            findings.append(f"{control.module}:missing_test")
        for schema_path in control.schema_paths:
            if not (root / schema_path).is_file():
                findings.append(f"{control.module}:missing_schema:{schema_path}")
        for token in control.guide_tokens:
            if token not in guide:
                findings.append(f"{control.module}:missing_guide_token:{token}")
        if control.public_api_doc:
            api_token = f"director_ai.core.customer_model_factory.{control.module}"
            if api_token not in api_page:
                findings.append(f"{control.module}:missing_api_doc")
        findings.extend(_public_docstring_findings(control.module, module_path))
        findings.extend(_export_findings(control.module, module_path, exported_symbols))

    return CustomerFactoryComplianceResult(
        checked_controls=len(CONTROL_MATRIX),
        findings=tuple(findings),
    )


def _read_optional(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _read_exported_symbols(path: Path) -> frozenset[str]:
    if not path.is_file():
        return frozenset()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return frozenset(_literal_strings(node.value))
    return frozenset()


def _literal_strings(node: ast.AST) -> tuple[str, ...]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return ()
    values: list[str] = []
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.append(item.value)
    return tuple(values)


def _public_docstring_findings(module: str, module_path: Path) -> tuple[str, ...]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    findings: list[str] = []
    if ast.get_docstring(tree) is None:
        findings.append(f"{module}:missing_module_docstring")
    for node in tree.body:
        if (
            isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and not node.name.startswith("_")
            and ast.get_docstring(node) is None
        ):
            findings.append(f"{module}:missing_public_docstring:{node.name}")
    return tuple(findings)


def _export_findings(
    module: str,
    module_path: Path,
    exported_symbols: frozenset[str],
) -> tuple[str, ...]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    public_symbols = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and not node.name.startswith("_")
    }
    findings = [
        f"{module}:missing_package_export:{symbol}"
        for symbol in sorted(public_symbols)
        if symbol not in exported_symbols
    ]
    return tuple(findings)


if __name__ == "__main__":
    raise SystemExit(main())
