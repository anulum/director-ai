# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""director-ai doctor and license CLI verification commands."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    pass


def _check_optional_module(module_name: str) -> tuple[bool, str]:
    import importlib
    import importlib.util

    if importlib.util.find_spec(module_name) is None:
        return False, "not installed"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return False, f"import failed: {exc}"
    return True, str(getattr(module, "__version__", "installed"))


def _stack_status() -> list[tuple[str, bool, str]]:
    import importlib.util
    import shutil

    return [
        ("Python-only core", True, "supported default"),
        (
            "Rust kernel",
            importlib.util.find_spec("backfire_kernel") is not None,
            "optional backfire_kernel",
        ),
        (
            "Docker Compose",
            shutil.which("docker") is not None,
            "optional quickstart runtime",
        ),
        (
            "Go gateway",
            shutil.which("director-ai-gateway") is not None,
            "optional advanced gateway",
        ),
        (
            "Julia tuner",
            shutil.which("julia") is not None,
            "optional research tuner",
        ),
        (
            "Lean verifier",
            shutil.which("lean") is not None or shutil.which("lake") is not None,
            "optional formal verifier",
        ),
    ]


def _stack_warnings(checks: list[tuple[str, bool, str]]) -> list[str]:
    from director_ai.core.config import DirectorConfig

    available = {name: ok for name, ok, _ in checks}
    warnings: list[str] = []
    try:
        cfg = DirectorConfig.from_env()
    except ValueError as exc:
        return [f"Invalid DIRECTOR_* configuration: {exc}"]

    deps = {
        "torch": _check_optional_module("torch")[0],
        "transformers": _check_optional_module("transformers")[0],
        "onnxruntime": _check_optional_module("onnxruntime")[0],
        "chromadb": _check_optional_module("chromadb")[0],
    }
    if cfg.use_nli and not (deps["torch"] and deps["transformers"]):
        warnings.append("DIRECTOR_USE_NLI=true but torch/transformers are missing.")
    if cfg.scorer_backend == "onnx" and not deps["onnxruntime"]:
        warnings.append("DIRECTOR_SCORER_BACKEND=onnx but onnxruntime is missing.")
    if cfg.scorer_backend == "onnx" and not cfg.onnx_path:
        warnings.append("DIRECTOR_SCORER_BACKEND=onnx but DIRECTOR_ONNX_PATH is empty.")
    if cfg.scorer_backend == "rust" and not available["Rust kernel"]:
        warnings.append("DIRECTOR_SCORER_BACKEND=rust but backfire_kernel is missing.")
    if cfg.vector_backend == "chroma" and not deps["chromadb"]:
        warnings.append("DIRECTOR_VECTOR_BACKEND=chroma but chromadb is missing.")
    if hasattr(cfg, "model_revision_health"):
        revision_health = cast(Mapping[str, Any], cfg.model_revision_health())
        if not revision_health.get("ok", True):
            revision_checks = cast(
                Mapping[str, Mapping[str, Any]],
                revision_health.get("checks", {}),
            )
            for label, check in revision_checks.items():
                if check.get("status") == "error":
                    warnings.append(
                        "Model revision health failed for "
                        f"{label}: {check.get('detail', 'unknown error')}"
                    )
    return warnings


def _cmd_doctor(args: list[str]) -> None:
    """Check runtime dependencies and print readiness summary."""
    if _is_help_request(args):
        _print_doctor_help()
        return

    import platform

    import director_ai

    checks: list[tuple[str, bool, str]] = []

    # Python version
    py_ver = platform.python_version()
    py_ok = tuple(int(x) for x in py_ver.split(".")[:2]) >= (3, 11)
    checks.append(("Python >= 3.11", py_ok, py_ver))

    # torch
    torch_ok, torch_detail = _check_optional_module("torch")
    if torch_ok:
        import torch

        torch_detail = f"{torch.__version__} (CUDA: {torch.cuda.is_available()})"
    checks.append(("torch", torch_ok, torch_detail))

    # transformers
    transformers_ok, transformers_detail = _check_optional_module("transformers")
    checks.append(("transformers", transformers_ok, transformers_detail))

    # NLI model availability
    try:
        from director_ai.core.scoring.nli import nli_available

        avail = nli_available()
        detail = "torch+transformers" if avail else "missing deps"
        checks.append(("NLI model ready", avail, detail))
    except Exception as exc:
        checks.append(("NLI model ready", False, str(exc)))

    # onnxruntime
    onnx_ok, onnx_detail = _check_optional_module("onnxruntime")
    if onnx_ok:
        import onnxruntime as ort

        provs = ort.get_available_providers()
        onnx_detail = f"{ort.__version__} ({', '.join(provs)})"
    checks.append(("onnxruntime", onnx_ok, onnx_detail))

    # chromadb
    chroma_ok, chroma_detail = _check_optional_module("chromadb")
    checks.append(("chromadb", chroma_ok, chroma_detail))

    # sentence_transformers
    st_ok, st_detail = _check_optional_module("sentence_transformers")
    checks.append(("sentence-transformers", st_ok, st_detail))

    # slowapi
    slowapi_ok, slowapi_detail = _check_optional_module("slowapi")
    checks.append(("slowapi", slowapi_ok, slowapi_detail))

    # grpcio
    grpc_ok, grpc_detail = _check_optional_module("grpc")
    checks.append(("grpcio", grpc_ok, grpc_detail))

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    stack = _stack_status()
    warnings = _stack_warnings(stack)

    print(f"director-ai {director_ai.__version__} — dependency check\n")
    for name, ok, detail in checks:
        mark = "+" if ok else "-"
        print(f"  [{mark}] {name}: {detail}")
    print(f"\n{passed}/{total} checks passed")
    print("\nRuntime stack:")
    for name, ok, detail in stack:
        mark = "+" if ok else "-"
        print(f"  [{mark}] {name}: {detail}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")


def _cmd_license(args: list[str]) -> None:
    """License management: status, generate, validate."""
    if _is_help_request(args):
        _print_license_help()
        return

    from director_ai.core.license import generate_license, load_license, validate_file

    if not args or args[0] == "status":
        info = load_license()
        print(f"Tier:     {info.tier}")
        print(f"Valid:    {info.valid}")
        print(f"Licensee: {info.licensee or '(community)'}")
        if info.expires:
            print(f"Expires:  {info.expires}")
        if info.key:
            print(f"Key:      {info.key[:20]}...")
        print(f"Message:  {info.message}")
        return

    if args[0] == "generate":
        import os

        admin_key = os.environ.get("DIRECTOR_ADMIN_KEY", "")
        if not admin_key:
            print(
                "Error: DIRECTOR_ADMIN_KEY environment variable required for license generation."
            )
            print("This command is for license administrators only.")
            sys.exit(1)

        import argparse

        p = argparse.ArgumentParser(prog="director-ai license generate")
        p.add_argument(
            "--tier", required=True, choices=["indie", "pro", "enterprise", "trial"]
        )
        p.add_argument("--licensee", required=True)
        p.add_argument("--email", required=True)
        p.add_argument("--days", type=int, default=365)
        p.add_argument("--deployments", type=int, default=1)
        p.add_argument("--output", default="license.json")
        parsed = p.parse_args(args[1:])

        import json
        from pathlib import Path

        try:
            data = generate_license(
                tier=parsed.tier,
                licensee=parsed.licensee,
                email=parsed.email,
                days=parsed.days,
                deployments=parsed.deployments,
            )
        except RuntimeError as exc:  # missing signing key / legacy opt-in
            print(f"Error: {exc}")
            sys.exit(1)
        Path(parsed.output).write_text(json.dumps(data, indent=2) + "\n")
        print(f"License generated: {parsed.output}")
        print(f"Key: {data['key']}")
        print(f"Tier: {data['tier']}")
        print(f"Licensee: {data['licensee']}")
        print(f"Expires: {data['expires']}")
        return

    if args[0] == "validate":
        if len(args) < 2:
            print("Usage: director-ai license validate <path>")
            sys.exit(1)
        info = validate_file(args[1])
        print(f"Valid:    {info.valid}")
        print(f"Tier:     {info.tier}")
        print(f"Licensee: {info.licensee}")
        print(f"Message:  {info.message}")
        sys.exit(0 if info.valid else 1)

    if args[0] in ("polar-env", "env"):
        import json

        from director_ai.core.polar_license import validate_polar_deployment_env

        report = validate_polar_deployment_env()
        if "--json" in args[1:]:
            print(
                json.dumps(
                    {
                        "ready": report.ready,
                        "errors": report.errors,
                        "warnings": report.warnings,
                    },
                    sort_keys=True,
                )
            )
            sys.exit(0 if report.ready else 1)
        print(f"Ready:    {report.ready}")
        if report.errors:
            print("Errors:")
            for item in report.errors:
                print(f"  - {item}")
        if report.warnings:
            print("Warnings:")
            for item in report.warnings:
                print(f"  - {item}")
        sys.exit(0 if report.ready else 1)

    print(f"Unknown license subcommand: {args[0]}")
    _print_license_help()
    sys.exit(1)


def _is_help_request(args: list[str]) -> bool:
    """Return whether a command received an explicit help token."""
    return bool(args) and args[0] in ("-h", "--help", "help")


def _print_doctor_help() -> None:
    """Print doctor command usage without importing optional runtimes."""
    print(
        "Usage: director-ai doctor\n"
        "\n"
        "Check runtime dependencies, stack readiness, and model revision health.\n"
    )


def _print_license_help() -> None:
    """Print license command usage without reading licence files or env secrets."""
    print(
        "Usage: director-ai license [status|generate|validate|polar-env]\n"
        "\n"
        "Subcommands:\n"
        "  status                         Show loaded licence metadata\n"
        "  generate [options]             Generate a licence with DIRECTOR_ADMIN_KEY\n"
        "  validate <path>                Validate a licence file\n"
        "  polar-env [--json]             Check Polar deployment environment\n"
    )
