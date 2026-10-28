# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — source-bound native strict Python typing snapshots
"""Capture native MyPy errors from all maintained Python and stub files."""

from __future__ import annotations

import configparser
import hashlib
import io
import json
import math
import os
import re
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_contracts import (
    SCHEMA,
    Finding,
    Snapshot,
    parse_snapshot,
    snapshot_payload,
)
from tools.quality_ratchet_inventory import (
    GENERATED_PYTHON,
    PYTHON_ROOTS,
    hash_sources,
    python_sources,
    source_path,
)
from tools.quality_ratchet_typing_protocol import (
    parse_typing_environment,
    parse_typing_result,
)

TYPING_CONFIGURATIONS = (
    "pyproject.toml",
    "requirements/ci-dev.txt",
    "requirements/ci-types.txt",
)
GENERATED_MODULES = frozenset(
    {
        "director_ai.director_pb2",
        "director_ai.director_pb2_grpc",
        "director_ai.proto.director.v1.director_pb2",
        "director_ai.proto.director.v1.director_pb2_grpc",
    }
)
TYPING_FLAGS = (
    "--strict",
    "--python-version=3.11",
    "--no-incremental",
    "--output=json",
    "--no-error-summary",
    "--no-color-output",
)
ENVIRONMENT_CODE = (
    "import importlib.metadata as m,json,sys; "
    "print(json.dumps({'python':list(sys.version_info[:3]),"
    "'packages':sorted(set((d.metadata['Name'],d.version) for d in m.distributions()))}))"
)
STRICT_OPTIONS = {
    "warn_unused_configs": True,
    "disallow_any_generics": True,
    "disallow_subclassing_any": True,
    "disallow_untyped_calls": True,
    "disallow_untyped_defs": True,
    "disallow_incomplete_defs": True,
    "check_untyped_defs": True,
    "disallow_untyped_decorators": True,
    "warn_redundant_casts": True,
    "warn_unused_ignores": True,
    "warn_return_any": True,
    "implicit_reexport": False,
    "strict_equality": True,
    "strict_bytes": True,
    "extra_checks": True,
}


def _option(value: object) -> str:
    """Encode an original native configuration value without implicit coercions."""
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, str) and not any(c in value for c in "\r\n\x00"):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return ",".join(_option(item) for item in value)
    raise ValueError("Unsupported original MyPy configuration value")


def _configuration(root: Path) -> str:
    """Retain original settings while removing maintained-module error masking."""
    original = tomllib.loads(source_path(root, "pyproject.toml").read_text())
    settings = object_mapping(object_mapping(original.get("tool")).get("mypy"))
    overrides = settings.pop("overrides", [])
    if settings.get("strict") is not True:
        raise ValueError("Original MyPy configuration does not require strict typing")
    settings["ignore_missing_imports"] = False
    settings["ignore_errors"] = False
    settings["follow_imports"] = "normal"
    settings["disable_error_code"] = []
    settings.update(STRICT_OPTIONS)
    config = configparser.ConfigParser(interpolation=None)
    config["mypy"] = {name: _option(value) for name, value in settings.items()}
    if not isinstance(overrides, list):
        raise ValueError("Invalid original MyPy override list")
    for raw in overrides:
        item = object_mapping(raw)
        modules = item.pop("module", None)
        names = [modules] if isinstance(modules, str) else modules
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name for name in names)
        ):
            raise ValueError("Invalid original MyPy module identity")
        module_names = [_option(name) for name in names]
        if any(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.*]*|\*", name) is None
            for name in module_names
        ):
            raise ValueError("Invalid original MyPy module pattern")
        generated = set(module_names).issubset(GENERATED_MODULES)
        if not generated:
            item["ignore_errors"] = False
            item["disable_error_code"] = []
            for name, value in STRICT_OPTIONS.items():
                if name in item:
                    item[name] = value
            if "strict" in item:
                item["strict"] = True
        if item.get("follow_imports") in ("skip", "silent"):
            item["follow_imports"] = "normal"
        section = "mypy-" + ",".join(module_names)
        if section in config:
            raise ValueError("Repeated original MyPy override identity")
        config[section] = {name: _option(value) for name, value in item.items()}
    output = io.StringIO()
    config.write(output)
    return output.getvalue()


def _run(
    argv: list[str], root: Path, deadline: float, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run a native process within the remaining whole-analysis deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("Native typing analysis exhausted its whole deadline")
    return subprocess.run(
        argv,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=remaining,
    )


def _environment(
    root: Path, python: str, deadline: float, environment: dict[str, str]
) -> object:
    """Capture actual compiler and interpreter identity within the whole deadline."""
    version = _run(
        [python, "-B", "-m", "mypy", "--version"], root, deadline, environment
    )
    installed = _run(
        [python, "-I", "-B", "-c", ENVIRONMENT_CODE], root, deadline, environment
    )
    return parse_typing_environment(version, installed)


def capture_python_typing(
    root: Path, python: str, *, timeout: float = 120.0
) -> Snapshot:
    """Analyse every maintained Python/stub file using the actual strict compiler.

    Parameters
    ----------
    root : Path
        Actual selected Git source repository, including pending source additions.
    python : str
        Interpreter with the pinned MyPy and actual declared typing environment.
    timeout : float
        Positive finite deadline for the entire native analysis, in seconds.

    Returns
    -------
    Snapshot
        Original error identities bound to complete Python/stub source bytes.
        Notebook typing needs its separate native profile; this profile does not
        claim to type-check notebook JSON or the whole repository.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If native process, configuration, syntax, source or environment is
        unqualified, incomplete or changed during analysis.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Native quality deadline must be positive and finite")
    deadline = time.monotonic() + timeout
    root = root.resolve(strict=True)
    inventory = python_sources(root)
    paths = tuple(path for path in inventory if Path(path).suffix != ".ipynb")
    if not paths:
        raise ValueError("Native typing source inventory is empty")
    sources = hash_sources(root, paths)
    hashes = hash_sources(root, TYPING_CONFIGURATIONS)
    configuration = _configuration(root)
    environment = dict(os.environ)
    environment["MYPYPATH"] = os.pathsep.join(
        str(root / path) for path in ("src", "packages")
    )
    installed = _environment(root, python, deadline, environment)
    identity = hashlib.sha256(
        json.dumps(
            {
                "profile": "python-types",
                "snapshot_schema": SCHEMA,
                "flags": TYPING_FLAGS,
                "configuration": configuration,
                "configuration_hashes": hashes,
                "import_roots": ["src", "packages"],
                "environment": installed,
                "source_roots": PYTHON_ROOTS,
                "generated_paths": GENERATED_PYTHON,
                "generated_modules": sorted(GENERATED_MODULES),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    main = tuple(path for path in paths if Path(path).name != "setup.py")
    setups = tuple((path,) for path in paths if Path(path).name == "setup.py")
    cohorts = ((main,) if main else ()) + setups
    findings: set[Finding] = set()
    advisories: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="quality-types-") as directory:
        temporary = Path(directory)
        config = temporary / "mypy.ini"
        config.write_text(configuration)
        for cohort in cohorts:
            result = _run(
                [
                    python,
                    "-B",
                    "-m",
                    "mypy",
                    *TYPING_FLAGS,
                    "--config-file",
                    str(config),
                    "--cache-dir",
                    str(temporary / "cache"),
                    "--",
                    *cohort,
                ],
                root,
                deadline,
                environment,
            )
            original, information = parse_typing_result(
                result, root, sources, identity, config, configuration
            )
            findings.update(original)
            advisories.update(information)
    if (
        python_sources(root) != inventory
        or hash_sources(root, paths) != sources
        or hash_sources(root, TYPING_CONFIGURATIONS) != hashes
        or _environment(root, python, deadline, environment) != installed
    ):
        raise ValueError(
            "Typing source, configuration or environment changed during analysis"
        )
    return parse_snapshot(
        snapshot_payload(
            Snapshot(
                "python-types",
                identity,
                sources,
                tuple(findings),
                True,
                tuple(sorted(advisories)),
            )
        )
    )
