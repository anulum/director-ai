# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — model lifecycle licence boundary tests

"""The kernel loads only Apache-2.0 repository code and the standard library.

An AGPL or BUSL host imports this package, so importing it must not load a
module under another licence, nor ``torch``, ``transformers`` or
``safetensors``. The probe runs as a plain string in a clean subprocess,
isolated (``-I``) and without the ``site`` hooks (``-S``), and reports what
the import loaded.
"""

from __future__ import annotations

import ast
import json
import subprocess  # nosec B404 - runs the current interpreter on a fixed probe
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
PACKAGE = SRC / "director_ai" / "model_lifecycle"
# REUSE-IgnoreStart
APACHE_HEADER = "# SPDX-License-Identifier: Apache-2.0"
# REUSE-IgnoreEnd
FOREIGN_PACKAGES = frozenset(
    {
        "director_class_ai",
        "synapse_channel",
        "synapse_fleet",
        "rigor_foundry",
        "remanentia",
        "scpn_control",
        "scpn_fusion",
        "sc_neurocore",
        "torch",
        "transformers",
        "safetensors",
    }
)

_PROBE = """
import json
import sys

src = sys.argv[1]
sys.path.insert(0, src)
import director_ai.model_lifecycle

headers = {}
for name, module in sorted(sys.modules.items()):
    path = getattr(module, "__file__", None) or ""
    if path.startswith(src):
        with open(path, encoding="utf-8") as handle:
            headers[name] = handle.readline().rstrip()
outside = sorted(
    {name.split(".")[0] for name in sys.modules}
    - set(sys.stdlib_module_names)
    - {"director_ai", "__main__"}
)
print(json.dumps({"headers": headers, "outside": [n for n in outside if not n.startswith("_")]}))
"""


def _probe(tmp_path: Path) -> dict[str, object]:
    """Import the package in a clean subprocess and return its report."""
    completed = subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, "-I", "-S", "-c", _PROBE, str(SRC)],
        capture_output=True,
        check=True,
        cwd=tmp_path,
        text=True,
        timeout=120,
    )
    report = json.loads(completed.stdout)
    assert isinstance(report, dict)
    return report


def _package_modules() -> set[str]:
    """Return the dotted names of every module file in the package."""
    names = {
        f"director_ai.model_lifecycle.{path.stem}" for path in PACKAGE.glob("*.py")
    }
    names.discard("director_ai.model_lifecycle.__init__")
    return names | {"director_ai.model_lifecycle"}


def test_import_loads_only_apache_repository_modules(tmp_path: Path) -> None:
    """Every repository module the import loads is Apache-2.0 on line 1."""
    report = _probe(tmp_path)
    headers = report["headers"]
    assert isinstance(headers, dict)
    assert _package_modules() <= set(headers)
    assert {
        name: header for name, header in headers.items() if header != APACHE_HEADER
    } == {}
    assert not any(name.startswith("director_ai.experimental") for name in headers)


def test_import_loads_no_third_party_package(tmp_path: Path) -> None:
    """Only the standard library is loaded besides the repository."""
    assert _probe(tmp_path)["outside"] == []


def test_package_sources_import_only_the_standard_library() -> None:
    """Every absolute import in the package names a standard-library module."""
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                assert top not in FOREIGN_PACKAGES, f"{path.name} imports {name}"
                assert top in sys.stdlib_module_names, f"{path.name} imports {name}"


def test_package_sources_carry_the_apache_header() -> None:
    """Every package file declares Apache-2.0 on its first line."""
    for path in sorted(PACKAGE.glob("*.py")):
        assert path.read_text(encoding="utf-8").splitlines()[0] == APACHE_HEADER, (
            path.name
        )
