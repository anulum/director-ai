# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — tier build hooks (free/paid wheel module slicing)
"""Build hooks that slice paid single modules between the tier wheels.

``[tool.setuptools.packages.find].exclude`` (and ``include`` on the paid
side) operates on whole PACKAGES only. The paid single modules that live
inside free packages — ``director_ai/server.py``,
``director_ai/core/verified_scorer.py``, … — need per-file handling, so the
three ``setup.py`` files plug these ``build_py`` subclasses in:

- the root (free ``director-ai``) build uses :class:`FreeTierBuildPy` to
  keep the paid modules OUT of the free wheel;
- ``packages/director-ai-pro`` and ``packages/director-ai-full`` use
  :class:`PaidTierBuildPy` to copy the same modules IN, layered into the
  ``director_ai`` namespace without the parent ``__init__.py``.

The shared module list lives in ``paid_tier_manifest.json`` next to this
file; ``tests/test_paid_tier_packaging.py`` pins the parity between the
manifest, the three pyproject files and ``MANIFEST.in``. No runtime code
changes hands here — the lazy facades in ``director_ai.core`` and
``director_ai.cli`` already degrade with the friendly "advanced tier"
error when these modules are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

from setuptools.command.build_py import build_py

_HERE = Path(__file__).resolve().parent
MANIFEST_PATH = _HERE / "paid_tier_manifest.json"
REPO_SRC = _HERE.parent / "src"


def paid_module_paths() -> frozenset[str]:
    """Return the paid single-module paths, POSIX-relative to ``src/``.

    Returns
    -------
    frozenset[str]
        Entries such as ``"director_ai/server.py"`` read from
        ``paid_tier_manifest.json``.
    """
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return frozenset(data["paid_modules"])


def paid_module_names() -> frozenset[str]:
    """Return the paid single modules as dotted module names.

    Returns
    -------
    frozenset[str]
        Entries such as ``"director_ai.server"`` derived from
        :func:`paid_module_paths`.
    """
    return frozenset(
        path[: -len(".py")].replace("/", ".") for path in paid_module_paths()
    )


class FreeTierBuildPy(build_py):
    """``build_py`` that keeps the paid single modules out of the free wheel."""

    _paid_names = paid_module_names()

    def find_package_modules(
        self, package: str, package_dir: str
    ) -> list[tuple[str, str, str]]:
        """Filter the paid modules from the modules of *package*.

        Parameters
        ----------
        package : str
            Dotted package name being built (e.g. ``"director_ai.core"``).
        package_dir : str
            Source directory of the package.

        Returns
        -------
        list[tuple[str, str, str]]
            The ``(package, module, file)`` triples setuptools would build,
            minus the entries listed in ``paid_tier_manifest.json``.
        """
        # type-ignore reason: types-setuptools leaves find_package_modules
        # unannotated, so the super() call is "untyped" to strict mypy.
        modules = super().find_package_modules(package, package_dir)  # type: ignore[no-untyped-call]
        return [
            (pkg, mod, path)
            for pkg, mod, path in modules
            if f"{pkg}.{mod}" not in self._paid_names
        ]


class PaidTierBuildPy(build_py):
    """``build_py`` that layers the paid single modules into a paid wheel."""

    def run(self) -> None:
        """Build the configured packages, then copy the paid modules in."""
        super().run()
        for rel in sorted(paid_module_paths()):
            source = REPO_SRC / rel
            target = Path(self.build_lib) / rel
            self.mkpath(str(target.parent))
            self.copy_file(str(source), str(target))
