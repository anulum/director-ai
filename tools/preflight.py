#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-push preflight gate — mirrors CI checks locally.

Gates (must match ci.yml):
  1. ruff-format        — src/ tests/ examples/
  2. ruff-check         — src/ tests/ examples/
  3. mojibake           — src/ tools/ tests/
  4. capability-matrix  — generated matrix current + gap ratchet (WCC-1)
  5. version-sync       — root, tier, citation, and archive metadata agree
  6. mypy               — src/director_ai/
  7. bandit             — src/director_ai/
  8. spdx-guard         — all .py files in src/ tests/
  9. pytest             — tests/ with the CI coverage gate (97%)
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

SPDX_DIRS = ["src", "tests"]
SPDX_EXTS = {".py"}
SPDX_MARKER = "SPDX-License-Identifier"
# Path fragments (relative to repo root) whose files are generated
# by external tooling (protoc etc.) and therefore cannot be
# hand-edited to carry our SPDX header. The ruff ``exclude`` list
# in ``pyproject.toml`` covers the same set.
SPDX_EXCLUDE_FRAGMENTS = ("src/director_ai/proto/",)

GateCommand = list[str]
Gate = tuple[str, GateCommand | None]

GATES: list[Gate] = [
    (
        "ruff-format",
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            "src/",
            "tests/",
            "examples/",
            "tools/",
        ],
    ),
    (
        "ruff-check",
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "src/",
            "tests/",
            "examples/",
            "tools/",
        ],
    ),
    (
        "mojibake-guard",
        [
            sys.executable,
            "tools/check_mojibake.py",
            "--root",
            ".",
        ],
    ),
    (
        "capability-matrix",
        [
            sys.executable,
            "tools/capability_matrix.py",
            "--check",
        ],
    ),
    ("version-sync", None),
    (
        "bandit",
        [
            sys.executable,
            "-m",
            "bandit",
            "-r",
            "src/director_ai/",
            "-c",
            "pyproject.toml",
            "-q",
        ],
    ),
    ("spdx-guard", None),
    (
        "mypy",
        [
            sys.executable,
            "-m",
            "mypy",
            "src/director_ai/",
            "--ignore-missing-imports",
            "--check-untyped-defs",
        ],
    ),
    (
        "pytest",
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--cov=director_ai",
            "--cov-report=term-missing",
            "--cov-fail-under=97",
        ],
    ),
]

SLOW_GATES = {"mypy", "pytest"}
WARN_ONLY = {"spdx-guard"}


def check_version_sync() -> bool:
    """Return True when all release-version declarations are identical."""
    root = pathlib.Path()
    try:
        import tomllib

        with open(root / "pyproject.toml", "rb") as f:
            v_toml = tomllib.load(f)["project"]["version"]
    except Exception:
        try:
            text = (root / "pyproject.toml").read_text()
            m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
            v_toml = m.group(1) if m else ""
        except Exception:
            print("  Cannot read pyproject.toml version")
            return False

    tier_versions: dict[str, str] = {}
    try:
        for tier in ("lite", "pro", "full"):
            tier_path = root / "packages" / f"director-ai-{tier}" / "pyproject.toml"
            with tier_path.open("rb") as file:
                tier_versions[tier] = tomllib.load(file)["project"]["version"]
    except Exception:
        print("  Cannot read tier package version")
        return False

    try:
        init = (root / "src" / "director_ai" / "__init__.py").read_text()
        m = re.search(r'__version__\s*=\s*"([^"]+)"', init)
        v_init = m.group(1) if m else ""
    except Exception:
        print("  Cannot read __init__.py version")
        return False

    try:
        cff = (root / "CITATION.cff").read_text()
        m = re.search(r'^version:\s*"?([^"\n]+)"?', cff, re.MULTILINE)
        v_cff = m.group(1).strip() if m else ""
    except Exception:
        print("  Cannot read CITATION.cff version")
        return False

    try:
        zenodo = json.loads((root / ".zenodo.json").read_text())
        v_zenodo = str(zenodo["version"])
    except Exception:
        print("  Cannot read .zenodo.json version")
        return False

    try:
        lite_init = (
            root
            / "packages"
            / "director-ai-lite"
            / "src"
            / "director_ai_lite"
            / "__init__.py"
        ).read_text()
        m = re.search(r'__version__\s*=\s*"([^"]+)"', lite_init)
        v_lite_init = m.group(1) if m else ""
    except Exception:
        print("  Cannot read Director-Lite __init__.py version")
        return False

    declared = {
        "pyproject.toml": v_toml,
        "__init__.py": v_init,
        "CITATION.cff": v_cff,
        ".zenodo.json": v_zenodo,
        "lite pyproject": tier_versions["lite"],
        "lite __init__": v_lite_init,
        "pro pyproject": tier_versions["pro"],
        "full pyproject": tier_versions["full"],
    }
    print("  " + "  ".join(f"{name}={version}" for name, version in declared.items()))
    if set(declared.values()) == {v_toml}:
        return True
    print("  Version mismatch!")
    return False


def check_spdx() -> bool:
    """Return True when all hand-maintained Python files carry SPDX headers."""
    missing = []
    for d in SPDX_DIRS:
        root = pathlib.Path(d)
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix not in SPDX_EXTS or "__pycache__" in p.parts:
                continue
            posix = p.as_posix()
            if any(fragment in posix for fragment in SPDX_EXCLUDE_FRAGMENTS):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")[:2048]
            except OSError:
                continue
            if SPDX_MARKER not in text:
                missing.append(str(p))
    if missing:
        print("Missing SPDX headers:")
        for f in missing:
            print(f"  {f}")
        return False
    return True


def run_gate(name: str, cmd: GateCommand | None) -> bool:
    """Execute one named preflight gate and return its pass/fail state."""
    print(f"\n{'=' * 60}")
    print(f"  GATE: {name}")
    print(f"{'=' * 60}")
    if cmd is None:
        if name == "version-sync":
            ok = check_version_sync()
        elif name == "spdx-guard":
            ok = check_spdx()
        else:
            ok = False
    else:
        ok = subprocess.run(cmd).returncode == 0
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    return ok


def main() -> int:
    """Run the configured preflight gates and return a process exit code."""
    parser = argparse.ArgumentParser(description="Director-AI preflight checks")
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip pytest and mypy (fast lint-only mode)",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Same as default (tests always include coverage)",
    )
    args = parser.parse_args()

    gates = list(GATES)
    if args.no_tests:
        gates = [(n, c) for n, c in gates if n not in SLOW_GATES]

    passed, failed, warned = [], [], []
    for name, cmd in gates:
        if run_gate(name, cmd):
            passed.append(name)
        elif name in WARN_ONLY:
            warned.append(name)
        else:
            failed.append(name)

    print(f"\n{'=' * 60}")
    print(
        f"  PREFLIGHT: {len(passed)} passed, {len(failed)} failed, {len(warned)} warned",
    )
    if warned:
        print(f"  WARNED: {', '.join(warned)}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
