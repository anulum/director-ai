# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — static Go API documentation builder
"""Render every Go package's ``go doc -all`` output as static HTML."""

from __future__ import annotations

import argparse
import html
import re
import subprocess
from pathlib import Path


def _run_go(module_dir: Path, *args: str) -> str:
    """Run one Go command in the module and return its standard output."""
    completed = subprocess.run(
        ("go", *args),
        cwd=module_dir,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout


def _safe_name(package: str) -> str:
    """Map an import path to a deterministic, filesystem-safe filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", package).strip("-") or "package"


def build(module_dir: Path, output_dir: Path) -> int:
    """Build static documentation for all packages below *module_dir*."""
    module_dir = module_dir.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    packages = sorted(filter(None, _run_go(module_dir, "list", "./...").splitlines()))
    if not packages:
        raise RuntimeError(f"go list returned no packages below {module_dir}")

    links: list[str] = []
    for package in packages:
        filename = f"{_safe_name(package)}.html"
        documentation = _run_go(module_dir, "doc", "-all", package)
        body = html.escape(documentation)
        page = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>{html.escape(package)} — Go API</title>"
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<style>body{font:16px/1.5 system-ui;max-width:1000px;margin:2rem auto;"
            "padding:0 1rem;color:#202124}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
            "a{color:#5e35b1}</style></head><body>"
            f'<p><a href="index.html">← All Go packages</a></p><h1>{html.escape(package)}</h1>'
            f"<pre>{body}</pre></body></html>"
        )
        (output_dir / filename).write_text(page, encoding="utf-8")
        links.append(f'<li><a href="{filename}">{html.escape(package)}</a></li>')

    index = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>Director-AI Go API</title>"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<style>body{font:16px/1.5 system-ui;max-width:900px;margin:2rem auto;"
        "padding:0 1rem}a{color:#5e35b1}</style></head><body>"
        "<h1>Director-AI Go API</h1><p>Generated with <code>go doc -all</code>.</p><ul>"
        + "".join(links)
        + "</ul></body></html>"
    )
    (output_dir / "index.html").write_text(index, encoding="utf-8")
    return len(packages)


def main() -> int:
    """Parse command-line arguments and build the package reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = build(args.module_dir, args.output)
    print(f"generated Go documentation for {count} packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
