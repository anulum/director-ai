#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - API reference consistency validator

from __future__ import annotations

import argparse
import importlib
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

REFERENCE_INDEX = Path("docs-site/api/index.md")


@dataclass(frozen=True)
class ReferenceRow:
    line_number: int
    symbol_cell: str
    module_cell: str


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_reference_rows(index_path: Path) -> list[ReferenceRow]:
    rows: list[ReferenceRow] = []
    for line_number, line in enumerate(
        index_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped.startswith("|") or set(stripped.replace("|", "").strip()) <= {
            "-"
        }:
            continue
        cells = _split_markdown_row(stripped)
        if len(cells) < 2:
            continue
        if cells[0].lower() in {
            "symbol",
            "class",
            "function",
            "interface",
            "exception",
        }:
            continue
        rows.append(ReferenceRow(line_number, cells[0], cells[1]))
    return rows


def _extract_markdown_links(cell: str) -> Iterable[str]:
    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", cell):
        yield match.group(1).strip()


def _extract_module(cell: str) -> str | None:
    match = re.search(r"`([^`]+)`", cell)
    if not match:
        return None
    return match.group(1).strip()


def _extract_symbol(cell: str) -> str | None:
    match = re.search(r"`([^`]+)`", cell)
    if not match:
        return None
    symbol = match.group(1).strip()
    return symbol.removesuffix("()")


def _normalise_link_target(raw_link: str) -> tuple[str, str]:
    split = urlsplit(raw_link)
    path = unquote(split.path)
    fragment = unquote(split.fragment)
    return path, fragment


def _slugify_heading(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\{:\s*#[^}]+\}\s*$", "", text)
    text = re.sub(r"\{#[^}]+\}\s*$", "", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


def _heading_anchors(markdown_path: Path) -> set[str]:
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            continue
        heading = match.group(2)
        explicit = re.search(r"\{:\s*#([^}\s]+)\s*\}\s*$", heading)
        if explicit is None:
            explicit = re.search(r"\{#([^}]+)\}\s*$", heading)
        base = explicit.group(1) if explicit else _slugify_heading(heading)
        if not base:
            continue
        count = seen.get(base, 0)
        seen[base] = count + 1
        anchors.add(base if count == 0 else f"{base}_{count}")
    return anchors


def _validate_markdown_link(
    *,
    root: Path,
    index_path: Path,
    line_number: int,
    raw_link: str,
) -> list[str]:
    path, fragment = _normalise_link_target(raw_link)
    if raw_link.startswith(("http://", "https://", "mailto:")):
        return []
    if not path and not fragment:
        return []

    target_path = (index_path.parent / path).resolve() if path else index_path
    try:
        target_path.relative_to(root.resolve())
    except ValueError:
        return [
            f"{REFERENCE_INDEX}:{line_number}: markdown target escapes repository {raw_link}"
        ]

    if not target_path.exists():
        display = raw_link.split("#", 1)[0] or f"#{fragment}"
        return [f"{REFERENCE_INDEX}:{line_number}: missing markdown target {display}"]

    if fragment and fragment not in _heading_anchors(target_path):
        return [
            f"{REFERENCE_INDEX}:{line_number}: missing anchor #{fragment} in {path or REFERENCE_INDEX}"
        ]

    return []


def _validate_importable_symbol(
    line_number: int, module_name: str, symbol: str
) -> list[str]:
    if not module_name.startswith("director_ai"):
        return []
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exercised through returned message
        return [f"{REFERENCE_INDEX}:{line_number}: cannot import {module_name}: {exc}"]

    if not hasattr(module, symbol):
        return [
            f"{REFERENCE_INDEX}:{line_number}: {module_name} does not expose {symbol}"
        ]
    return []


def validate_api_reference(root: Path) -> list[str]:
    root = root.resolve()
    index_path = root / REFERENCE_INDEX
    if not index_path.exists():
        return [f"{REFERENCE_INDEX}: missing API reference index"]

    errors: list[str] = []
    for row in _parse_reference_rows(index_path):
        for link in _extract_markdown_links(row.symbol_cell):
            errors.extend(
                _validate_markdown_link(
                    root=root,
                    index_path=index_path,
                    line_number=row.line_number,
                    raw_link=link,
                )
            )

        module_name = _extract_module(row.module_cell)
        symbol = _extract_symbol(row.symbol_cell)
        if module_name and symbol:
            errors.extend(
                _validate_importable_symbol(row.line_number, module_name, symbol)
            )

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="Repository root containing docs-site/api/index.md",
    )
    args = parser.parse_args(argv)

    errors = validate_api_reference(args.root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("api_reference_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
