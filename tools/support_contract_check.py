#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — the CI gate of the production support contract
"""Check the production support contract against the source tree.

``--check`` refuses:

- a contract that does not classify every export and command exactly once,
  names an unknown one, a subcommand its dispatcher lacks or an unknown
  extra;
- an import that is not a module of the package, or a route no decorator
  declares;
- a production default that no row owns, a stale one, a streaming setting
  turned on, or a weaker model-backed NLI requirement;
- a stale generated table.

``--write`` regenerates the table. ``--receipts DIR … --tree TREE`` checks
the installed probe receipts of the package job against the Git tree they
must have tested, so availability is never kept by hand.
Run it with ``PYTHONPATH=src``.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

from director_ai.support_contract import SupportContract, contract_refusals
from director_ai.support_contract_catalog import CONTRACT
from tools.support_contract_inventory import Inventory, read_inventory
from tools.support_contract_receipts import check_receipts
from tools.support_contract_render import (
    BLOCK_END,
    BLOCK_START,
    DOC_BLOCKS,
    GENERATED_PATHS,
    render_block,
    render_table,
)

NLI_REQUIREMENTS = (
    "coherence_require_model_backed_nli",
    "injection_require_model_backed_nli",
    "use_nli",
)
"""Settings the production profile must keep true; a weaker one is refused."""
QUICKSTART_NLI_REQUIREMENTS = NLI_REQUIREMENTS[:2]
"""The subset the generated quickstart must also keep true."""
SUPPORT_CLAIM = re.compile(
    r"supported[- ]for[- ]production|production[- ]qualified", re.I
)
"""A support claim that only the generated block may make on a public page."""


def _module_exists(root: Path, module: str) -> bool:
    """Return whether a dotted module of the package exists in the tree."""
    base = root / "src" / Path(*module.split("."))
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def surface_refusals(
    contract: SupportContract, inventory: Inventory, root: Path
) -> tuple[str, ...]:
    """Refuse an import that does not exist or a route that is not declared.

    Parameters
    ----------
    contract : SupportContract
        The contract.
    inventory : Inventory
        The tree's inventory.
    root : Path
        Repository root.

    Returns
    -------
    tuple[str, ...]
        The refusals.
    """
    refusals: list[str] = []
    for operation in contract.operations:
        refusals.extend(
            f"{operation.identifier}: no such module: {module}"
            for module in operation.imports
            if not _module_exists(root, module)
        )
        refusals.extend(
            f"{operation.identifier}: no decorator declares the route {route}"
            for route in operation.routes
            if route not in inventory.routes
        )
    refusals.extend(
        f"An experimental namespace does not exist: {name}"
        for name in contract.experimental_namespaces
        if not _module_exists(root, name)
    )
    return tuple(refusals)


def defaults_refusals(
    contract: SupportContract, inventory: Inventory
) -> tuple[str, ...]:
    """Refuse production defaults that the contract does not admit.

    Parameters
    ----------
    contract : SupportContract
        The contract, whose production defaults name each setting's owner.
    inventory : Inventory
        The settings the production profile and quickstart turn on.

    Returns
    -------
    tuple[str, ...]
        The refusals.
    """
    enabled = set(inventory.profile_enabled) | set(inventory.quickstart_enabled)
    admitted = {default.setting for default in contract.production_defaults}
    refusals = [
        f"A production default turns on a setting no row owns: {setting}"
        for setting in sorted(enabled - admitted)
    ]
    refusals.extend(
        f"An admitted production default is no longer turned on: {setting}"
        for setting in sorted(admitted - enabled)
    )
    streaming = sorted(
        name
        for name, value in inventory.profile_values.items()
        if "streaming" in name and value not in (False, None, 0, "")
    )
    streaming += sorted(
        name for name in inventory.quickstart_enabled if "streaming" in name
    )
    refusals.extend(
        f"A production default turns on streaming: {name}" for name in streaming
    )
    refusals.extend(
        f"The production profile weakens a model-backed NLI requirement: {name}"
        for name in NLI_REQUIREMENTS
        if inventory.profile_values.get(name) is not True
    )
    refusals.extend(
        f"The production quickstart weakens a model-backed NLI requirement: {name}"
        for name in QUICKSTART_NLI_REQUIREMENTS
        if name not in inventory.quickstart_enabled
    )
    return tuple(refusals)


def _block_span(text: str) -> tuple[int, int] | None:
    """Return where a page's one generated block starts and ends, if it has one."""
    if text.count(BLOCK_START) != 1 or text.count(BLOCK_END) != 1:
        return None
    start, end = text.index(BLOCK_START), text.index(BLOCK_END) + len(BLOCK_END)
    return (start, end) if start < end else None


def docs_refusals(contract: SupportContract, root: Path) -> tuple[str, ...]:
    """Refuse a public page whose support statements are not the generated block.

    Parameters
    ----------
    contract : SupportContract
        The contract.
    root : Path
        Repository root.

    Returns
    -------
    tuple[str, ...]
        A refusal for a page without exactly one block, with a stale block, or
        with a support claim outside its block.
    """
    refusals = []
    for relative, link in DOC_BLOCKS:
        path = root / relative
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        span = _block_span(text)
        if span is None:
            refusals.append(f"{relative}: needs exactly one generated support block")
            continue
        start, end = span
        if text[start:end] != render_block(contract, link):
            refusals.append(f"{relative}: the generated support block is stale")
        if SUPPORT_CLAIM.search(text[:start] + text[end:]):
            refusals.append(f"{relative}: a support claim outside the generated block")
        if not (path.parent / link).is_file():
            refusals.append(
                f"{relative}: the link to the full table does not resolve: {link}"
            )
    return tuple(refusals)


def write_blocks(contract: SupportContract, root: Path) -> None:
    """Rewrite each public page's generated block from the contract.

    Parameters
    ----------
    contract : SupportContract
        The contract.
    root : Path
        Repository root.

    Raises
    ------
    ValueError
        If a page does not hold exactly one pair of block markers.
    """
    for relative, link in DOC_BLOCKS:
        path = root / relative
        text = path.read_text(encoding="utf-8")
        span = _block_span(text)
        if span is None:
            raise ValueError(f"{relative} has no single pair of support block markers")
        start, end = span
        path.write_text(
            text[:start] + render_block(contract, link) + text[end:], encoding="utf-8"
        )


def tree_refusals(contract: SupportContract, root: Path) -> tuple[str, ...]:
    """Return every refusal of the contract against a source tree.

    Parameters
    ----------
    contract : SupportContract
        The contract.
    root : Path
        Repository root.

    Returns
    -------
    tuple[str, ...]
        The refusals, in a stable order.
    """
    inventory = read_inventory(root)
    refusals = list(
        contract_refusals(
            contract,
            exports=inventory.exports,
            commands=inventory.commands,
            subcommands=inventory.subcommands,
            extras=inventory.extras,
        )
    )
    refusals.extend(surface_refusals(contract, inventory, root))
    refusals.extend(defaults_refusals(contract, inventory))
    table = render_table(contract)
    for relative in GENERATED_PATHS:
        generated = root / relative
        current = generated.read_text(encoding="utf-8") if generated.is_file() else None
        if current != table:
            refusals.append(f"The generated table is stale: {relative}")
    refusals.extend(docs_refusals(contract, root))
    return tuple(refusals)


def main(
    argv: Sequence[str] | None = None, contract: SupportContract = CONTRACT
) -> int:
    """Run the check, the regeneration or the receipt check.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command-line arguments; None reads ``sys.argv``.
    contract : SupportContract
        The contract to check; the shipped catalog by default.

    Returns
    -------
    int
        0 when everything qualifies, 1 when a refusal is printed.
    """
    parser = argparse.ArgumentParser(
        description="Check the production support contract against the source tree."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--receipts", type=Path, nargs="+")
    parser.add_argument(
        "--tree", help="The Git tree the receipts must have tested; needs --receipts."
    )
    args = parser.parse_args(argv)
    if (args.tree is None) != (args.receipts is None):
        parser.error("--tree and --receipts are given together")
    root = args.root.resolve()
    if args.write:
        for relative in GENERATED_PATHS:
            (root / relative).write_text(render_table(contract), encoding="utf-8")
        try:
            write_blocks(contract, root)
        except ValueError as error:
            print(f"REFUSED: {error}")
            return 1
        print("Wrote the generated tables and the support blocks")
        return 0
    if args.receipts:
        refusals = check_receipts(contract, args.receipts, tree=args.tree)
    else:
        refusals = tree_refusals(contract, root)
    for refusal in refusals:
        print(f"REFUSED: {refusal}")
    if not refusals:
        print("The production support contract qualifies")
    return 1 if refusals else 0


if __name__ == "__main__":
    raise SystemExit(main())
