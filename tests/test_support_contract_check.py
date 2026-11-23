# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — tests of the support contract's CI gate and its rendered table
"""Pin the gate's refusals: the regression controls the plan names."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from director_ai.support_contract import Operation, ProfileDefault
from director_ai.support_contract_catalog import CONTRACT
from tools.support_contract_check import (
    defaults_refusals,
    docs_refusals,
    main,
    surface_refusals,
    tree_refusals,
    write_blocks,
)
from tools.support_contract_inventory import read_inventory
from tools.support_contract_render import (
    BLOCK_END,
    BLOCK_START,
    DOC_BLOCKS,
    GENERATED_PATH,
    GENERATED_PATHS,
    render_block,
    render_table,
)

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = read_inventory(ROOT)


def test_an_unclassified_export_or_command_is_refused() -> None:
    """Refuse a new export or command the contract does not classify."""
    exports = dataclasses.replace(
        CONTRACT, experimental_exports=CONTRACT.experimental_exports[1:]
    )
    first = CONTRACT.experimental_exports[0]
    assert f"An export is not classified: {first}" in tree_refusals(exports, ROOT)
    commands = dataclasses.replace(
        CONTRACT, experimental_commands=CONTRACT.experimental_commands[1:]
    )
    command = CONTRACT.experimental_commands[0]
    assert f"A command is not classified: {command}" in tree_refusals(commands, ROOT)


def test_an_unavailable_module_or_undeclared_route_is_refused() -> None:
    """Refuse a row that names a missing module, route or namespace."""
    row = Operation(
        identifier="ghost",
        title="Ghost",
        support="experimental",
        activation="default",
        imports=("director_ai.no_such_module",),
        routes=("/v1/no-such-route",),
    )
    changed = dataclasses.replace(
        CONTRACT,
        operations=CONTRACT.operations + (row,),
        experimental_namespaces=("director_ai.model_lifecycle", "director_ai.nowhere"),
    )
    assert surface_refusals(changed, INVENTORY, ROOT) == (
        "ghost: no such module: director_ai.no_such_module",
        "ghost: no decorator declares the route /v1/no-such-route",
        "An experimental namespace does not exist: director_ai.nowhere",
    )


def test_a_labs_flag_in_a_production_default_is_refused() -> None:
    """Refuse a setting the production profile turns on without an owner."""
    enabled = dataclasses.replace(
        INVENTORY,
        profile_enabled=INVENTORY.profile_enabled + ("swarm_economics_enabled",),
    )
    assert defaults_refusals(CONTRACT, enabled) == (
        "A production default turns on a setting no row owns: swarm_economics_enabled",
    )


def test_a_stale_admitted_default_is_refused() -> None:
    """Refuse an admitted default the profile no longer turns on."""
    extra = ProfileDefault("zz_retired_setting", "model-scoring")
    changed = dataclasses.replace(
        CONTRACT, production_defaults=CONTRACT.production_defaults + (extra,)
    )
    assert defaults_refusals(changed, INVENTORY) == (
        "An admitted production default is no longer turned on: zz_retired_setting",
    )


def test_streaming_turned_on_by_default_is_refused() -> None:
    """Refuse streaming enabled in the profile or the generated quickstart."""
    values = dict(INVENTORY.profile_values, streaming_contradiction_halt=True)
    profile = dataclasses.replace(INVENTORY, profile_values=values)
    assert defaults_refusals(CONTRACT, profile) == (
        "A production default turns on streaming: streaming_contradiction_halt",
    )
    quickstart = dataclasses.replace(
        INVENTORY, quickstart_enabled=INVENTORY.quickstart_enabled + ("streaming_halt",)
    )
    assert defaults_refusals(CONTRACT, quickstart) == (
        "A production default turns on a setting no row owns: streaming_halt",
        "A production default turns on streaming: streaming_halt",
    )


def test_a_weaker_model_backed_nli_requirement_is_refused() -> None:
    """Refuse a profile or quickstart that drops the model-backed NLI requirement."""
    values = dict(INVENTORY.profile_values, coherence_require_model_backed_nli=False)
    enabled = tuple(
        name
        for name in INVENTORY.profile_enabled
        if name != "coherence_require_model_backed_nli"
    )
    profile = dataclasses.replace(
        INVENTORY, profile_values=values, profile_enabled=enabled
    )
    assert (
        "The production profile weakens a model-backed NLI requirement: "
        "coherence_require_model_backed_nli"
    ) in defaults_refusals(CONTRACT, profile)
    quick = tuple(
        name
        for name in INVENTORY.quickstart_enabled
        if name != "injection_require_model_backed_nli"
    )
    quickstart = dataclasses.replace(INVENTORY, quickstart_enabled=quick)
    assert defaults_refusals(CONTRACT, quickstart) == (
        "The production quickstart weakens a model-backed NLI requirement: "
        "injection_require_model_backed_nli",
    )


def test_a_stale_generated_table_is_refused() -> None:
    """Refuse a committed table that no longer matches the contract."""
    renamed = dataclasses.replace(
        CONTRACT.operations[0], title=CONTRACT.operations[0].title + " (renamed)"
    )
    changed = dataclasses.replace(
        CONTRACT, operations=(renamed,) + CONTRACT.operations[1:]
    )
    assert tree_refusals(changed, ROOT) == (
        *(f"The generated table is stale: {path}" for path in GENERATED_PATHS),
        *(
            f"{page}: the generated support block is stale"
            for page, _link in DOC_BLOCKS
        ),
    )


def test_the_table_states_each_field_and_the_scope() -> None:
    """Render support, installation with extras, activation, coverage and notes."""
    table = render_table(CONTRACT)
    assert table.startswith("<!-- Generated by tools/support_contract_check.py")
    assert CONTRACT.meaning in table
    assert "| core+pro + `server` | command | `serve`, `/v1/health`" in table
    assert "`compliance governance`, `compliance report`" in table
    assert "| not claimed | opt-in |" in table
    assert "| `redact_pii` | `pii-redactor` | production-qualified |" in table
    assert "| `use_nli` | `model-scoring` | experimental |" in table
    assert "`director_ai.model_lifecycle`" in table
    assert (ROOT / GENERATED_PATH).read_text(encoding="utf-8") == table


def test_the_command_line_checks_writes_and_reads_receipts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pass on the tree, write the table, and refuse an empty receipt set."""
    assert main(["--check", "--root", str(ROOT)]) == 0
    assert "qualifies" in capsys.readouterr().out
    for page, _link in DOC_BLOCKS:
        _page(tmp_path, page, f"{BLOCK_START}\n{BLOCK_END}\n")
    for path in GENERATED_PATHS:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
    assert main(["--write", "--root", str(tmp_path)]) == 0
    for path in GENERATED_PATHS:
        assert (tmp_path / path).read_text(encoding="utf-8") == render_table(CONTRACT)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "core.json").write_text(json.dumps({"schema": "x"}), encoding="utf-8")
    assert main(["--receipts", str(receipts), "--tree", "8" * 40]) == 1
    assert "REFUSED: core.json: unknown receipt schema" in capsys.readouterr().out
    for unpaired in (["--receipts", str(receipts)], ["--check", "--tree", "8" * 40]):
        with pytest.raises(SystemExit):
            main(unpaired)
    assert "given together" in capsys.readouterr().err
    stale = dataclasses.replace(
        CONTRACT, experimental_namespaces=("director_ai.nowhere",)
    )
    assert main(["--check", "--root", str(ROOT)], contract=stale) == 1


def test_the_module_runs_as_a_program() -> None:
    """Exit 0 from ``python -m tools.support_contract_check --check``."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"}
    result = subprocess.run(
        [sys.executable, "-m", "tools.support_contract_check", "--check"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "qualifies" in result.stdout


def test_the_public_block_lists_the_qualified_rows_and_the_experimental_defaults() -> (
    None
):
    """Render a qualified row's table, or say that none is qualified yet."""
    unqualified = dataclasses.replace(
        CONTRACT,
        operations=tuple(
            dataclasses.replace(op, support="experimental", custody=())
            for op in CONTRACT.operations
        ),
    )
    assert "No operation is production-qualified yet." in render_block(
        unqualified, "x.md"
    )
    row = dataclasses.replace(
        unqualified.operation("input-sanitizer"), support="production-qualified"
    )
    rows = tuple(
        row if op.identifier == row.identifier else op for op in unqualified.operations
    )
    block = render_block(dataclasses.replace(unqualified, operations=rows), "x.md")
    assert block.startswith(BLOCK_START) and block.endswith(BLOCK_END)
    assert (
        "| Rule-based prompt-injection screen | core | default | Blocks the declared "
        "override patterns and allows the probed benign input; its detection rate on "
        "injection attacks is not measured or qualified; consults no model. |" in block
    )
    assert "Rule-based prompt-injection screen;" not in block
    assert "[production support contract](x.md)" in block


def _page(root: Path, relative: str, text: str) -> None:
    """Write one public page under a test tree."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _targets(root: Path) -> None:
    """Create the full tables that the pages' links point to."""
    for page, link in DOC_BLOCKS:
        _page(root, str(Path(page).parent / link), "table\n")


def test_each_public_page_must_carry_one_current_block(tmp_path: Path) -> None:
    """Refuse a missing, doubled or stale block, a claim outside it, a dead link."""
    (readme, readme_link), (checklist, checklist_link), (guide, guide_link) = DOC_BLOCKS
    _targets(tmp_path)
    _page(tmp_path, readme, "# Readme\n")
    pair = f"{BLOCK_START}\n{BLOCK_END}"
    _page(tmp_path, checklist, f"{pair}\n{pair}\n")
    current = render_block(CONTRACT, guide_link)
    _page(tmp_path, guide, f"Our supported for production surface.\n{current}\n")
    assert docs_refusals(CONTRACT, tmp_path) == (
        f"{readme}: needs exactly one generated support block",
        f"{checklist}: needs exactly one generated support block",
        f"{guide}: a support claim outside the generated block",
    )
    _page(tmp_path, readme, f"{BLOCK_END}\n{BLOCK_START}\n")
    _page(tmp_path, checklist, f"{pair}\n")
    _page(tmp_path, guide, f"{current}\n")
    assert docs_refusals(CONTRACT, tmp_path) == (
        f"{readme}: needs exactly one generated support block",
        f"{checklist}: the generated support block is stale",
    )
    assert render_block(CONTRACT, readme_link) != render_block(CONTRACT, checklist_link)
    (tmp_path / Path(guide).parent / guide_link).unlink()
    assert docs_refusals(CONTRACT, tmp_path)[-1] == (
        f"{guide}: the link to the full table does not resolve: {guide_link}"
    )


def test_every_real_page_link_resolves() -> None:
    """Resolve each page's link from that page in the repository."""
    for page, link in DOC_BLOCKS:
        assert (ROOT / page).parent.joinpath(link).resolve().is_file(), page


def test_the_blocks_are_rewritten_in_place(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refresh every block, and refuse a page without its markers."""
    pair = f"{BLOCK_START}\nold\n{BLOCK_END}"
    _targets(tmp_path)
    for relative, _link in DOC_BLOCKS:
        _page(tmp_path, relative, f"before\n{pair}\nafter\n")
    write_blocks(CONTRACT, tmp_path)
    assert docs_refusals(CONTRACT, tmp_path) == ()
    assert (
        (tmp_path / DOC_BLOCKS[0][0]).read_text(encoding="utf-8").startswith("before\n")
    )
    _page(tmp_path, DOC_BLOCKS[1][0], "no markers\n")
    assert main(["--write", "--root", str(tmp_path)]) == 1
    assert "has no single pair of support block markers" in capsys.readouterr().out
