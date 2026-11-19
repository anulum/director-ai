# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native Lean authored-source selection integrity controls
"""Refuse original syntax transport that loses declaration identity or source placement."""

from __future__ import annotations

import json

import pytest

from formal.HaltMonitor.tests.test_quality_ratchet_lean_declarations import (
    documented_native as documented_native,
)
from formal.HaltMonitor.tests.test_quality_ratchet_lean_declarations import (
    native_reader as native_reader,
)
from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_contracts import decode_quality_json
from tools.quality_ratchet_lean_declarations import lean_documentation_findings


def _change_kind(
    node: dict[str, object], kind: str, changes: dict[str, object]
) -> bool:
    """Change one real syntax node while retaining the complete original elsewhere."""
    if node["kind"] == kind:
        node.update(changes)
        return True
    children = node["children"]
    assert isinstance(children, list)
    typed = [object_mapping(child) for child in children]
    node["children"] = typed
    return any(_change_kind(child, kind, changes) for child in typed)


@pytest.mark.parametrize(
    "attack",
    [
        "identifier-duplicate",
        "identifier-empty",
        "identifier-missing",
        "instance-keyword",
        "example-keyword",
        "scoped-command",
        "mutual-command",
        "declaration-command",
        "declaration-form",
        "member-identity",
        "finding-position",
        "selection-position",
    ],
)
def test_native_selection_and_placement_cannot_be_invented(
    documented_native: tuple[bytes, bytes],
    attack: str,
) -> None:
    """Reject deliberately corrupted original AST ownership and real positions.

    Parameters
    ----------
    documented_native : tuple[bytes, bytes]
        Actual original compiler output and unchanged source bytes.
    attack : str
        Specific transport corruption applied after original native execution.
    """
    original, source = documented_native
    row = object_mapping(decode_quality_json(original))
    values = row["commands"]
    assert isinstance(values, list)
    commands = [object_mapping(value) for value in values]
    row["commands"] = commands
    declaration = next(
        command
        for command in commands
        if command["kind"] == "Lean.Parser.Command.declaration"
    )
    if attack == "identifier-duplicate":
        parts = declaration["children"]
        assert isinstance(parts, list)
        inner = object_mapping(parts[1])
        parts[1] = inner
        children = inner["children"]
        assert isinstance(children, list)
        children.append(children[1])
    elif attack in ("identifier-empty", "selection-position"):
        changes: dict[str, object] = (
            {"identifier": ""} if attack == "identifier-empty" else {"start": None}
        )
        assert _change_kind(declaration, "ident", changes)
    elif attack == "identifier-missing":
        assert _change_kind(declaration, "Lean.Parser.Command.declId", {"kind": "null"})
    elif attack in ("instance-keyword", "example-keyword"):
        kind = attack.removesuffix("-keyword")
        assert any(_change_kind(command, kind, {"atom": None}) for command in commands)
    elif attack == "scoped-command":
        declaration["kind"] = "Lean.Parser.Command.in"
    elif attack == "mutual-command":
        assert any(
            _change_kind(command, "Lean.Parser.Command.mutual", {"children": []})
            for command in commands
        )
    elif attack == "declaration-command":
        declaration["children"] = []
    elif attack == "declaration-form":
        assert _change_kind(
            declaration,
            "Lean.Parser.Command.definition",
            {"kind": "Lean.Parser.Command.classInductive"},
        )
    elif attack == "member-identity":
        assert any(
            _change_kind(
                command, "Lean.Parser.Command.structSimpleBinder", {"children": []}
            )
            for command in commands
        )
    else:
        commands.pop(0)
        commands[0]["start"] = None
        header = object_mapping(row["header"])
        row["header"] = header
        header.update({"start": None, "end": None, "children": []})
    with pytest.raises(ValueError):
        lean_documentation_findings(
            json.dumps(row).encode(), source, "formal/Admission.lean", "Admission"
        )
