# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native Lean declarations and original source documentation
"""Join authored syntax to elaborated declarations without counting generated APIs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_contracts import decode_quality_json, repository_path

NATIVE_SCHEMA = "quality-ratchet.lean-native.v1"
LEAN_VERSION = "4.34.1"
LEAN_COMMIT = "5045d0056413266e57c625dcd7c365b10e377c52"
COMMAND = "Lean.Parser.Command."


def _integer(value: object, minimum: int = 0) -> int:
    """Require an actual native integer, retaining zero-based columns."""
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid native Lean integer")
    return value


def _text(value: object, *, nullable: bool = False) -> str | None:
    """Validate lexical text, preserving absent identifiers and empty atoms."""
    if nullable and value is None:
        return None
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("Invalid native Lean text")
    return value


def _array(value: object) -> list[object]:
    """Require a real JSON array without filtering invalid members."""
    if not isinstance(value, list):
        raise ValueError("Expected native Lean array")
    return list(value)


def _position(value: object, source: str) -> tuple[int, int, int] | None:
    """Bind original character coordinates to a real UTF-8 source boundary."""
    if value is None:
        return None
    row = object_mapping(value)
    if set(row) != {"line", "column", "byte_offset"}:
        raise ValueError("Invalid native Lean position fields")
    line = _integer(row["line"], 1)
    column = _integer(row["column"])
    offset = _integer(row["byte_offset"])
    original = source.encode("utf-8")
    if offset > len(original):
        raise ValueError("Native Lean position exceeds original source")
    prefix = original[:offset].decode("utf-8")
    if (line, column) != (prefix.count("\n") + 1, len(prefix.rsplit("\n", 1)[-1])):
        raise ValueError("Native Lean coordinates differ from original source")
    return line, column, offset


@dataclass(frozen=True)
class _Syntax:
    """Retain one native syntax node and its original nullable source bounds.

    Attributes
    ----------
    kind : str
        Compiler parser kind, including lexical and empty nodes.
    identifier, atom : str | None
        Original compiler identifier or lexical atom.
    start, end : tuple[int, int, int] | None
        Native line, character column and UTF-8 byte offset.
    children : tuple[_Syntax, ...]
        Complete ordered native child nodes.
    """

    kind: str
    identifier: str | None
    atom: str | None
    start: tuple[int, int, int] | None
    end: tuple[int, int, int] | None
    children: tuple[_Syntax, ...]

    @classmethod
    def read(cls, value: object, source: str) -> _Syntax:
        """Validate every original node before interpreting source ownership."""
        row = object_mapping(value)
        if set(row) != {"kind", "identifier", "atom", "start", "end", "children"}:
            raise ValueError("Invalid native Lean syntax fields")
        kind = row["kind"]
        if not isinstance(kind, str) or not kind:
            raise ValueError("Native Lean syntax has no kind")
        start = _position(row["start"], source)
        end = _position(row["end"], source)
        if start is not None and end is not None and start[2] > end[2]:
            raise ValueError("Native Lean syntax has reversed source bounds")
        return cls(
            kind,
            _text(row["identifier"], nullable=True),
            _text(row["atom"], nullable=True),
            start,
            end,
            tuple(cls.read(child, source) for child in _array(row["children"])),
        )

    def nodes(self) -> Iterator[_Syntax]:
        """Visit complete native syntax without searching original source prose."""
        yield self
        for child in self.children:
            yield from child.nodes()


def _comment(node: _Syntax, kind: str) -> bool:
    """Require nonempty text in an actual native documentation comment."""
    return any(
        item.kind == COMMAND + kind
        and len(item.children) == 2
        and item.children[1].atom is not None
        and bool(item.children[1].atom.removesuffix("-/").strip())
        for item in node.nodes()
    )


def _constants(
    value: object, source: str
) -> dict[tuple[int, int], list[tuple[str, bool]]]:
    """Index actual elaborated constants by original identifier selection.

    Parameters
    ----------
    value : object
        Complete original constant array from the native environment.
    source : str
        Exact original source inspected by the compiler.

    Returns
    -------
    dict[tuple[int, int], list[tuple[str, bool]]]
        Native names and documentation presence at original selections.

    Raises
    ------
    ValueError
        If constant identity, documentation or source range is malformed.
    """
    result: dict[tuple[int, int], list[tuple[str, bool]]] = {}
    names: set[str] = set()
    lines = source.split("\n")
    keys = {
        "line",
        "column",
        "end_line",
        "end_column",
        "selection_line",
        "selection_column",
        "selection_end_line",
        "selection_end_column",
    }
    for value_row in _array(value):
        row = object_mapping(value_row)
        name = row.get("name")
        if (
            set(row) != {"name", "doc", "range"}
            or not isinstance(name, str)
            or not name
            or name in names
        ):
            raise ValueError("Invalid or duplicated native Lean constant")
        names.add(name)
        doc = _text(row["doc"], nullable=True)
        if row["range"] is None:
            continue
        bounds = object_mapping(row["range"])
        if set(bounds) != keys:
            raise ValueError("Invalid native Lean declaration range fields")
        positions = []
        for prefix in ("", "end_", "selection_", "selection_end_"):
            line = _integer(bounds[prefix + "line"], 1)
            column = _integer(bounds[prefix + "column"])
            if line > len(lines) or column > len(lines[line - 1]):
                raise ValueError("Native Lean declaration range exceeds source")
            positions.append((line, column))
        beginning, end, selection, selection_end = positions
        if not beginning <= selection <= selection_end <= end:
            raise ValueError("Native Lean declaration range is inconsistent")
        result.setdefault(selection, []).append((name, bool(doc and doc.strip())))
    return result


def _selection(node: _Syntax) -> _Syntax:
    """Find a direct declaration identifier or actual anonymous-instance keyword."""
    candidates = [
        item
        for child in node.children
        for item in child.nodes()
        if child.kind in (COMMAND + "declId", "null")
        and item.kind == COMMAND + "declId"
    ]
    if candidates:
        if len(candidates) != 1 or not candidates[0].children:
            raise ValueError("Ambiguous native Lean declaration identifier")
        identifier = candidates[0].children[0]
        if not identifier.identifier:
            raise ValueError("Native Lean declaration identifier is absent")
        return identifier
    if node.kind in {COMMAND + "instance", COMMAND + "example"}:
        keyword = node.kind.removeprefix(COMMAND)
        tokens = [child for child in node.children if child.atom == keyword]
        if len(tokens) == 1:
            return tokens[0]
    raise ValueError("Native Lean declaration has no source selection")


def _owners(command: _Syntax) -> Iterator[tuple[_Syntax, _Syntax, bool]]:
    """Visit authored command declarations, constructors and structure fields.

    Parameters
    ----------
    command : _Syntax
        Complete pre-expansion native command, preserving real source syntax.

    Yields
    ------
    tuple[_Syntax, _Syntax, bool]
        Actual declaration, original name selection and attached source doc.

    Raises
    ------
    ValueError
        If a command or declaration form has no qualified source ownership.
    """
    if command.kind == COMMAND + "in":
        if len(command.children) != 3:
            raise ValueError("Invalid native Lean scoped command")
        yield from _owners(command.children[0])
        yield from _owners(command.children[2])
    elif command.kind == COMMAND + "mutual":
        if len(command.children) != 3 or command.children[1].kind != "null":
            raise ValueError("Invalid native Lean mutual command")
        for child in command.children[1].children:
            yield from _owners(child)
    elif command.kind == COMMAND + "declaration":
        if (
            len(command.children) != 2
            or command.children[0].kind != COMMAND + "declModifiers"
        ):
            raise ValueError("Invalid native Lean declaration command")
        modifiers, declaration = command.children
        forms = {
            "definition",
            "theorem",
            "abbrev",
            "opaque",
            "axiom",
            "structure",
            "inductive",
            "instance",
            "example",
        }
        if declaration.kind not in {COMMAND + form for form in forms}:
            raise ValueError("Unsupported native Lean declaration form")
        yield declaration, _selection(declaration), _comment(modifiers, "docComment")
        for node in declaration.nodes():
            if node.kind == "Lean.Parser.Term.whereDecls" and node.children:
                raise ValueError(
                    "Local Lean where declarations need native ownership qualification"
                )
            if node.kind in {
                COMMAND + kind
                for kind in (
                    "structExplicitBinder",
                    "structImplicitBinder",
                    "structInstBinder",
                    "structCtor",
                    "computedFields",
                )
            }:
                raise ValueError(
                    "Lean member form needs native ownership qualification"
                )
        if declaration.kind not in {COMMAND + "structure", COMMAND + "inductive"}:
            return
        for node in declaration.nodes():
            if node.kind not in {COMMAND + "ctor", COMMAND + "structSimpleBinder"}:
                continue
            names = [child for child in node.children if child.identifier is not None]
            if len(names) != 1:
                raise ValueError("Native Lean member has no unique source identifier")
            yield node, names[0], _comment(node, "docComment")
    elif command.kind not in {
        COMMAND + kind
        for kind in (
            "moduleDoc",
            "namespace",
            "section",
            "end",
            "open",
            "variable",
            "set_option",
            "eoi",
        )
    }:
        raise ValueError("Unsupported native Lean command source ownership")


def lean_documentation_findings(
    original: bytes, source: bytes, path: str, module: str
) -> list[dict[str, object]]:
    """Evaluate full native Lean source documentation and actual masking commands.

    Parameters
    ----------
    original : bytes
        Original stdout from the pinned native inspection recipe.
    source : bytes
        Complete unchanged UTF-8 source inspected by that recipe.
    path : str
        Canonical repository-relative maintained source path.
    module : str
        Actual native source module identity supplied to elaboration.

    Returns
    -------
    list[dict[str, object]]
        Findings at real native coordinates. Implicit constructors, deriving
        instances and auxiliary elaboration constants inherit their authored
        source owner's documentation requirement; they are not extra authored
        declarations. Private declarations remain mandatory.

    Raises
    ------
    ValueError
        If bytes, compiler identity, source ownership or complete elaboration
        cannot qualify. Unsupported source forms never produce a clean result.
    """
    path = repository_path(path)
    text = source.decode("utf-8")
    row = object_mapping(decode_quality_json(original))
    keys = {
        "schema",
        "compiler",
        "compiler_commit",
        "mode",
        "module",
        "source",
        "has_errors",
        "header",
        "imports",
        "messages",
        "commands",
        "constants",
    }
    if (
        set(row) != keys
        or (
            row["schema"],
            row["compiler"],
            row["compiler_commit"],
            row["mode"],
            row["module"],
            row["source"],
            row["has_errors"],
        )
        != (NATIVE_SCHEMA, LEAN_VERSION, LEAN_COMMIT, "inspect", module, text, False)
        or type(row["has_errors"]) is not bool
        or not module
    ):
        raise ValueError("Native Lean inspection identity or complete source differs")
    for name in _array(row["imports"]):
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid native Lean import identity")
    for value in _array(row["messages"]):
        message = object_mapping(value)
        if set(message) != {"severity", "line", "column", "message"} or message[
            "severity"
        ] not in ("warning", "information"):
            raise ValueError("Native Lean inspection retains an error diagnostic")
        _integer(message["line"], 1)
        _integer(message["column"])
        _text(message["message"])
    header = _Syntax.read(row["header"], text)
    if header.kind != "Lean.Parser.Module.header":
        raise ValueError("Native Lean source header is absent")
    commands = tuple(_Syntax.read(command, text) for command in _array(row["commands"]))
    if not commands or commands[-1].kind != COMMAND + "eoi":
        raise ValueError("Native Lean command inventory lacks end of source")
    constants = _constants(row["constants"], text)
    findings: list[dict[str, object]] = []
    covered: set[tuple[int, int]] = set()

    def report(node: _Syntax, rule: str, message: str) -> None:
        """Retain a real source position, refusing synthetic diagnostic placement."""
        if node.start is None:
            raise ValueError("Native Lean finding lacks an original source position")
        findings.append(
            {
                "path": path,
                "rule": rule,
                "line": node.start[0],
                "column": node.start[1] + 1,
                "message": message,
                "cell": None,
            }
        )

    if not any(_comment(command, "moduleDoc") for command in commands):
        location = next(
            (node for node in header.nodes() if node.start is not None), commands[0]
        )
        report(
            location, "module-documentation", "Missing native Lean module documentation"
        )
    for command in commands:
        for node in command.nodes():
            if (
                node.kind == COMMAND + "set_option"
                and any(
                    child.identifier is not None
                    and child.identifier.startswith("linter.")
                    for child in node.children
                )
                and any(child.atom == "false" for child in node.children)
            ):
                report(
                    node,
                    "source-directive",
                    "Native Lean command disables source linting",
                )
        for declaration, selection, attached in _owners(command):
            if selection.start is None:
                raise ValueError(
                    "Native Lean declaration lacks an original source selection"
                )
            covered.add(selection.start[:2])
            for node in declaration.nodes():
                if node.kind == COMMAND + "derivingClass" and node.start is not None:
                    covered.add(node.start[:2])
            if declaration.kind == COMMAND + "example":
                documented = attached
                name = "example"
            else:
                candidates = constants.get(selection.start[:2], [])
                name = selection.identifier
                matching = [
                    (identity, doc)
                    for identity, doc in candidates
                    if name is None or identity == name or identity.endswith("." + name)
                ]
                if len(matching) != 1:
                    raise ValueError(
                        "Authored Lean declaration lacks a unique native constant"
                    )
                name, documented = matching[0]
            if not documented:
                report(
                    selection,
                    "declaration-documentation",
                    "Missing native Lean documentation for " + name,
                )
    if set(constants) - covered:
        raise ValueError("Native Lean ranged constants lack authored source ownership")
    return findings
