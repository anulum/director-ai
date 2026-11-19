# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — original Protobuf descriptor declarations and source documentation
"""Retain native declaration addresses, comment attachment and source positions."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence

from google.protobuf import descriptor_pb2
from google.protobuf.descriptor_pb2 import (
    DescriptorProto,
    EnumDescriptorProto,
    FieldDescriptorProto,
    FileDescriptorProto,
)
from google.protobuf.message import DecodeError

DOCUMENTATION_MASK = re.compile(
    r"^\s*(?:@exclude\b|buf:lint:ignore(?:\s|$))", re.MULTILINE
)


def declarations(
    file: FileDescriptorProto,
) -> Iterator[tuple[tuple[int, ...], str, tuple[int, ...]]]:
    """Enumerate actual named schema APIs using native descriptor field paths.

    Parameters
    ----------
    file : FileDescriptorProto
        Original protoc file descriptor with source information.

    Yields
    ------
    tuple[tuple[int, ...], str, tuple[int, ...]]
        Exact declaration address, name and documentation attachment address.
        Native synthetic map entries and optional-field oneofs are excluded;
        their actual source fields remain mandatory. A proto2 group field uses
        documentation attached by protoc to its corresponding group type.
    """

    def enum_items(
        enum: EnumDescriptorProto,
        address: tuple[int, ...],
    ) -> Iterator[tuple[tuple[int, ...], str, tuple[int, ...]]]:
        """Enumerate a real enum and its values without flattening source identity."""
        yield address, "enum " + enum.name, address
        for index, value in enumerate(enum.value):
            yield (*address, 2, index), "enum value " + value.name, (*address, 2, index)

    def field_items(
        fields: Sequence[FieldDescriptorProto],
        nested: Sequence[DescriptorProto],
        address: tuple[int, ...],
        field_tag: int,
        nested_tag: int,
        kind: str,
    ) -> Iterator[tuple[tuple[int, ...], str, tuple[int, ...]]]:
        """Retain each field and the compiler's actual group-comment association."""
        groups = {message.name: index for index, message in enumerate(nested)}
        for index, field in enumerate(fields):
            declaration = (*address, field_tag, index)
            attachment = declaration
            if field.type == FieldDescriptorProto.TYPE_GROUP:
                name = field.type_name.rsplit(".", 1)[-1]
                if name not in groups:
                    raise ValueError("Native Protobuf group field has no original type")
                attachment = (*address, nested_tag, groups[name])
            yield declaration, kind + " " + field.name, attachment

    def message_items(
        message: DescriptorProto,
        address: tuple[int, ...],
    ) -> Iterator[tuple[tuple[int, ...], str, tuple[int, ...]]]:
        """Enumerate real message declarations, nested APIs and extension fields."""
        if message.options.map_entry:
            return
        yield address, "message " + message.name, address
        yield from field_items(
            message.field, message.nested_type, address, 2, 3, "field"
        )
        synthetic = {
            field.oneof_index for field in message.field if field.proto3_optional
        }
        for index, oneof in enumerate(message.oneof_decl):
            if index not in synthetic:
                yield (*address, 8, index), "oneof " + oneof.name, (*address, 8, index)
        yield from field_items(
            message.extension, message.nested_type, address, 6, 3, "extension"
        )
        for index, nested in enumerate(message.nested_type):
            yield from message_items(nested, (*address, 3, index))
        for index, enum in enumerate(message.enum_type):
            yield from enum_items(enum, (*address, 4, index))

    yield (12,), "schema file", (12,)
    for index, message in enumerate(file.message_type):
        yield from message_items(message, (4, index))
    for index, enum in enumerate(file.enum_type):
        yield from enum_items(enum, (5, index))
    yield from field_items(file.extension, file.message_type, (), 7, 4, "extension")
    for index, service in enumerate(file.service):
        yield (6, index), "service " + service.name, (6, index)
        for method_index, method in enumerate(service.method):
            yield (
                (6, index, 2, method_index),
                "RPC " + method.name,
                (6, index, 2, method_index),
            )


def descriptor_findings(
    original: bytes,
    paths: tuple[str, ...],
) -> list[dict[str, object]]:
    """Decode original native descriptors and preserve real source positions.

    Parameters
    ----------
    original : bytes
        Original protoc descriptor-set stdout, never a projected report.
    paths : tuple[str, ...]
        Complete source inventory, including every allowed import.

    Returns
    -------
    list[dict[str, object]]
        Documentation findings on actual declarations and source masks.
        Positions identify the associated declaration, because protoc does
        not expose comment-token spans.

    Raises
    ------
    ValueError
        If native descriptors, inventories or mandatory locations are incomplete.
    """
    try:
        descriptor = descriptor_pb2.FileDescriptorSet.FromString(original)
    except DecodeError as error:
        raise ValueError(
            "Original Protobuf descriptor bytes cannot be decoded"
        ) from error
    names = [file.name for file in descriptor.file]
    if len(names) != len(set(names)) or set(names) != set(paths):
        raise ValueError("Native Protobuf descriptor inventory differs from source")
    findings: list[dict[str, object]] = []
    for file in descriptor.file:
        locations: dict[
            tuple[int, ...], list[descriptor_pb2.SourceCodeInfo.Location]
        ] = {}
        for location in file.source_code_info.location:
            address = tuple(location.path)
            if len(location.span) not in (3, 4) or any(n < 0 for n in location.span):
                raise ValueError("Native Protobuf location has no valid original span")
            comments = (
                location.leading_comments,
                location.trailing_comments,
                *location.leading_detached_comments,
            )
            if any(DOCUMENTATION_MASK.search(comment) for comment in comments):
                findings.append(
                    {
                        "path": file.name,
                        "rule": "source-directive",
                        "line": location.span[0] + 1,
                        "column": location.span[1] + 1,
                        "message": "Documentation mask attached to native schema location",
                        "cell": None,
                    }
                )
            locations.setdefault(address, []).append(location)
        for address, name, attachment in declarations(file):
            if address not in locations or attachment not in locations:
                raise ValueError("Native Protobuf declaration has no original location")
            if len(locations[address]) != 1 or len(locations[attachment]) != 1:
                raise ValueError("Native Protobuf declaration location is ambiguous")
            location = locations[address][0]
            documentation = locations[attachment][0]
            attached = (documentation.leading_comments, documentation.trailing_comments)
            if not any(
                comment.strip() and not DOCUMENTATION_MASK.search(comment)
                for comment in attached
            ):
                findings.append(
                    {
                        "path": file.name,
                        "rule": "declaration-documentation",
                        "line": location.span[0] + 1,
                        "column": location.span[1] + 1,
                        "message": "Missing attached documentation for " + name,
                        "cell": None,
                    }
                )
    return findings
