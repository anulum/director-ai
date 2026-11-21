# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Canonical JSON and Digests

"""Canonical JSON, record digests and strict record decoding.

The canonical rule is ``json.dumps(value, sort_keys=True,
separators=(",", ":"), ensure_ascii=False, allow_nan=False)`` encoded as
UTF-8, with dataclasses turned into objects, tuples into arrays and every
float finite and never a negative zero, so ``-0.0`` and ``0.0`` cannot give
two digests of one record. :func:`record_digest` is the SHA-256 of that
encoding.

:func:`decode_record` turns JSON bytes back into a record: strict UTF-8, no
duplicate member names, no ``NaN`` or ``Infinity``, no negative-zero number
token (``-0``, ``-0.0``, ``-0e0``), exactly the record's field names, and no
``bool``/``int`` or ``int``/``float`` coercion. It then runs the record's
validator, which the caller must supply.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import types
from collections.abc import Callable
from typing import Literal, TypeVar, Union, cast, get_args, get_origin, get_type_hints

from .errors import NonCanonicalFloatError, NonCanonicalJsonError

RecordT = TypeVar("RecordT")


def _to_json_value(value: object) -> object:
    """Return ``value`` as plain JSON data under the canonical rule."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, tuple | list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise NonCanonicalJsonError("canonical JSON objects need string keys")
        return {key: _to_json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NonCanonicalFloatError("canonical JSON floats must be finite")
        if value == 0.0 and math.copysign(1.0, value) < 0.0:
            raise NonCanonicalFloatError(
                "canonical JSON floats must not be negative zero"
            )
        return value
    raise NonCanonicalJsonError(f"{type(value).__name__} has no canonical JSON form")


def canonical_json_bytes(value: object) -> bytes:
    """Encode a record or JSON value under the canonical rule.

    Parameters
    ----------
    value : object
        A frozen record dataclass, or plain JSON data built from objects,
        arrays, strings, integers, finite floats, booleans and ``None``.

    Returns
    -------
    bytes
        The UTF-8 canonical encoding.

    Raises
    ------
    NonCanonicalJsonError
        If the value holds a non-finite or negative-zero float
        (:class:`~.errors.NonCanonicalFloatError`), a string that does not
        encode as UTF-8 (a lone surrogate), or a type without a JSON form.
    """
    text = json.dumps(
        _to_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise NonCanonicalJsonError(
            "canonical JSON strings must encode as UTF-8"
        ) from exc


def record_digest(value: object) -> str:
    """Return the SHA-256 hex digest of a value's canonical JSON.

    Parameters
    ----------
    value : object
        A record or plain JSON data, as for :func:`canonical_json_bytes`.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _refuse_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object, refusing a repeated member name."""
    result: dict[str, object] = {}
    for key, item in pairs:
        if key in result:
            raise NonCanonicalJsonError(f"JSON member {key!r} appears twice")
        result[key] = item
    return result


def _parse_float(literal: str) -> float:
    """Decode a JSON number with a fraction or exponent, refusing negative zero."""
    value = float(literal)
    if value == 0.0 and literal.startswith("-"):
        raise NonCanonicalFloatError(f"JSON number {literal} is a negative zero")
    return value


def _parse_int(literal: str) -> int:
    """Decode a JSON integer, refusing ``-0``."""
    value = int(literal)
    if value == 0 and literal.startswith("-"):
        raise NonCanonicalFloatError("JSON number -0 is a negative zero")
    return value


def _refuse_constant(name: str) -> object:
    """Refuse the non-standard ``NaN`` and ``Infinity`` JSON constants."""
    raise NonCanonicalJsonError(f"JSON constant {name} is not allowed")


def strict_json_loads(raw: bytes) -> object:
    """Decode JSON bytes strictly.

    Parameters
    ----------
    raw : bytes
        UTF-8 JSON text.

    Returns
    -------
    object
        The decoded value.

    Raises
    ------
    NonCanonicalJsonError
        On invalid UTF-8, invalid JSON, a duplicate member name, a ``NaN``
        or ``Infinity`` constant, a string escape that decodes to a lone
        surrogate, or a negative-zero number token
        (:class:`~.errors.NonCanonicalFloatError`).
    """
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_refuse_duplicates,
            parse_constant=_refuse_constant,
            parse_float=_parse_float,
            parse_int=_parse_int,
        )
        json.dumps(value, ensure_ascii=False).encode("utf-8")
        return value
    except UnicodeEncodeError as exc:
        raise NonCanonicalJsonError(
            "JSON holds a string that is not valid Unicode"
        ) from exc
    except UnicodeDecodeError as exc:
        raise NonCanonicalJsonError("JSON is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise NonCanonicalJsonError(f"JSON is malformed at offset {exc.pos}") from exc


def _build(annotation: object, value: object, path: str) -> object:
    """Convert decoded JSON ``value`` to ``annotation``, refusing coercion."""
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return _build_record(annotation, value, path)
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Literal:
        if not any(type(value) is type(choice) and value == choice for choice in args):
            raise NonCanonicalJsonError(f"{path} must be one of {list(args)}")
        return value
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        members = [arg for arg in args if arg is not type(None)]
        if len(members) != 1:
            raise NonCanonicalJsonError(f"{path} has an unsupported union annotation")
        return _build(members[0], value, path)
    if origin is tuple:
        if not isinstance(value, list):
            raise NonCanonicalJsonError(f"{path} must be an array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_build(args[0], item, f"{path}[]") for item in value)
        if len(value) != len(args):
            raise NonCanonicalJsonError(f"{path} must have {len(args)} items")
        return tuple(
            _build(arg, item, f"{path}[]")
            for arg, item in zip(args, value, strict=True)
        )
    if annotation in (str, int, float, bool):
        if type(value) is not annotation:
            name = getattr(annotation, "__name__", "the annotated type")
            raise NonCanonicalJsonError(f"{path} must be {name}")
        return value
    raise NonCanonicalJsonError(f"{path} has an unsupported annotation")


def _build_record(cls: type, value: object, path: str) -> object:
    """Build a dataclass from a JSON object with exactly its field names."""
    if not isinstance(value, dict):
        raise NonCanonicalJsonError(f"{path} must be a JSON object")
    names = [field.name for field in dataclasses.fields(cls)]
    if set(value) != set(names):
        missing, extra = (
            sorted(set(names) - set(value)),
            sorted(set(value) - set(names)),
        )
        raise NonCanonicalJsonError(
            f"{path} keys differ: missing {missing}, extra {extra}"
        )
    hints = get_type_hints(cls)
    return cls(
        **{name: _build(hints[name], value[name], f"{path}.{name}") for name in names}
    )


def decode_record(
    cls: type[RecordT], raw: bytes, validate: Callable[[RecordT], object]
) -> RecordT:
    """Decode JSON bytes into a record and run its validator.

    Parameters
    ----------
    cls : type
        The record dataclass.
    raw : bytes
        UTF-8 JSON text of one record.
    validate : callable
        The record's validator, run on the decoded record.

    Returns
    -------
    RecordT
        The decoded and validated record.

    Raises
    ------
    NonCanonicalJsonError
        If the bytes break the strict decoding rules or the record's shape.
    """
    record = cast(RecordT, _build_record(cls, strict_json_loads(raw), cls.__name__))
    validate(record)
    return record
