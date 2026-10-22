# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Strict event feedback JSON encoding

"""Decode original JSON without duplicate members or value coercion."""

from __future__ import annotations

import json
import math
from typing import TypeAlias

Json: TypeAlias = str | int | float | bool | None | list["Json"] | dict[str, "Json"]
Object: TypeAlias = dict[str, Json]


class FeedbackInvalidError(ValueError):
    """Reject unqualified feedback data with caller-safe fixed text."""

    def __init__(self) -> None:
        """Create a refusal without original query, paths or exception details."""
        super().__init__("Feedback evidence could not be validated.")


class FeedbackWriteUnknownError(RuntimeError):
    """Preserve uncertainty when post-write original custody cannot be completed."""

    def __init__(self) -> None:
        """Report uncertain durability without leaking private transport details."""
        super().__init__(
            "Feedback write remains uncertain; replay the original submission."
        )


def _members(pairs: list[tuple[str, Json]]) -> Object:
    result: Object = {}
    for name, value in pairs:
        if name in result:
            raise FeedbackInvalidError()
        result[name] = value
    return result


def _constant(_value: str) -> Json:
    raise FeedbackInvalidError()


def _finite_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise FeedbackInvalidError()
    return number


def decode(raw: bytes) -> Object:
    """Read one UTF-8 JSON object without duplicate or nonfinite values.

    Parameters
    ----------
    raw : bytes
        Complete original response or retained object bytes.

    Returns
    -------
    Object
        Parsed object retaining every original value.

    Raises
    ------
    FeedbackInvalidError
        If encoding, shape, member uniqueness or numeric values are invalid.
    """
    try:
        value: Json = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_members,
            parse_constant=_constant,
            parse_float=_finite_number,
        )
        if not isinstance(value, dict):
            raise FeedbackInvalidError()
        canonical(value)
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise FeedbackInvalidError() from exc


def canonical(value: Json) -> bytes:
    """Encode sorted compact UTF-8 JSON without normalising string values.

    Parameters
    ----------
    value : Json
        Strict protocol data to bind into a digest or signature.

    Returns
    -------
    bytes
        Canonical encoding shared by the wire contract.

    Raises
    ------
    FeedbackInvalidError
        If values cannot be represented as finite UTF-8 JSON.
    """
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (ValueError, UnicodeError, TypeError, RecursionError) as exc:
        raise FeedbackInvalidError() from exc


def text(value: Json) -> str:
    """Require an original nonblank string without stripping its contents.

    Parameters
    ----------
    value : Json
        Original protocol member.

    Returns
    -------
    str
        The exact original string.
    """
    if not isinstance(value, str) or not value.strip():
        raise FeedbackInvalidError()
    return value


def object_value(value: Json) -> Object:
    """Require an object instead of coercing a protocol value.

    Parameters
    ----------
    value : Json
        Original member expected to contain a mapping.

    Returns
    -------
    Object
        Original mapping.
    """
    if not isinstance(value, dict):
        raise FeedbackInvalidError()
    return value


def array(value: Json) -> list[Json]:
    """Require an array without accepting strings or other iterables.

    Parameters
    ----------
    value : Json
        Original member expected to contain ordered items.

    Returns
    -------
    list[Json]
        Original ordered items.
    """
    if not isinstance(value, list):
        raise FeedbackInvalidError()
    return value
