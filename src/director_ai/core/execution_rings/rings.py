# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Execution-ring taxonomy

"""Risk rings for agent actions, from read-only to exfiltration.

An LLM agent that has been prompt-injected will try to act. The execution-ring
model bounds the blast radius: each action is classified into an ordered ring,
and higher rings demand more human authorisation factors before the action runs,
so even a fully bypassed guardrail cannot delete data or exfiltrate it without
out-of-band human confirmation. :class:`ExecutionRing` is the ordered taxonomy;
:data:`RING_REQUIRED_FACTORS` maps each ring to its cumulative factor set.
"""

from __future__ import annotations

from enum import IntEnum

from .authorization import AuthorizationFactor

__all__ = ["RING_REQUIRED_FACTORS", "ExecutionRing", "classify_operation"]


class ExecutionRing(IntEnum):
    """The risk ring of an agent action; higher ordinals demand more factors."""

    READ = 0
    """Read-only access — no side effect (get, list, search)."""

    WRITE = 1
    """Creates or mutates state (create, update, append)."""

    DELETE = 2
    """Destroys state (delete, drop, purge)."""

    EXECUTE = 3
    """Runs code or invokes a tool/command (execute, run, shell)."""

    EXFILTRATE = 4
    """Moves data outside the trust boundary (export, send, upload)."""


# Cumulative authorisation factors required per ring (Microsoft execution-rings).
RING_REQUIRED_FACTORS: dict[ExecutionRing, frozenset[AuthorizationFactor]] = {
    ExecutionRing.READ: frozenset(),
    ExecutionRing.WRITE: frozenset({AuthorizationFactor.OPERATOR_APPROVAL}),
    ExecutionRing.DELETE: frozenset(
        {
            AuthorizationFactor.OPERATOR_APPROVAL,
            AuthorizationFactor.COOLING_PERIOD,
        }
    ),
    ExecutionRing.EXECUTE: frozenset(
        {
            AuthorizationFactor.OPERATOR_APPROVAL,
            AuthorizationFactor.COOLING_PERIOD,
            AuthorizationFactor.SECOND_OPERATOR,
        }
    ),
    ExecutionRing.EXFILTRATE: frozenset(
        {
            AuthorizationFactor.OPERATOR_APPROVAL,
            AuthorizationFactor.COOLING_PERIOD,
            AuthorizationFactor.SECOND_OPERATOR,
            AuthorizationFactor.CISO_NOTIFICATION,
        }
    ),
}

# Operation-verb keywords mapped to a ring. Checked highest-ring-first so a
# compound verb ("export and delete") is classified by its most dangerous part.
_RING_KEYWORDS: tuple[tuple[ExecutionRing, frozenset[str]], ...] = (
    (
        ExecutionRing.EXFILTRATE,
        frozenset(
            {
                "export",
                "send",
                "email",
                "upload",
                "transmit",
                "exfiltrate",
                "publish",
                "share",
                "leak",
                "post",
            }
        ),
    ),
    (
        ExecutionRing.EXECUTE,
        frozenset(
            {
                "execute",
                "exec",
                "run",
                "shell",
                "invoke",
                "spawn",
                "eval",
                "compile",
                "deploy",
            }
        ),
    ),
    (
        ExecutionRing.DELETE,
        frozenset(
            {
                "delete",
                "drop",
                "purge",
                "remove",
                "destroy",
                "truncate",
                "wipe",
                "erase",
            }
        ),
    ),
    (
        ExecutionRing.WRITE,
        frozenset(
            {
                "write",
                "create",
                "update",
                "insert",
                "append",
                "modify",
                "set",
                "patch",
                "put",
                "rename",
                "move",
            }
        ),
    ),
    (
        ExecutionRing.READ,
        frozenset(
            {
                "read",
                "get",
                "list",
                "search",
                "query",
                "fetch",
                "view",
                "select",
                "describe",
                "show",
            }
        ),
    ),
)


def classify_operation(operation: str) -> ExecutionRing:
    """Classify an operation verb/name into its :class:`ExecutionRing`.

    Tokenises on non-alphanumeric boundaries and matches highest-ring-first, so
    the most dangerous verb in a compound operation wins. Unknown operations
    default to :attr:`ExecutionRing.EXECUTE` — fail-closed: an unrecognised
    action is treated as high-risk, not waved through as a read.
    """
    tokens = {t for t in _tokenise(operation)}
    for ring, keywords in _RING_KEYWORDS:
        if tokens & keywords:
            return ring
    return ExecutionRing.EXECUTE


def _tokenise(operation: str) -> list[str]:
    token = []
    out: list[str] = []
    for ch in operation.lower():
        if ch.isalnum():
            token.append(ch)
        elif token:
            out.append("".join(token))
            token = []
    if token:
        out.append("".join(token))
    return out
