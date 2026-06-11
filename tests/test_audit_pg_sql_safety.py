# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Postgres audit SQL safety tests

"""Structural SQL safety tests for the enterprise audit sink."""

from __future__ import annotations

import pytest

from director_ai.enterprise.audit_pg import (
    PostgresAuditSink,
    _quote_identifier,
    _validate_identifier,
)


@pytest.mark.parametrize(
    "identifier",
    [
        "audit_logs; DROP TABLE audit_logs",
        "audit-logs",
        "audit.logs",
        " audit_logs",
        "1audit_logs",
        "audit_logs/*",
        'audit_logs"',
        "a" * 64,
    ],
)
def test_rejects_injection_shaped_table_identifiers_before_sql(identifier):
    with pytest.raises(ValueError, match="Invalid table_name"):
        PostgresAuditSink("sqlite:///:memory:", table_name=identifier)


def test_quote_identifier_validates_before_quoting():
    with pytest.raises(ValueError, match="Invalid identifier"):
        _quote_identifier("audit_logs; DROP TABLE audit_logs")


def test_accepts_and_quotes_structural_identifier():
    assert _validate_identifier("_tenant_audit_2026", "table_name") == (
        "_tenant_audit_2026"
    )
    assert _quote_identifier("_tenant_audit_2026") == '"_tenant_audit_2026"'
