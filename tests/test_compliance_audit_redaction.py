# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SEC-2 compliance-audit PII redaction at the sink
"""Real-surface coverage for opt-in PII redaction in the compliance audit log.

The compliance trail is durable and tamper-evident (SHA-256 content hash +
prev_hash linkage + HMAC chain tag), so raw PII written into it would persist
forever inside the seal. SEC-2 masks the prompt and response **before** they are
sealed when a redactor is supplied (opt-in; the wiring layer enables it via
``redact_pii``). These tests exercise the real ``AuditLog`` against the real
``PIIRedactor`` — no mocks — and prove: the default retains raw content, an
enabled redactor masks both fields before the seal, the caller's entry is not
mutated, and the seal covers the stored (redacted) content so it verifies (and
detects tampering) without the redactor present at read time.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from director_ai.compliance.audit_log import AuditEntry, AuditLog
from director_ai.core.redactor import PIIRedactor

_EMAIL = "jane.doe@example.com"
_SSN = "123-45-6789"


def _entry(prompt: str, response: str) -> AuditEntry:
    """Build a scored interaction with sensible compliance defaults."""
    return AuditEntry(
        prompt=prompt,
        response=response,
        model="test-model",
        provider="test",
        score=0.9,
        approved=True,
        verdict_confidence=0.8,
        task_type="review",
        domain="",
        latency_ms=1.5,
        timestamp=1_700_000_000.0,
        tenant_id="tenant-a",
    )


def test_default_audit_log_retains_raw_prompt_and_response(tmp_path: Path) -> None:
    """With no redactor the compliance trail stores the raw interaction (opt-out)."""
    log = AuditLog(str(tmp_path / "audit.db"), hmac_secret="s")
    log.log(_entry(f"email {_EMAIL}", f"ssn {_SSN}"))
    rows = log.query()
    assert rows[0].prompt == f"email {_EMAIL}"
    assert rows[0].response == f"ssn {_SSN}"
    assert log.verify_chain() == (True, None)
    log.close()


def test_redactor_masks_both_fields_before_seal(tmp_path: Path) -> None:
    """An enabled redactor masks prompt and response; raw PII never lands on disk."""
    log = AuditLog(
        str(tmp_path / "audit.db"),
        hmac_secret="s",
        redactor=PIIRedactor(enabled=True),
    )
    log.log(_entry(f"contact {_EMAIL} please", f"his ssn is {_SSN}"))
    rows = log.query()
    assert rows[0].prompt == "contact [EMAIL] please"
    assert rows[0].response == "his ssn is [SSN]"
    assert _EMAIL not in rows[0].prompt
    assert _SSN not in rows[0].response
    # Redaction happens before the seal, so the sealed chain is still consistent.
    assert log.verify_chain() == (True, None)
    log.close()


def test_log_does_not_mutate_caller_entry(tmp_path: Path) -> None:
    """Redaction copies the entry; the caller's object keeps its raw values."""
    log = AuditLog(
        str(tmp_path / "audit.db"),
        hmac_secret="s",
        redactor=PIIRedactor(enabled=True),
    )
    entry = _entry(f"reach {_EMAIL}", "ok")
    log.log(entry)
    assert entry.prompt == f"reach {_EMAIL}"
    assert entry.response == "ok"
    log.close()


def test_disabled_redactor_is_passthrough(tmp_path: Path) -> None:
    """A disabled redactor is a no-op: identical to supplying no redactor."""
    log = AuditLog(
        str(tmp_path / "audit.db"),
        hmac_secret="s",
        redactor=PIIRedactor(enabled=False),
    )
    log.log(_entry(f"email {_EMAIL}", f"ssn {_SSN}"))
    rows = log.query()
    assert rows[0].prompt == f"email {_EMAIL}"
    assert rows[0].response == f"ssn {_SSN}"
    log.close()


def test_seal_covers_redacted_content_verifiable_without_redactor(
    tmp_path: Path,
) -> None:
    """The seal covers stored (redacted) content and verifies with no redactor."""
    db = str(tmp_path / "audit.db")
    writer = AuditLog(db, hmac_secret="shared", redactor=PIIRedactor(enabled=True))
    writer.log(_entry(f"first {_EMAIL}", "r1"))
    writer.log(_entry("second", f"ssn {_SSN}"))
    writer.close()

    # A reader that never sees the redactor re-derives the chain from the stored
    # redacted columns and confirms the seal.
    reader = AuditLog(db, hmac_secret="shared")
    assert reader.verify_chain() == (True, None)
    rows = reader.query()
    assert all(_EMAIL not in r.prompt for r in rows)
    assert all(_SSN not in r.response for r in rows)
    reader.close()


def test_tampering_with_redacted_row_breaks_the_seal(tmp_path: Path) -> None:
    """Editing a stored (already redacted) row is detected by verify_chain."""
    db = str(tmp_path / "audit.db")
    log = AuditLog(db, hmac_secret="shared", redactor=PIIRedactor(enabled=True))
    log.log(_entry(f"first {_EMAIL}", "r1"))
    log.log(_entry("second", "r2"))
    log.close()

    conn = sqlite3.connect(db)
    conn.execute("UPDATE audit_log SET prompt = 'tampered' WHERE id = 1")
    conn.commit()
    conn.close()

    reader = AuditLog(db, hmac_secret="shared")
    ok, bad_id = reader.verify_chain()
    assert ok is False
    assert bad_id == 1
    reader.close()
