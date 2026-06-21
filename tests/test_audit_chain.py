# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — audit hash-chain tests
"""Tamper-evidence tests for the AuditLogger hash chain.

Covers chain-field population, first-entry zero parent and entry-to-entry
linkage, a clean verify, and every tamper vector the chain must catch: an edited
payload field, a forged entry_hash, a deleted/reordered record, and a forged
chain tag (which needs the secret). Also covers cross-restart chain continuity
over an existing file, the no-path guard, verification under a wrong secret, and
that concurrent writers still produce a verifiable chain.
"""

from __future__ import annotations

import json
import threading

import pytest

from director_ai.core.safety.audit import _ZERO_HASH, AuditLogger

_SECRET = "unit-test-audit-secret-key-which-is-long-enough"


def _logger(tmp_path, name="a.jsonl"):
    return AuditLogger(path=tmp_path / name, hmac_secret=_SECRET)


def _log(logger, query="q", response="resp", approved=True, score=0.9):
    return logger.log_review(
        query=query, response=response, approved=approved, score=score
    )


def _read(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestChainPopulation:
    def test_fields_populated(self, tmp_path):
        entry = _log(_logger(tmp_path))
        assert entry.prev_hash == _ZERO_HASH
        assert len(entry.entry_hash) == 64
        assert len(entry.chain_tag) == 64

    def test_links_to_previous(self, tmp_path):
        logger = _logger(tmp_path)
        first = _log(logger, query="one")
        second = _log(logger, query="two")
        assert second.prev_hash == first.entry_hash
        assert first.prev_hash == _ZERO_HASH

    def test_distinct_payloads_distinct_hashes(self, tmp_path):
        logger = _logger(tmp_path)
        a = _log(logger, query="one")
        b = _log(logger, query="two")
        assert a.entry_hash != b.entry_hash


class TestVerifyClean:
    def test_clean_chain_verifies(self, tmp_path):
        logger = _logger(tmp_path)
        for i in range(5):
            _log(logger, query=f"q{i}")
        assert logger.verify_chain() == (True, None)

    def test_verify_accepts_explicit_path(self, tmp_path):
        logger = _logger(tmp_path)
        _log(logger)
        # A fresh logger with the same secret can verify the file.
        verifier = AuditLogger(hmac_secret=_SECRET)
        assert verifier.verify_chain(tmp_path / "a.jsonl") == (True, None)

    def test_verify_without_path_raises(self):
        with pytest.raises(ValueError, match="needs a path"):
            AuditLogger(hmac_secret=_SECRET).verify_chain()

    def test_verify_chain_tolerates_blank_lines(self, tmp_path):
        logger = _logger(tmp_path)
        for i in range(3):
            _log(logger, query=f"q{i}")
        path = tmp_path / "a.jsonl"
        # A blank line in the trail must be skipped, not fail verification.
        path.write_text("\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
        assert logger.verify_chain(path) == (True, None)


class TestChainResume:
    def test_blank_file_resumes_from_zero_hash(self, tmp_path):
        path = tmp_path / "blank.jsonl"
        path.write_text("   \n\n", encoding="utf-8")
        logger = AuditLogger(path=path, hmac_secret=_SECRET)
        assert logger._prev_hash == _ZERO_HASH

    def test_corrupt_last_line_resumes_from_zero_hash(self, tmp_path):
        path = tmp_path / "corrupt.jsonl"
        path.write_text("this is not valid json\n", encoding="utf-8")
        logger = AuditLogger(path=path, hmac_secret=_SECRET)
        assert logger._prev_hash == _ZERO_HASH


class TestTamperDetection:
    def _write_back(self, path, records):
        path.write_text(
            "\n".join(json.dumps(r, separators=(",", ":")) for r in records) + "\n",
            encoding="utf-8",
        )

    def test_edited_payload_detected(self, tmp_path):
        logger = _logger(tmp_path)
        _log(logger, query="one")
        _log(logger, query="two")
        path = tmp_path / "a.jsonl"
        records = _read(path)
        records[1]["approved"] = not records[1]["approved"]  # tamper payload
        self._write_back(path, records)
        assert logger.verify_chain(path) == (False, 1)

    def test_forged_entry_hash_detected(self, tmp_path):
        logger = _logger(tmp_path)
        _log(logger)
        path = tmp_path / "a.jsonl"
        records = _read(path)
        records[0]["entry_hash"] = "f" * 64
        self._write_back(path, records)
        assert logger.verify_chain(path) == (False, 0)

    def test_deleted_record_breaks_linkage(self, tmp_path):
        logger = _logger(tmp_path)
        _log(logger, query="one")
        _log(logger, query="two")
        _log(logger, query="three")
        path = tmp_path / "a.jsonl"
        records = _read(path)
        del records[1]  # remove the middle entry
        self._write_back(path, records)
        # The (former) third entry's prev_hash no longer matches its predecessor.
        assert logger.verify_chain(path) == (False, 1)

    def test_reordered_records_detected(self, tmp_path):
        logger = _logger(tmp_path)
        _log(logger, query="one")
        _log(logger, query="two")
        path = tmp_path / "a.jsonl"
        records = _read(path)
        records.reverse()
        self._write_back(path, records)
        assert logger.verify_chain(path) == (False, 0)

    def test_forged_tag_detected(self, tmp_path):
        logger = _logger(tmp_path)
        _log(logger)
        path = tmp_path / "a.jsonl"
        records = _read(path)
        records[0]["chain_tag"] = "0" * 64
        self._write_back(path, records)
        assert logger.verify_chain(path) == (False, 0)

    def test_wrong_secret_fails_tag(self, tmp_path):
        _log(_logger(tmp_path))
        other = AuditLogger(hmac_secret="a-completely-different-secret-key-32bytes!!")
        # Content + linkage are intact, but the HMAC tag was keyed differently.
        assert other.verify_chain(tmp_path / "a.jsonl") == (False, 0)


class TestContinuityAndConcurrency:
    def test_continues_chain_across_restart(self, tmp_path):
        first = _logger(tmp_path)
        _log(first, query="one")
        last_hash = _log(first, query="two").entry_hash
        # New logger over the same file continues the chain.
        second = _logger(tmp_path)
        third = _log(second, query="three")
        assert third.prev_hash == last_hash
        assert second.verify_chain() == (True, None)

    def test_concurrent_writes_form_valid_chain(self, tmp_path):
        logger = _logger(tmp_path)

        def worker(n):
            for i in range(20):
                _log(logger, query=f"{n}-{i}")

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        records = _read(tmp_path / "a.jsonl")
        assert len(records) == 80
        assert logger.verify_chain() == (True, None)


class TestVerifyAuditCli:
    def test_cli_verifies_clean_log(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("DIRECTOR_AUDIT_HMAC_SECRET", _SECRET)
        path = tmp_path / "a.jsonl"
        logger = AuditLogger(path=path, hmac_secret=_SECRET)
        _log(logger)
        from director_ai.cli import main

        main(["verify-audit", str(path)])  # exit 0 → no SystemExit
        assert "VERIFIED" in capsys.readouterr().out

    def test_cli_flags_tampered_log(self, tmp_path, capsys):
        path = tmp_path / "a.jsonl"
        logger = AuditLogger(path=path, hmac_secret=_SECRET)
        _log(logger)
        records = _read(path)
        records[0]["score"] = 0.0  # tamper
        path.write_text(
            json.dumps(records[0], separators=(",", ":")) + "\n", encoding="utf-8"
        )
        from director_ai.cli import main

        with pytest.raises(SystemExit) as exc:
            main(["verify-audit", str(path), "--secret", _SECRET])
        assert exc.value.code == 1
        assert "TAMPERED" in capsys.readouterr().out

    def test_cli_missing_file(self, tmp_path):
        from director_ai.cli import main

        with pytest.raises(SystemExit) as exc:
            main(["verify-audit", str(tmp_path / "nope.jsonl")])
        assert exc.value.code == 1
