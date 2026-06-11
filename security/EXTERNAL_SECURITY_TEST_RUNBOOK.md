<!--
SPDX-License-Identifier: Apache-2.0
Commercial licence available
Concepts 1996-2026 Miroslav Sotek. All rights reserved.
Code 2020-2026 Miroslav Sotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# External Security Test Runbook

This runbook defines the execution gate for the external security test packet.
The ROADMAP item is complete only after an independent tester returns evidence
that passes the validator in this repository.

Packet:
`security/EXTERNAL_SECURITY_TEST_PACKET.md`

Validator:
`tools/validate_external_security_run.py`

## Target

The external run must test the exact commit intended for release. The tester
records that commit in `security-validation/environment.json` and repeats it in
`security-validation/summary.md`.

Required tracks:

- `streaming_interception`
- `multi_tenant_isolation`
- `knowledge_ingestion`
- `physical_hooks`
- `attestation`
- `cross_language_trust_boundary`

## Evidence Directory

The tester returns one directory named `security-validation/` with:

| Path | Required content |
|---|---|
| `environment.json` | Full 40-hex target commit, package version, Python, platform, enabled extras, config fingerprint, tester, UTC start time, UTC completion time |
| `http_transcripts/` | Non-empty redacted HTTP request and response captures |
| `websocket_frames.jsonl` | Accepted, rejected, halted, and cancelled stream frames with non-empty session ids and no unknown frame classes |
| `tenant_matrix.csv` | At least two tenants, including an allowed same-tenant case and a denied cross-tenant isolation case |
| `ingestion_matrix.csv` | Tenant, case, expected status, and actual status |
| `physical_matrix.csv` | Tenant, case, expected decision, and actual decision |
| `attestation_matrix.csv` | Issuer, case, expected status, and actual status |
| `contract_matrix.csv` | Boundary, case, expected status, and actual status |
| `findings.jsonl` | Finding track id, severity, surface, reproduction, evidence path, and high/critical disposition |
| `summary.md` | Per-track pass/fail summary and target commit |

All returned files and directories must resolve inside the returned
`security-validation/` directory. Symlinks or path aliases that point outside
the evidence bundle are rejected.

For every CSV matrix, each `actual_*` outcome must match the corresponding
`expected_*` outcome after case and surrounding-whitespace normalisation. A
matrix row that records a failed expectation is rejected by the evidence
validator and must be moved into `findings.jsonl` with its reproduction and
redacted evidence path.

`summary.md` must include the target commit and one status line per track in
these exact shapes:

```text
target_commit: 0123456789abcdef0123456789abcdef01234567
- streaming_interception: pass
```

Accepted statuses are the lower-case tokens `pass`, `fail`, `blocked`, and
`skipped`; mixed-case variants are rejected.
Every `fail` status must have at least one `findings.jsonl` record with the
same `track_id` and severity above `info`. Every finding `track_id` must match
a packet track, and every finding `surface` must be listed under that packet
track. Tracks marked `pass` may only carry informational findings. Every
`high` or `critical` finding must include either a full 40-hex `fix_commit` or
a non-empty `accepted_risk` disposition that describes owner and rationale.
Each track may appear only once in `summary.md`, and track-shaped status lines
for unknown track ids are rejected. `pass` and `fail` status lines must contain
only the status token; `blocked` and `skipped` status lines must include a
reason after the status token. The space after each colon is required, and
target-commit alias lines such as `target_commit_sha` are rejected.

## Validation

Run:

```bash
python tools/validate_external_security_run.py security-validation
```

The validator rejects the evidence when:

- a required file or directory is missing;
- a required file, required directory, transcript file, finding evidence file,
  or auxiliary returned file resolves outside the evidence directory;
- JSON, JSONL, or CSV files do not parse;
- required WebSocket frame classes are absent, frame records use unknown frame
  classes, or frame records omit non-empty `type` or `session_id` values;
- `http_transcripts/` contains no redacted transcript file or contains a blank
  transcript file;
- matrix files have missing columns or no rows;
- matrix rows contain blank required cells;
- the tenant matrix does not include at least two tenants and a denied
  isolation case;
- physical, attestation, or cross-language evidence matrices are absent;
- `environment.json` identity fields or enabled-extra entries are blank,
  `target_commit` is not a full 40-hex git SHA, timestamps are not ISO-8601 UTC
  values, or `completed_at` is not after `started_at`;
- finding records omit `track_id`, use an unknown `track_id`, use a noncanonical
  severity token, contain empty `surface`, `reproduction`, or `evidence_path`
  fields, name a surface not declared for the finding track, reference
  directories, reference missing evidence files, or reference paths outside the
  evidence directory;
- summary text omits a required track id, omits a per-track status line, uses an
  unsupported status, duplicates a track status, marks a track `pass` while a
  non-info finding exists for that track, marks a track `fail` without a
  matching non-info finding, adds a reason to `pass` or `fail`, marks a track
  `blocked` or `skipped` without a reason, has zero or multiple `target_commit`
  lines, or does not exactly match the `target_commit` recorded in
  `environment.json`;
- returned files contain unredacted bearer, cookie, or credential-header markers.

## Completion Rule

Do not mark the ROADMAP item complete until:

1. the tester is independent from the implementer of the assessed changes;
2. the returned evidence validates without errors;
3. every high or critical finding has a fix commit or an accepted risk entry;
4. the final report names the tested commit and optional extras;
5. a follow-up validation run confirms fixes against the same tracks.
