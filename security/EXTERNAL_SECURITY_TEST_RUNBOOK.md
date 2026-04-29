<!--
SPDX-License-Identifier: AGPL-3.0-or-later
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

## Evidence Directory

The tester returns one directory named `security-validation/` with:

| Path | Required content |
|---|---|
| `environment.json` | Target commit, package version, Python, platform, enabled extras, config fingerprint, tester, start time, completion time |
| `http_transcripts/` | Redacted HTTP request and response captures |
| `websocket_frames.jsonl` | Accepted, rejected, halted, and cancelled stream frames |
| `tenant_matrix.csv` | Tenant, surface, action, expected status, and actual status |
| `ingestion_matrix.csv` | Tenant, case, expected status, and actual status |
| `findings.jsonl` | Finding severity, surface, reproduction, and evidence path |
| `summary.md` | Per-track pass/fail summary and target commit |

## Validation

Run:

```bash
python tools/validate_external_security_run.py security-validation
```

The validator rejects the evidence when:

- a required file or directory is missing;
- JSON, JSONL, or CSV files do not parse;
- required WebSocket frame classes are absent;
- matrix files have missing columns or no rows;
- finding records reference missing evidence paths;
- summary text omits a required track id or target commit;
- returned files contain unredacted bearer, cookie, or credential-header markers.

## Completion Rule

Do not mark the ROADMAP item complete until:

1. the tester is independent from the implementer of the assessed changes;
2. the returned evidence validates without errors;
3. every high or critical finding has a fix commit or an accepted risk entry;
4. the final report names the tested commit and optional extras;
5. a follow-up validation run confirms fixes against the same tracks.
