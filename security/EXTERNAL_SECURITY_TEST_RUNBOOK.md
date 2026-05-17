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
- `physical_hooks`
- `attestation`
- `cross_language_trust_boundary`

## Evidence Directory

The tester returns one directory named `security-validation/` with:

| Path | Required content |
|---|---|
| `environment.json` | Target commit, package version, Python, platform, enabled extras, config fingerprint, tester, start time, completion time |
| `http_transcripts/` | Redacted HTTP request and response captures |
| `websocket_frames.jsonl` | Accepted, rejected, halted, and cancelled stream frames |
| `tenant_matrix.csv` | At least two tenants, including an allowed same-tenant case and a denied cross-tenant isolation case |
| `ingestion_matrix.csv` | Tenant, case, expected status, and actual status |
| `physical_matrix.csv` | Tenant, case, expected decision, and actual decision |
| `attestation_matrix.csv` | Issuer, case, expected status, and actual status |
| `contract_matrix.csv` | Boundary, case, expected status, and actual status |
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
- `http_transcripts/` contains no redacted transcript file;
- matrix files have missing columns or no rows;
- the tenant matrix does not include at least two tenants and a denied
  isolation case;
- physical, attestation, or cross-language evidence matrices are absent;
- finding records reference missing evidence paths or paths outside the
  evidence directory;
- summary text omits a required track id or the exact `target_commit` recorded
  in `environment.json`;
- returned files contain unredacted bearer, cookie, or credential-header markers.

## Completion Rule

Do not mark the ROADMAP item complete until:

1. the tester is independent from the implementer of the assessed changes;
2. the returned evidence validates without errors;
3. every high or critical finding has a fix commit or an accepted risk entry;
4. the final report names the tested commit and optional extras;
5. a follow-up validation run confirms fixes against the same tracks.
