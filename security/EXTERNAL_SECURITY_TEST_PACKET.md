<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# External Security Test Packet

This packet gives an independent tester a bounded plan for Director-AI
security testing around streaming interception, multi-tenant isolation, and
knowledge-base ingestion.

Machine-readable packet:
`security/external_security_test_packet.toml`

Policy reference:
`SECURITY.md`

## Test Scope

| Track id | Surface | Primary code paths | Existing regression tests |
|---|---|---|---|
| `streaming_interception` | `/v1/stream` WebSocket | `server.py`, `StreamingKernel`, `AsyncStreamingKernel` | `test_server_ws_mux.py`, `test_cov_server_deep.py`, `test_coverage_streaming.py`, `test_async_streaming.py` |
| `multi_tenant_isolation` | tenant REST and gRPC routing | `server.py`, `grpc_server.py`, `TenantRouter` | `test_server_tenant.py`, `test_server_ws_mux.py`, `test_enterprise_modules.py`, `test_audit.py` |
| `knowledge_ingestion` | `/v1/knowledge/*` routes | `knowledge_api.py`, document registry, ingestion plugins | `test_coverage_knowledge_api.py`, `test_knowledge_edge_cases.py`, `test_cli_ingest_formats.py` |

## Streaming Interception Checks

The tester should verify:

1. WebSocket connections without a valid key close before application frames are
   accepted when keys are configured.
2. Claimed tenant ids cannot override key-to-tenant bindings.
3. Streaming oversight emits a halt frame when coherence falls below the hard
   limit.
4. Concurrent WebSocket sessions keep frames bound to the correct session id.
5. Cancellation prevents later result frames for the cancelled session.

Required evidence:

- handshake matrix;
- ordered WebSocket frame log;
- halt event sample;
- cancellation sample.

## Multi-Tenant Isolation Checks

The tester should verify:

1. Tenant A cannot list tenant B facts or documents.
2. Tenant A cannot write tenant B facts with a bound key.
3. Tenant A vector facts do not appear in tenant B vector search.
4. gRPC metadata cannot override configured key-to-tenant binding.
5. Audit rows include tenant ids without raw prompt text.

Required evidence:

- tenant access matrix;
- cross-tenant write attempts;
- vector-store partition sample;
- audit redaction sample.

## Knowledge-Base Ingestion Checks

The tester should verify:

1. Invalid tenant ids, document ids, sources, file types, content length values,
   and chunk overlap values are rejected.
2. Tenant A search does not return tenant B chunks.
3. Document replacement removes stale chunks before writing replacements.
4. Delete removes document metadata and reachable chunks.
5. Ingestion metadata does not store credentials from plugin clients.

Required evidence:

- ingestion rejection matrix;
- tenant-scoped search sample;
- update/delete trace;
- metadata redaction sample.

## Required Outputs

| Output path | Contents |
|---|---|
| `security-validation/environment.json` | Runtime, package, server config, and optional extras fingerprint |
| `security-validation/http_transcripts/` | HTTP requests and responses with credentials redacted |
| `security-validation/websocket_frames.jsonl` | Accepted, rejected, halted, and cancelled stream frames |
| `security-validation/tenant_matrix.csv` | Tenant read, write, list, stream, and gRPC access matrix |
| `security-validation/ingestion_matrix.csv` | Accepted and rejected ingestion payload cases |
| `security-validation/findings.jsonl` | Finding severity, surface, reproduction, and evidence path |
| `security-validation/summary.md` | Pass/fail summary per track, residual risk, and fixes |

## Report Rules

1. Do not include raw prompts, credentials, cookies, or bearer tokens in output
   files.
2. Preserve exact HTTP status codes, WebSocket close codes, and response bodies
   after redaction.
3. State which optional extras were installed for each run.
4. Mark skipped checks with the missing dependency or disabled config flag.
5. Include one replay command or script path for every finding.
