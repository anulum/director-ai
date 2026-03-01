# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.7.x   | Yes |
| < 1.7   | No  |

Only the latest release receives security fixes.

## Reporting a Vulnerability

1. **Email:** protoscience@anulum.li
2. **Subject:** `[SECURITY] Director-AI — <brief description>`
3. **Do not** open a public GitHub issue for security vulnerabilities.

We will acknowledge receipt within 48 hours and aim to provide a fix within
7 days for critical issues.

## Scope

Security concerns for Director-AI:

- **Prompt injection**: adversarial inputs designed to bypass coherence oversight
- **Metric evasion**: inputs crafted to produce high coherence scores for
  hallucinated outputs (false negatives)
- **Knowledge base poisoning**: malicious entries that corrupt factual scoring
- **Model deserialization**: unsafe loading of NLI model weights
- **Dependency supply chain**: compromised upstream packages

## Security Measures

- **Dual-entropy scoring**: NLI contradiction detection + RAG fact-checking
- **Streaming halt**: token-level coherence monitoring with three halt mechanisms
- **Safety kernel**: hardware-level output interlock with emergency stop
- **Prompt injection hardening**: `InputSanitizer` detects instruction overrides, role-play injections, delimiter tricks, output manipulation, and data exfiltration attempts; scrubs null bytes, control chars, and homoglyphs
- **YAML policy engine**: `Policy` blocks forbidden phrases, enforces length limits, requires citations, and evaluates custom regex rules
- **Multi-tenant isolation**: `TenantRouter` guarantees per-tenant KB separation with thread-safe access
- **Structured audit trail**: `AuditLogger` writes JSONL with SHA-256 query hashes (never plaintext queries) for compliance and forensic review
- **Minimal dependencies**: core requires only numpy and requests
- **No pickle.load of untrusted data** in any module
- **CI security audit**: `pip-audit` runs on every push

## Known Limitations

- No fuzzing harness. Adversarial prompt testing is manual.
- No third-party security audit.
- Heuristic scorer (without NLI model) is deterministic and trivially bypassed.
