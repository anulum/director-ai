# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 3.x     | Yes |
| < 3.0   | No  |

Only the latest release receives security fixes.

## Reporting a Vulnerability

1. **GitHub Security Advisories** (preferred): [Report here](https://github.com/anulum/director-ai/security/advisories/new)
2. **Email:** protoscience@anulum.li
3. **Subject:** `[SECURITY] Director-AI — <brief description>`
4. **Do not** open a public GitHub issue for security vulnerabilities.

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
- **Two-stage prompt injection detection**: Stage 1 — `InputSanitizer` detects instruction overrides, role-play injections, delimiter tricks, output manipulation, and data exfiltration attempts; scrubs null bytes, control chars, and homoglyphs. Stage 2 — `InjectionDetector` measures output divergence from original intent via bidirectional NLI scoring; catches injection effects regardless of encoding; per-claim attribution with grounded/drifted/injected verdicts
- **YAML policy engine**: `Policy` blocks forbidden phrases, enforces length limits, requires citations, and evaluates custom regex rules
- **Multi-tenant isolation**: `TenantRouter` guarantees per-tenant KB separation with thread-safe access
- **Structured audit trail**: `AuditLogger` writes JSONL with SHA-256 query hashes (never plaintext queries) for compliance and forensic review
- **Minimal dependencies**: core requires only numpy and requests
- **No pickle.load of untrusted data** in any module
- **CI security audit**: `pip-audit` runs on every push

## AGPL-3.0 Compliance

Director-AI is licensed under GNU AGPL v3. Key obligations:

1. **Source disclosure**: if you modify Director-AI and deploy it as a
   network service, you must make your modified source available to users
   of that service under the same license.
2. **Commercial alternative**: a commercial license is available for
   organisations that cannot comply with AGPL requirements. Contact
   protoscience@anulum.li.
3. **Dependency compatibility**: all runtime dependencies are
   permissively licensed (MIT/Apache-2.0/BSD). The AGPL obligation
   applies to Director-AI code, not to your application code that calls
   it through the public API.

## Known Limitations

- No third-party security audit.
- Heuristic scorer (without NLI model) is deterministic and trivially bypassed.

## Residual Risks (documented for transparency)

### Regex-based injection detection (Stage 1) bypass

``InputSanitizer`` Stage 1 uses regex pattern matching. Sophisticated
adversaries can bypass it via:
- Unicode homoglyphs (Cyrillic а vs Latin a)
- Zero-width characters inserted between keywords
- Base64 or ROT13 encoding of instructions
- Prompt-level obfuscation (indirect references)

**Mitigation**: Stage 2 (``InjectionDetector``) uses NLI divergence
scoring to detect the *effect* of injection in the output regardless
of encoding. The dual-stage design means Stage 1 is a fast filter,
not the primary defence. Enable both stages for production.

### Knowledge base poisoning

If an attacker can modify KB entries (e.g., via an unprotected
ingestion API), they can insert false "ground truth" that the scorer
will validate against. Hallucinated outputs matching poisoned KB
entries will score as grounded.

**Mitigation**: Use ``TenantRouter`` with strict ACLs on KB writes.
Enable ``AuditLogger`` to detect unexpected KB modifications. Use
signed/hashed KB entries for tamper detection (future roadmap).

### NLI model evasion

Adversaries can craft outputs that the NLI model fails to detect as
contradictions (adversarial examples). FactCG-DeBERTa-v3-Large is
robust for general text but may miss:
- Numerical inconsistencies (e.g., "100" vs "101")
- Subtle logical inversions in complex sentences
- Domain-specific terminology substitutions

**Mitigation**: Use the rules engine (Tier 2) for numeric consistency
checks. Enable ``AdversarialTester`` for red-teaming. Consider
multi-scorer consensus for high-stakes domains.

### Metric evasion in streaming mode

In token-level streaming, an adversary could front-load coherent
tokens to build trust, then inject hallucinated content after the
coherence window has shifted.

**Mitigation**: ``StreamingKernel`` uses adaptive window sizing and
three independent halt mechanisms. ``ContradictionTracker`` catches
cross-turn inconsistencies. Set ``hard_limit`` conservatively for
high-risk applications.

### Dependency supply chain

Despite SHA-pinned HuggingFace models and ``pip-audit`` in CI,
transitive dependencies (torch, transformers, ONNX) have a broad
attack surface. A compromised upstream package could execute arbitrary
code at model-load time.

**Mitigation**: ``MODEL_REGISTRY`` with pinned revision SHAs.
``use_model=False`` fallback available. SBOM generation in release
pipeline. Sigstore signing of published packages. Consider airgapped
deployment for highest-security environments.
