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
- **Opt-in streaming contradiction halt**: completed streamed claims can be
  checked against retrieved grounding facts; this is an interlock for grounded
  streaming deployments, not the sole production gate
- **Safety kernel**: hardware-level output interlock with emergency stop
- **Two-stage prompt injection detection**: Stage 1 — `InputSanitizer` detects instruction overrides, role-play injections, delimiter tricks, output manipulation, and data exfiltration attempts; scrubs null bytes, control chars, and homoglyphs. Stage 2 — `InjectionDetector` measures output divergence from original intent via bidirectional NLI scoring; catches injection effects regardless of encoding; per-claim attribution with grounded/drifted/injected verdicts
- **YAML policy engine**: `Policy` blocks forbidden phrases, enforces length limits, requires citations, and evaluates custom regex rules
- **Multi-tenant isolation**: `TenantRouter` guarantees per-tenant KB separation with thread-safe access
- **Structured audit trail**: `AuditLogger` writes JSONL with SHA-256 query hashes (never plaintext queries) for compliance and forensic review
- **Minimal dependencies**: core requires only numpy and requests
- **No pickle.load of untrusted data** in any module
- **CI security audit**: `pip-audit` runs on every push

## Licensing

Director-AI is open core:

1. **Core — Apache-2.0**: the guardrail engine and supporting surfaces are
   permissively licensed and free for any use, including production and
   closed-source products, with no source-disclosure obligation.
2. **Advanced & Labs — BUSL-1.1**: source-available; free for non-production and
   evaluation, with each file converting to Apache-2.0 on its change date.
   Production or hosted use of the advanced tier needs a commercial license —
   contact protoscience@anulum.li.
3. **Dependency compatibility**: all runtime dependencies are permissively
   licensed (MIT/Apache-2.0/BSD), compatible with both tiers.

## Known Limitations

- No third-party security audit.
- Heuristic scorer (without NLI model) is deterministic and trivially bypassed.

External security test packet:
[`security/EXTERNAL_SECURITY_TEST_PACKET.md`](security/EXTERNAL_SECURITY_TEST_PACKET.md).
Execution gate:
[`security/EXTERNAL_SECURITY_TEST_RUNBOOK.md`](security/EXTERNAL_SECURITY_TEST_RUNBOOK.md).

## Known Open Advisories (no upstream fix available)

Two transitive dependencies carry advisories for which **no patched release
exists**, so they cannot be resolved by upgrade. Both are assessed as
**not exploitable in Director-AI's execution path**; each is left as an open,
tracked Dependabot alert and will be upgraded the moment a fixed version ships.

### chromadb — CVE-2026-45829 / GHSA-f4j7-r4q5-qw2c (critical)

Pre-authentication code injection via the ChromaDB **server's** collections
endpoint when a request supplies a malicious model repository with
`trust_remote_code=true`. Vulnerable range `>=1.0.0, <=1.5.9`; the latest PyPI
release (1.5.9, re-verified from PyPI on 2026-06-18) is the top of that range,
so there is no fixed version.

**Exposure: effectively nil.** chromadb is an optional vector backend. The
`ChromaBackend` integration uses chromadb in **embedded** mode only
(`PersistentClient(path=…)` / in-memory `Client()`), never `HttpClient` against a
running server and never `trust_remote_code` — the vulnerable code path (the
server's HTTP collections endpoint) is not reachable from Director-AI. This
embedded-only boundary is covered by vector-store unit tests so future Chroma
adapter changes cannot accidentally switch to the server client path.

Dependabot PR #114 is not a safe remediation for this advisory: it bumps
chromadb only to 1.5.9, still inside the vulnerable no-fix range, and its
generated lockfile is invalid for the CI `uv` parser. Keep the alert open until
an upstream fixed release exists, then regenerate and verify the vector extra
lock from current `main`.

### torch — GHSA-rrmf-rvhw-rf47 (low)

Memory corruption through `torch.jit.script`. Vulnerable range `<= 2.12.0`; no
patched release exists.

**Exposure: effectively nil.** Director-AI does not call `torch.jit.script` (the
only reference is a deprecation-warning suppression comment, not an invocation),
so the vulnerable function is never reached.

When the NLI model repository itself becomes unavailable on the Hub, the opt-in
**fallback model registry** (`DirectorConfig.model_fallback_enabled`) degrades to
a vetted, revision-pinned alternate model rather than failing — narrowing the
supply-chain availability surface.

## Residual Risks (documented for transparency)

### Regex-based injection detection (Stage 1) bypass

``InputSanitizer`` Stage 1 uses regex pattern matching. Before matching it
**defangs** the input — NFKC normalisation, null/control/zero-width stripping,
Cyrillic/Greek homoglyph folding to ASCII — and additionally scans the
ROT13-decoded form, so the classic literal-evasion vectors are caught at Stage 1:
- Unicode homoglyphs (Cyrillic а vs Latin a) — folded before matching
- Zero-width characters inserted between keywords — stripped before matching
- ROT13-encoded instructions — re-scanned in clear
- Base64 payloads — detected by a dedicated decoder check

The folding is conservative (only ASCII-confusable code points), so benign Latin
and non-Latin prose is not false-halted. The residual Stage 1 gap is **semantic /
prompt-level obfuscation** (indirect references that contain no literal attack
phrase in any encoding).

**Mitigation for the residual**: Stage 2 (``InjectionDetector``) uses bidirectional
NLI divergence scoring to detect the *effect* of injection regardless of phrasing,
and the optional model-backed Stage 1 classifier (``prompt_guard``) adds
adaptive-attack coverage. Stage 1 remains a fast filter, not the sole defence —
enable all stages for production.

### Knowledge base poisoning

If an attacker can modify KB entries (e.g., via an unprotected
ingestion API), they can insert false "ground truth" that the scorer
will validate against. Hallucinated outputs matching poisoned KB
entries will score as grounded.

**Mitigation**: KB writes support HMAC-signed entries with tamper
detection. Set ``knowledge_write_require_signature=True`` and supply
``knowledge_write_hmac_keys`` so writes with a missing or invalid
signature are rejected; ``production_mode`` forces signature
enforcement on. Combine
with ``TenantRouter`` strict ACLs on KB writes and ``AuditLogger`` to
detect unexpected modifications. The opt-in pre-model **evidence
firewall** additionally screens retrieved chunks before they reach the
model — checking tenant match, provenance, signature, content hash,
expiry, source owner, sensitivity, allowed use case, and poisoning
heuristics — and quarantines failing chunks.

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
``use_model=False`` fallback available. The opt-in fallback model registry
(``model_fallback_enabled``) degrades to a vetted, revision-pinned alternate
model if the primary is delisted on the Hub. SBOM generation in release
pipeline. Sigstore signing of published packages. Consider airgapped
deployment for highest-security environments. Currently open, no-fix
advisories are tracked under *Known Open Advisories* above.

Deployment notes for torch, transformers, ONNX Runtime, Chroma, and other
heavy optional packages live in
``docs-site/deployment/supply-chain.md``.

### Physical-action residual risks

Cyber-physical hooks can screen proposed actions before they reach a robot,
simulator, or actuator gateway, but they do not make unsafe hardware safe by
themselves. Residual risks include:

- **Hardware damage**: a caller can still request a physically unsafe move if
  the deployment has incomplete constraints, stale world state, or incorrect
  actuator calibration.
- **Malformed action payloads**: invalid coordinates, oversized vectors, or
  unexpected actuator ids can stress adapters if callers bypass
  ``PhysicalAction`` validation.
- **Expensive solver payloads**: repeated inverse-kinematics or collision checks
  can exhaust simulator or robotics runtimes without per-tenant budgets.
- **Simulator dependency isolation**: ROS 2, MuJoCo, CARLA, and similar stacks
  bring large native dependency surfaces and should not run in the default web
  API process.

**Mitigation**: Physical hooks are warn-only by default. Blocking real-world
actions requires both ``physical_action_mode="block"`` and
``allow_physical_action_blocking=True``. Use ``TenantPhysicalBudget`` to cap
action validation, inverse-kinematics, and simulation checks per tenant. Install
simulator stacks only in an isolated ``director-ai[physical]`` runtime, keep
hardware drivers behind a local gateway, and require an external emergency stop
outside Director-AI for live robots or machinery.
