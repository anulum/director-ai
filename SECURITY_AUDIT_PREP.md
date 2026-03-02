# Security Audit Preparation — Director-AI v2.3.0

## Threat Model

### Assets
1. **User prompts/responses** — PII may be present in review payloads
2. **Knowledge base contents** — proprietary ground truth data
3. **NLI model weights** — large files, integrity matters
4. **API keys** — `X-API-Key` auth tokens, LLM provider keys
5. **Audit logs** — contain full review history
6. **Configuration** — thresholds affect safety-critical decisions

### Threat Actors
- **Malicious end-users** — prompt injection, data exfiltration via review API
- **Compromised LLM** — adversarial outputs designed to bypass scoring
- **Supply chain** — compromised PyPI dependency or model weights
- **Internal misconfiguration** — overly permissive thresholds, disabled NLI

### Attack Surfaces
| Surface | Entry Point | Risk |
|---------|-------------|------|
| REST API | `/v1/review`, `/v1/process` | Prompt injection, DoS |
| WebSocket | `/v1/stream` | Message flooding |
| gRPC | `DirectorService` RPCs | Deserialization attacks |
| CLI | `director-ai batch <file>` | Path traversal, file bombs |
| Config | `from_yaml()`, `from_env()` | YAML bomb, env injection |
| Knowledge base | `GroundTruthStore.add()` | Poisoning attacks |
| Dependencies | torch, transformers, onnxruntime | Known CVEs |

## Existing Mitigations

- [x] `InputSanitizer` with 9 injection patterns + Unicode detection
- [x] `X-API-Key` middleware with timing-safe comparison
- [x] Rate limiting via slowapi (`rate_limit_rpm`)
- [x] Input length limits (100KB prompts, 100MB batch files)
- [x] CORS origin validation
- [x] Null byte stripping and Unicode normalization
- [x] Non-root Docker user (`appuser`)
- [x] Bandit + Semgrep SAST in CI
- [x] `pip-audit` dependency scanning

## Needed Mitigations Checklist

- [ ] Path traversal detection in `InputSanitizer`
- [ ] YAML bomb rejection in `from_yaml()`
- [ ] JSON depth limit on API request bodies
- [ ] Excessive Unicode escape detection
- [ ] Request body size limit in FastAPI middleware
- [ ] Secrets rotation documentation
- [ ] SBOM generation (CycloneDX) in CI
- [ ] Fuzz testing with Hypothesis
- [ ] Content-Security-Policy headers on dashboard
- [ ] Audit log tamper detection (HMAC signing)

## SBOM Generation

```bash
pip install cyclonedx-bom
cyclonedx-py environment --output sbom.json --format json
```

CI generates SBOM as artifact on every push to main.

## Fuzzing Guidance

Property-based tests in `tests/test_fuzz.py` use Hypothesis to verify:
1. `CoherenceScorer.review()` never crashes on arbitrary Unicode strings
2. `InputSanitizer.sanitize()` / `.check()` never crash
3. All scores remain in [0, 1] regardless of input
4. NLI heuristic fallback handles extreme inputs

Run fuzzing locally:

```bash
pytest tests/test_fuzz.py -v --hypothesis-seed=0
```

For extended fuzzing (CI runs 200 examples, local can run more):

```bash
HYPOTHESIS_MAX_EXAMPLES=10000 pytest tests/test_fuzz.py -v
```

## Dependency Risk Assessment

| Dependency | Risk | Mitigation |
|-----------|------|------------|
| torch | Known CVEs, large attack surface | Pin version, `pip-audit` |
| transformers | Model loading from untrusted sources | Pin version, verify checksums |
| onnxruntime | Native code execution | Pin version, sandbox inference |
| fastapi/uvicorn | HTTP parsing | Pin version, SAST |
| grpcio | Protobuf deserialization | Pin version, message size limits |
| numpy | Buffer overflow history | Pin version |

## Audit Scope Recommendation

1. **API input validation** — all REST/gRPC/WebSocket entry points
2. **Prompt injection resistance** — InputSanitizer bypass testing
3. **Dependency audit** — full SBOM review + known CVE check
4. **Configuration hardening** — YAML/JSON parsing, env var injection
5. **Authentication/authorization** — API key handling, tenant isolation
6. **Denial of service** — batch processing limits, streaming timeouts
