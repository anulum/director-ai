# Security Policy

## Supported Versions

| Version | Supported          | Notes |
|---------|--------------------|-------|
| 0.2.x   | :white_check_mark: | Current development release |
| < 0.2   | :x:                | Pre-release / unreleased |

Only the latest release receives security fixes.

## Reporting a Vulnerability

If you discover a security vulnerability in Director-Class AI, please report it
responsibly:

1. **Email:** protoscience@anulum.li
2. **Subject:** `[SECURITY] Director-Class AI — <brief description>`
3. **Do not** open a public GitHub issue for security vulnerabilities.

We will acknowledge receipt within 48 hours and aim to provide a fix within
7 days for critical issues.

## Scope

Director-Class AI is an AI safety research prototype. Security concerns are
primarily:

- **Prompt injection**: Adversarial inputs designed to bypass the Director's
  entropy oversight or the Backfire Kernel safety mechanism
- **SEC metric evasion**: Inputs crafted to produce low-entropy scores for
  harmful outputs (false negatives)
- **RAG poisoning**: Malicious entries in the knowledge base that corrupt
  factual entropy calculations
- **Model deserialization**: Unsafe loading of NLI model weights or pickled
  objects
- **Dependency supply chain**: Compromised upstream packages (transformers,
  torch, numpy, scipy)

## Security Measures in Place

### Entropy Oversight (v0.2.0)
The Director Module implements dual-entropy monitoring:
- **Logical Entropy**: NLI-based contradiction detection
- **Factual Entropy**: RAG-based ground truth verification
- **SEC Threshold**: Hard cutoff at 0.6 — any action below triggers halt

### Backfire Kernel (v0.2.0)
Safety gate simulation that can sever the token stream
when entropy exceeds the safety threshold (0.5 hard limit).

### Dependency Auditing
- Dependencies are minimal (`numpy`, `scipy`, `torch`, `transformers`)
- No `pickle.load` of untrusted data in any module
- Model weights loaded only from trusted sources (Hugging Face Hub)

## Known Limitations

- **No fuzzing harness yet.** Adversarial prompt testing is manual.
- **No third-party security audit.** The codebase has not been reviewed by
  an external security firm.
- **Mock NLI in prototype.** The mock entropy calculations are deterministic
  and can be trivially bypassed. Production deployment requires real NLI models.
- **No CVE history.** No vulnerabilities have been reported to date.

Contributions to improve security coverage (adversarial test suites, model
robustness evaluation, formal verification) are welcome.

## Disclosure Timeline

| Date | Event |
|------|-------|
| 2025-12-29 | v0.1.0 initial prototype |
| 2026-01-21 | v0.2.0 Consilium subsystem added |
| 2026-02-15 | Repository professionalized, dual-license established |
