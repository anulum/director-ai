# NOTICE — Licensing & Commercial Use

© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li

## Open-Core Licensing

Director-AI is open core. Every source file carries an SPDX
`SPDX-License-Identifier`, and the repository is
[REUSE](https://reuse.software/)-compliant (`reuse lint`).

### Core — Apache-2.0

The guardrail engine, 5-tier scoring (rules → embeddings → NLI), SDK guard,
FastAPI middleware, REST/gRPC server, injection detection, streaming halt, and
the agent/MCP preflight guard are licensed under
[Apache-2.0](LICENSES/Apache-2.0.txt). Free for any use, including production and
closed-source products, with no source-disclosure obligation.

### Advanced & Labs — BUSL-1.1

The advanced capabilities (under `core/<advanced>/`, `enterprise/`, `voice/`,
`ui/`, `experimental/`, `compliance/`, `agentic/`) are source-available under
[BUSL-1.1](LICENSES/BUSL-1.1.txt). Free for non-production and evaluation use;
each file converts to Apache-2.0 on its change date. A commercial license is
available for organisations that:

- Run the advanced tier in production or as a hosted/SaaS service
- Need SLA-backed support, indemnification, or custom builds

Contact [protoscience@anulum.li](mailto:protoscience@anulum.li) for terms.

## License Boundary

| Use Case | License |
|----------|---------|
| Core in any setting (research, internal, production, closed-source) | Apache-2.0 (free) |
| Advanced & Labs — evaluation / non-production | BUSL-1.1 (free) |
| Advanced & Labs — production or SaaS | Commercial |

## Third-Party Components

| Component | License | Location |
|-----------|---------|----------|
| NumPy | BSD-3-Clause | runtime dependency |
| SciPy | BSD-3-Clause | runtime dependency |
| Requests | Apache-2.0 | runtime dependency |
| PyTorch | BSD-3-Clause | `[nli]` optional |
| Hugging Face Transformers | Apache-2.0 | `[nli]` optional |
| ChromaDB | Apache-2.0 | `[vector]` optional |
| FastAPI | MIT | `[server]` optional |
| PyO3 | MIT/Apache-2.0 | Rust FFI binding |

See `sbom.json` (attached to each GitHub Release) for the full software
bill of materials.

## SCPN Framework

Director-AI implements Layer 16 (The Director) of the
Self-Consistent Phenomenological Network (SCPN) Framework.

Related repositories:
- [SCPN Fusion Core](https://github.com/anulum/scpn-fusion-core)
- [SC-NeuroCore](https://github.com/anulum/sc-neurocore)
- [Holonomic Atlas](https://github.com/anulum/HolonomicAtlas)
