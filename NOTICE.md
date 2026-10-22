<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- DIRECTOR-AI — NOTICE.md -->

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

### Advanced & Labs — LicenseRef-BUSL-1.1-ANULUM-TDM

The advanced capabilities (under `core/<advanced>/`, `enterprise/`, `voice/`,
`ui/`, `experimental/`, `compliance/`, `agentic/`) are source-available under
[LicenseRef-BUSL-1.1-ANULUM-TDM](LICENSES/LicenseRef-BUSL-1.1-ANULUM-TDM.txt). Non-production use is subject
to the complete modified terms. The permanent Use Limitation requires the
Licensor's prior written permission for text and data mining and machine-learning
uses, including model evaluation. It does not restrict running the software for
its documented purpose and survives the Change Date unless withdrawn in writing.
Original Change Date and Change License parameters remain unchanged pending the
Licensor's later relicensing decision. A commercial license is available for organisations that:

- Run the advanced tier in production or as a hosted/SaaS service
- Need SLA-backed support, indemnification, or custom builds

Contact [protoscience@anulum.li](mailto:protoscience@anulum.li) for terms.

## License Boundary

| Use Case | License |
|----------|---------|
| Core in any setting (research, internal, production, closed-source) | Apache-2.0 (free) |
| Advanced & Labs — evaluation / non-production | LicenseRef-BUSL-1.1-ANULUM-TDM; subject to the permanent Use Limitation above |
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

## Text and data mining reservation

The rightsholder, Miroslav Šotek (ANULUM), expressly reserves the use of the
works in this repository for text and data mining, including the collection of
this content for, and its use in, training, fine-tuning or evaluating machine
learning and artificial intelligence models, in accordance with Article 4(3) of
Directive (EU) 2019/790 and the national provisions transposing it.

No permission for such use has been granted beyond what the licence in
`LICENSE` grants. This reservation does not change that licence: uses the
licence permits remain permitted on its terms, including its attribution and
share-alike conditions.

Reservation made on 2026-10-19. Requests for permission: contact the
rightsholder through the contact listed in this repository.
