<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-Class AI — polyglot generated API reference landing page -->

# Polyglot API Reference

The documentation pipeline generates language-native API references from the
maintained source surfaces. Generated HTML is an output artefact; it is not
committed to the repository.

| Surface | Generator | Published reference |
| --- | --- | --- |
| Rust workspace | `cargo doc` / rustdoc | <a href="../../reference/rust/backfire_kernel/index.html">Rust API</a> |
| Go gateway | `go doc -all` | <a href="../../reference/go/index.html">Go API</a> |
| TypeScript middleware | TypeDoc | <a href="../../reference/typescript/index.html">TypeScript API</a> |
| Julia threshold tuner | Documenter.jl | <a href="../../reference/julia/index.html">Julia API</a> |
| Lean halt monitor | doc-gen4 | <a href="../../reference/lean/index.html">Lean API</a> |
| Protobuf contracts | protoc-gen-doc | <a href="../../reference/protobuf/index.html">Protobuf API</a> |

Run `make docs-all` to build the MkDocs site and every generated reference.
Individual `docs-<language>` targets provide focused local parity with CI.

The `studio-web` tree is not in this gate. It remains a pre-ratification
consumer of shared SCPN-STUDIO packages and does not yet have a buildable,
owned public API surface. Add it only after those dependencies and its
publication contract land.
