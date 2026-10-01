# Bulletproofs dependency provenance

Source: [Bulletproofs 5.0.0](https://crates.io/crates/bulletproofs/5.0.0), published by zkcrypto.
Archive: `https://static.crates.io/crates/bulletproofs/bulletproofs-5.0.0.crate`.
SHA-256: `012e2e5f88332083bd4235d445ae78081c00b2558443821a9ca5adfe1070073d`.
Licence: MIT; the original notice is retained in `LICENSE.txt`.

This copy replaces `clear_on_drop::Clear` with `zeroize::Zeroize` in the existing scalar cleanup calls. The Drop implementations remain unconditional. Proof arithmetic, transcript labels, encodings and public APIs retain their upstream implementations. The migration follows the direction of [RustSec RUSTSEC-2026-0283](https://rustsec.org/advisories/RUSTSEC-2026-0283.html) and [upstream proposal #17](https://github.com/zkcrypto/bulletproofs/pull/17); it does not adopt the proposal's optional cleanup feature.

Upstream CI workflows, development-only project files and binary illustration assets are omitted. Source, tests, benchmarks, textual documentation and the published Cargo manifest are retained.

The experimental R1CS decoder converts curve25519-dalek 4.x `CtOption<Scalar>` values to `Option<Scalar>` before returning the existing format error. A public proof serialization test covers the valid round trip and rejection of each noncanonical scalar field.

Repository formatting hooks normalize text whitespace and correct spelling in upstream comments and documentation; these changes do not alter proof code.
