# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native Rust macro provenance conformance
"""Exercise original external macro generation and native import admission."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.quality_ratchet_rust_tests.test_quality_ratchet_rust import (
    rust_reader as rust_reader,
)
from tools.quality_ratchet_rust_tests.test_quality_ratchet_rust import (
    rust_tools as rust_tools,
)
from tools.quality_ratchet_rust_tests.test_quality_ratchet_rust_native import _reader


@pytest.mark.parametrize(
    ("body", "status"),
    (
        (
            "/// Owned values.\nmod owned { /// Original value.\npub const VALUE: bool = true; }\nuse crate::owned::*; /// Execute.\nfn run() { assert!(VALUE); }",
            0,
        ),
        (
            "/// Owned values.\nmod owned { /// Original value.\npub const VALUE: bool = true; }\nuse self::owned::*; /// Execute.\nfn run() { assert!(VALUE); }",
            0,
        ),
        (
            "/// Parent.\nmod outer { /// Child.\nmod inner { "
            "use super::super::*; /// Local macro use.\nfn read() { assert!(true); } } }",
            0,
        ),
        (
            "/// Child.\nmod inner { use super::super::*; "
            "/// Unresolved parent.\nfn read() { assert!(true); } }",
            2,
        ),
        (
            "/// Out-of-line module.\nmod bridge; use self::bridge::*; "
            "/// Unresolved package alias.\nfn read() { serde_json::json!(); }",
            2,
        ),
        (
            "/// Platform module.\n#[cfg(unix)] mod bridge;\n"
            "/// Inline module.\n#[cfg(windows)] mod bridge { /// Value.\n"
            "pub const VALUE: bool = true; }\n"
            "use self::bridge::*; /// Original caller.\nfn read() { assert!(VALUE); }",
            2,
        ),
        (
            "use ::std::*; /// Absolute standard glob.\nfn read() { assert!(true); }",
            0,
        ),
        (
            "use crate::bridge::*; /// Unknown wildcard.\n"
            "#[derive(Debug)] struct Read;",
            2,
        ),
        (
            "use foreign::*; use std::format; /// Explicit standard macro.\n"
            'fn read() { format!("value"); }',
            0,
        ),
        (
            "use foreign as serde_json; use serde_json::json; /// Execute.\nfn run() { json!({}); }",
            2,
        ),
        (
            'use thiserror::Error; /// Error.\n#[derive(Error, Debug)] #[error("original error")] struct Failure { /// Original source.\n#[from] original: std::io::Error }',
            0,
        ),
        (
            'use thiserror::Error; /// Error.\n#[derive(Error, Debug)] #[error("original error")] struct Failure { /// Original source.\n#[source] original: std::io::Error }',
            0,
        ),
        (
            'use thiserror::Error; /// Error.\n#[derive(Error, Debug)] #[error("original error")] struct Failure { /// Original backtrace.\n#[backtrace] trace: std::backtrace::Backtrace }',
            0,
        ),
        (
            'use serde::Serialize; /// Value.\n#[derive(Serialize)] struct Value { /// Original field.\n#[serde(rename = "field")] field: u8 }',
            0,
        ),
        (
            'use serde::Deserialize; /// Value.\n#[derive(Deserialize)] struct Value { /// Original field.\n#[serde(rename = "field")] field: u8 }',
            0,
        ),
        (
            '/// Value.\nstruct Value { /// Original field.\n#[serde(rename = "field")] field: u8 }',
            2,
        ),
        (
            "use wasm_bindgen_test::wasm_bindgen_test; /// Execute.\n#[wasm_bindgen_test] fn run() {}",
            0,
        ),
        ("use foreign::assert; /// Execute.\nfn run() { assert!(true); }", 2),
        (
            "/// Nested.\nmod nested { use foreign::assert; /// Execute.\nfn run() { assert!(true); } }",
            2,
        ),
        ("use foreign::*; /// Execute.\nfn run() { assert!(true); }", 2),
        (
            "use foreign as serde_json; /// Execute.\nfn run() { serde_json::json!({}); }",
            2,
        ),
        (
            "extern crate foreign as serde_json; /// Execute.\nfn run() { serde_json::json!({}); }",
            2,
        ),
        (
            "/// Local namespace.\nmod serde_json {} /// Execute.\nfn run() { serde_json::json!({}); }",
            2,
        ),
        ("/// Execute.\nfn run() { json!({}); }", 2),
        ("use foreign::pyfunction; /// Execute.\n#[pyfunction] fn run() {}", 2),
        (
            "use pyo3::prelude::*; use foreign::pyfunction; /// Execute.\n#[pyfunction] fn run() {}",
            2,
        ),
        ("/// Execute.\n#[foreign_procedure] fn run() {}", 2),
        ("/// Value.\n#[derive(Foreign)] struct Value;", 2),
        ("/// Value.\n#[derive(foreign::Generated)] struct Value;", 2),
        (
            "#[macro_use] extern crate foreign; /// Execute.\nfn run() { assert!(true); }",
            2,
        ),
        ("/// Value.\n#[derive] struct Value;", 2),
        ("/// Value.\n#[derive(,)] struct Value;", 2),
        ("/// Value.\n#[derive()] struct Value;", 2),
        ("use foreign::Default; /// Value.\n#[derive(Default)] struct Value;", 2),
        ('use std::format; /// Execute.\nfn run() { format!("value"); }', 0),
        ("use serde_json::{json}; /// Execute.\nfn run() { json!({}); }", 0),
        (
            "use pyo3::prelude::*; /// Execute.\nfn run() { wrap_pyfunction!(run, module); }",
            0,
        ),
        ("use serde_json::*; /// Execute.\nfn run() { json!({}); }", 0),
        ("/// Value.\n#[derive(serde::Serialize)] struct Value;", 0),
        (
            "use serde::{Serialize, Deserialize}; /// Value.\n#[derive(Serialize, Deserialize)] struct Value;",
            0,
        ),
        (
            "use syn::Token; /// Nested.\nmod nested { use foreign::Token; /// Comma.\ntype Comma = Token![,]; }",
            2,
        ),
        (
            'use thiserror::Error; /// Value.\n#[derive(Error)] #[error("failure")] struct Value;',
            0,
        ),
        ("use wasm_bindgen::prelude::*; /// Value.\n#[wasm_bindgen] struct Value;", 0),
        ("use wasm_bindgen_test::*; /// Execute.\n#[wasm_bindgen_test] fn run() {}", 0),
        (
            "use pyo3::prelude::*; /// Value.\n#[pyclass] struct Value; /// Methods.\n#[pymethods] impl Value { /// Construct.\n#[new] fn new() -> Self { Self } }",
            0,
        ),
    ),
)
def test_native_macro_and_attribute_imports_are_source_bound(
    rust_reader: Path, tmp_path: Path, body: str, status: int
) -> None:
    """Refuse rebound or unsupported generation recipes through the actual native process."""
    actual, report = _reader(
        rust_reader,
        tmp_path / "lib.rs",
        "//! Original module.\n" + body + "\n",
        crate_root=True,
    )
    assert actual == status
    assert bool(report["refusals"]) == (status == 2)


@pytest.mark.parametrize("kind", ("expression", "attribute"))
def test_compiled_external_generation_requires_source_provenance(
    rust_reader: Path, rust_tools: tuple[str, str], tmp_path: Path, kind: str
) -> None:
    """Compile and execute genuine external generation before requiring native refusal."""
    _, rustc = rust_tools
    header = (
        (Path(__file__).resolve().parents[2] / "tools/quality_ratchet_rust/src/main.rs")
        .read_text()
        .splitlines()
    )
    prefix = "\n".join(header[:7]) + "\n"
    if kind == "expression":
        library_source = (
            "//! External expression macro.\n/// Generate the actual function.\n"
            '#[macro_export] macro_rules! assert { () => { fn generated() { println!("observed"); } }; }\n'
        )
        consumer = (
            "//! Actual consumer.\nuse foreign::assert;\n"
            "/// Execute original expansion.\nfn main() { assert!(); generated(); }\n"
        )
        crate_type, filename = "rlib", "libforeign.rlib"
    else:
        library_source = (
            "//! External procedural attribute.\nextern crate proc_macro;\n"
            "use proc_macro::TokenStream;\n"
            "/// Retain the original item and generate the observed function.\n"
            "#[proc_macro_attribute] pub fn foreign_procedure(_: TokenStream, item: TokenStream) -> TokenStream {\n"
            'let mut result = item; result.extend("fn generated() { println!(\\"observed\\"); }".parse::<TokenStream>().unwrap()); result }\n'
        )
        consumer = (
            "//! Actual consumer.\nuse foreign::foreign_procedure;\n"
            "/// Execute original expansion.\n#[foreign_procedure] fn main() { generated(); }\n"
        )
        crate_type, filename = "proc-macro", "libforeign.so"
    library, source = tmp_path / "library.rs", tmp_path / "consumer.rs"
    library.write_text(prefix + library_source)
    source.write_text(prefix + consumer)
    output = tmp_path / filename
    commands = (
        [
            rustc,
            "--edition",
            "2021",
            "--crate-name",
            "foreign",
            "--crate-type",
            crate_type,
            "-D",
            "warnings",
            str(library),
            "-o",
            str(output),
        ],
        [
            rustc,
            "--edition",
            "2021",
            "-D",
            "warnings",
            str(source),
            "--extern",
            "foreign=" + str(output),
            "-o",
            str(tmp_path / "consumer"),
        ],
        [str(tmp_path / "consumer")],
    )
    for index, command in enumerate(commands):
        original = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
        (tmp_path / f"original-process-{index}.json").write_text(
            json.dumps(
                {
                    "argv": command,
                    "returncode": original.returncode,
                    "stdout": original.stdout,
                    "stderr": original.stderr,
                }
            )
        )
        assert original.returncode == 0 and not original.stderr
    assert original.stdout == "observed\n"
    status, report = _reader(rust_reader, source, prefix + consumer)
    assert status == 2 and report["refusals"]


@pytest.mark.parametrize("crate_root", (False, True))
def test_crate_glob_requires_an_actual_target_root(
    rust_reader: Path, tmp_path: Path, crate_root: bool
) -> None:
    """Use crate-relative provenance only when the caller establishes the crate root."""
    source = (
        "//! Original root.\n"
        "/// Owned values.\nmod owned { /// Original value.\npub const VALUE: bool = true; }\n"
        "use crate::owned::*; /// Read.\nfn read() { assert!(VALUE); }\n"
    )
    status, report = _reader(
        rust_reader, tmp_path / "lib.rs", source, crate_root=crate_root
    )
    assert status == (0 if crate_root else 2)
    assert bool(report["refusals"]) is (not crate_root)


def test_compiled_out_of_line_reexport_requires_native_provenance(
    rust_reader: Path, rust_tools: tuple[str, str], tmp_path: Path
) -> None:
    """Refuse an unresolved wildcard after Rust proves the imported macro is external."""
    _, rustc = rust_tools
    library = tmp_path / "library.rs"
    library.write_text(
        "#[macro_export] macro_rules! fixture_value { () => { 7u8 } }\n"
        "pub mod recipe { pub use crate::fixture_value as json; }\n"
    )
    (tmp_path / "bridge.rs").write_text(
        "pub use provenance_fixture::recipe as serde_json;\n"
    )
    consumer = tmp_path / "consumer.rs"
    body = (
        "//! Consumer source.\nuse super::bridge::*;\n"
        "/// Return the imported result.\npub fn result() -> u8 { serde_json::json!() }\n"
    )
    consumer.write_text(body)
    main = tmp_path / "main.rs"
    main.write_text(
        'mod bridge; mod consumer; fn main() { println!("{}", consumer::result()); }\n'
    )
    dependency = tmp_path / "libprovenance_fixture.rlib"
    executable = tmp_path / "runtime"
    for command in (
        [
            rustc,
            "--edition=2021",
            "-Dwarnings",
            "--crate-name",
            "provenance_fixture",
            "--crate-type=rlib",
            str(library),
            "-o",
            str(dependency),
        ],
        [
            rustc,
            "--edition=2021",
            "-Dwarnings",
            str(main),
            "--extern",
            f"provenance_fixture={dependency}",
            "-o",
            str(executable),
        ],
    ):
        compiled = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert compiled.returncode == 0, compiled.stderr
    runtime = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=20
    )
    assert runtime.returncode == 0 and runtime.stdout == "7\n" and not runtime.stderr
    status, report = _reader(rust_reader, consumer, body)
    assert status == 2
    assert report["refusals"] == [
        "Macro expression import is not qualified: serde_json::json"
    ]
