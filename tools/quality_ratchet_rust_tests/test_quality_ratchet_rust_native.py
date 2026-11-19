# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native Rust reader conformance contracts
"""Exercise complete authored Rust syntax through the production process entry."""

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


def _reader(
    binary: Path,
    path: Path,
    source: str,
    *,
    criterion: bool = False,
    crate_root: bool = False,
) -> tuple[int, dict[str, object]]:
    """Run actual unchanged source through the reader's public process entry."""
    path.write_text(source)
    option = "--source"
    if criterion:
        option = "--criterion-source"
    elif crate_root:
        option = "--crate-root"
    result = subprocess.run(
        [
            str(binary),
            option,
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert not result.stderr
    report = json.loads(result.stdout)
    return result.returncode, report["files"][0]


def test_private_cfg_nested_members_and_unicode_positions(
    rust_reader: Path, tmp_path: Path
) -> None:
    """Retain every authored hidden owner and native Unicode location."""
    source = (
        "//! Complete authored source scope.\n"
        "#[cfg(any())]\n"
        "mod disabled {\n"
        "    /* π */ pub struct Δ { pub café: u8, private: u8 }\n"
        "    fn private_function() { fn nested_function() {} }\n"
        "}\n"
        "trait Contract { type Result; const VALUE: u8; fn read(&self) -> u8; }\n"
        "struct Stored(u8);\n"
        "impl Contract for Stored {\n"
        "type Result = u8; const VALUE: u8 = 1; fn read(&self) -> u8 { self.0 }\n"
        "}\n"
        'extern "C" { fn external_contract(); }\n'
        "enum Decision { Accepted { value: u8 }, Refused(u8) }\n"
        "union Bits { integer: u32, floating: f32 }\n"
        "const LIMIT: u8 = 1; static CACHE: u8 = 1; type Score = u8;\n"
    )
    status, report = _reader(rust_reader, tmp_path / "source.rs", source)
    assert status == 1 and report["refusals"] == []
    findings = report["findings"]
    assert isinstance(findings, list)
    messages = {finding["message"] for finding in findings}
    for name in (
        "disabled",
        "Δ",
        "café",
        "private",
        "private_function",
        "nested_function",
        "Contract",
        "Result",
        "VALUE",
        "read",
        "external_contract",
        "Decision",
        "Accepted",
        "Refused",
        "Bits",
        "integer",
        "floating",
        "LIMIT",
        "CACHE",
        "Score",
    ):
        assert any(f" {name} requires" in message for message in messages), name
    assert any("positional field" in message for message in messages)
    position = next(f for f in findings if f["message"].startswith("Struct Δ "))
    assert (
        position["line"] == 4
        and position["column"] == source.splitlines()[3].index("Δ") + 1
    )


@pytest.mark.parametrize(
    "attribute",
    (
        "#![allow(missing_docs)]",
        "#![expect(missing_docs)]",
        "#![cfg_attr(any(), allow(missing_docs))]",
        "#![doc(hidden)]",
        "#![cfg_attr(any(), doc(hidden))]",
    ),
)
def test_quality_masks_keep_original_native_locations(
    rust_reader: Path, tmp_path: Path, attribute: str
) -> None:
    """Expose unconditional and cfg-disabled original documentation/lint masks."""
    status, report = _reader(
        rust_reader,
        tmp_path / "source.rs",
        f"//! Original module description.\n{attribute}\n/// Actual value.\nconst VALUE: u8 = 1;\n",
    )
    assert status == 1 and report["refusals"] == []
    findings = report["findings"]
    assert isinstance(findings, list)
    assert any(f["rule"] == "source-directive" and f["line"] == 2 for f in findings)


@pytest.mark.parametrize(
    "source",
    (
        '//! Module.\n#[doc = include_str!("description.md")] pub fn value() {}\n',
        "//! Module.\nmacro_rules! declare { () => { fn hidden() {} } } declare!();\n",
        "//! Module.\n/// Compute a value.\nfn value() { assert!({ fn hidden() {} true }); }\n",
        "//! Module.\n/// Compute a value.\nfn value() { unknown_expression!(); }\n",
        "//! Module.\n/// Public value type.\ntype Value = Unknown!();\n",
        "//! Module.\nuse syn::Token;\n/// Public value type.\ntype Value = Unknown![,];\n",
        "//! Module.\nuse syn::Token;\n/// Public value type.\ntype Value = Token![;];\n",
    ),
)
def test_unqualified_computed_docs_and_macro_declarations_refuse(
    rust_reader: Path, tmp_path: Path, source: str
) -> None:
    """Return incomplete-source refusal through actual native syntax, never clean."""
    status, report = _reader(rust_reader, tmp_path / "source.rs", source)
    assert status == 2 and report["refusals"]


def test_criterion_registrations_bind_real_imports_functions_and_groups(
    rust_reader: Path, tmp_path: Path
) -> None:
    """Admit the exact native Criterion recipe and refuse missing or rebound targets."""
    original = (
        "//! Actual authored benchmark.\n"
        "use criterion::{criterion_group, criterion_main};\n"
        "/// Evaluate the actual benchmark.\nfn benchmark() {}\n"
        "criterion_group!(benches, benchmark);\ncriterion_main!(benches);\n"
    )
    status, report = _reader(
        rust_reader, tmp_path / "bench.rs", original, criterion=True
    )
    assert status == 0 and report["findings"] == [] and report["refusals"] == []
    for changed in (
        original.replace("criterion::{", "foreign::{"),
        original.replace("benches, benchmark", "benches, absent"),
        original.replace("criterion_main!(benches)", "criterion_main!(absent)"),
        original.replace("criterion_main!(benches)", "criterion_main!()"),
        original.replace("benches, benchmark", "benches"),
        original + "unknown_registration!();\n",
        original.replace("criterion_group, criterion_main", "criterion_group"),
        original + "mod nested { criterion_group!(nested_benches, benchmark); }\n",
        original + "criterion_group!(benches, benchmark);\n",
        original.replace("benches, benchmark", "benches, owner::benchmark"),
        original.replace("benches, benchmark", "benches, ::benchmark"),
        original.replace("benches, benchmark", "benches, benchmark::<u8>"),
        original.replace("benches, benchmark", "benches, ?"),
        original.replace(
            "criterion_group, criterion_main",
            "criterion_group as renamed, criterion_main",
        ),
        original.replace("criterion_group, criterion_main", "*"),
        original.replace(
            "criterion_group!(benches, benchmark);",
            "#[allow(missing_docs)] criterion_group!(benches, benchmark);",
        ),
    ):
        status, report = _reader(
            rust_reader, tmp_path / "bench.rs", changed, criterion=True
        )
        if "#[allow(missing_docs)]" in changed:
            assert status == 1 and report["findings"] and report["refusals"] == []
        else:
            assert status == 2 and report["refusals"]


@pytest.mark.parametrize(
    "mode", ("empty", "option", "path", "duplicate", "absent", "utf8", "bom", "syntax")
)
def test_native_cli_refuses_incomplete_original_inputs(
    rust_reader: Path, tmp_path: Path, mode: str
) -> None:
    """Refuse real CLI, file and syntax errors without emitting a clean carrier."""
    path = tmp_path / "source.rs"
    path.write_text("//! Original source.\n")
    arguments = ["--source", str(path)]
    if mode == "empty":
        arguments = []
    elif mode == "option":
        arguments = ["--unknown", str(path)]
    elif mode == "path":
        arguments = ["--source"]
    elif mode == "duplicate":
        arguments *= 2
    elif mode == "absent":
        path.unlink()
    elif mode == "utf8":
        path.write_bytes(b"\xff")
    elif mode == "bom":
        path.write_text("\ufeff//! Original source.\n")
    else:
        path.write_text("fn incomplete(\n")
    result = subprocess.run(
        [str(rust_reader), *arguments],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2 and not result.stdout and result.stderr.strip()


@pytest.mark.parametrize(
    "attribute",
    (
        '#![doc = "Module description."]',
        '#![crate_name = "original"]',
        '#![doc = ""]',
        "#![doc = 1]",
        '#![doc(alias = "Original")]',
        "#![doc]",
        "#![doc(unknown_metadata)]",
        "#![doc(alias =)]",
        "#![cfg_attr]",
        "#![cfg_attr(any())]",
        "#![cfg_attr(any(),)]",
        '#![cfg_attr(any(), doc = "Conditional module description.")]',
    ),
)
def test_native_documentation_metadata_retains_qualification_boundaries(
    rust_reader: Path, tmp_path: Path, attribute: str
) -> None:
    """Accept literal original prose while refusing opaque metadata and conditional docs."""
    status, report = _reader(rust_reader, tmp_path / "source.rs", attribute + "\n")
    if attribute == '#![doc = "Module description."]':
        assert status == 0 and report["findings"] == [] and report["refusals"] == []
    elif (
        'alias = "Original"' in attribute
        or "Conditional module" in attribute
        or attribute == '#![doc = ""]'
        or attribute == '#![crate_name = "original"]'
    ):
        assert status == 1 and report["refusals"] == []
        assert report["findings"]
    else:
        assert status == 2 and report["refusals"]


@pytest.mark.parametrize("imported", (True, False))
def test_native_type_macro_requires_actual_syn_import(
    rust_reader: Path, tmp_path: Path, imported: bool
) -> None:
    """Bind the comma token type recipe to its actual Syn package import."""
    source = "//! Original type declarations.\n"
    if imported:
        source += "use syn::{Token};\n"
    source += "/// Original comma token.\ntype Comma = Token![,];\n"
    status, report = _reader(rust_reader, tmp_path / "source.rs", source)
    assert status == (0 if imported else 2)
    assert bool(report["refusals"]) is not imported


def test_native_trait_alias_and_foreign_members_require_documentation(
    rust_reader: Path, tmp_path: Path
) -> None:
    """Traverse actual trait aliases and all foreign ABI member declarations."""
    source = (
        "//! Native declaration syntax.\n"
        "trait Alias = Send;\n"
        'extern "C" { static VALUE: u8; type External; fn read(); }\n'
    )
    status, report = _reader(rust_reader, tmp_path / "source.rs", source)
    assert status == 1 and report["refusals"] == []
    findings = report["findings"]
    assert isinstance(findings, list)
    for name in ("Alias", "VALUE", "External", "read"):
        assert any(f" {name} requires" in finding["message"] for finding in findings)


@pytest.mark.parametrize(
    "source",
    (
        "fn without_body();",
        "trait Contract { pub fn foreign_visibility(); }",
        "impl Stored { fn without_body(); }",
        'extern "C" { fn with_body() {} }',
        "trait Contract { opaque!(); }",
        "impl Stored { opaque!(); }",
        'extern "C" { opaque!(); }',
    ),
)
def test_native_opaque_declaration_contexts_refuse(
    rust_reader: Path, tmp_path: Path, source: str
) -> None:
    """Refuse actual unparsed and macro-owned item, trait, implementation and ABI syntax."""
    status, report = _reader(
        rust_reader,
        tmp_path / "source.rs",
        "//! Original declaration scope.\n" + source,
    )
    assert status == 2 and report["refusals"]


def test_native_multi_source_preserves_refusal_status(
    rust_reader: Path, tmp_path: Path
) -> None:
    """Retain incomplete status when later files contain ordinary documentation debt."""
    paths = [tmp_path / "opaque.rs", tmp_path / "undocumented.rs"]
    paths[0].write_text("//! Original opaque source.\nopaque!();\n")
    paths[1].write_text("//! Original authored source.\nfn value() {}\n")
    result = subprocess.run(
        [str(rust_reader), "--source", str(paths[0]), "--source", str(paths[1])],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2 and not result.stderr
    reports = json.loads(result.stdout)["files"]
    assert reports[0]["refusals"] and reports[1]["findings"]
