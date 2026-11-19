// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — native Rust declaration reader
//! Read original Rust source and emit complete native documentation evidence.
/// Native documentation values and quality directive analysis.
mod attributes;
/// Complete authored declaration traversal and evidence.
mod declarations;
/// Qualified macro source and generated registration boundaries.
mod macros;
use declarations::Analysis;
use serde_json::{json, Value};
use std::{collections::BTreeSet, env, fs, process::ExitCode};

/// Read each explicit source once and refuse invalid or duplicated native inputs.
fn run() -> Result<(Value, u8), String> {
    let mut arguments = env::args().skip(1);
    let mut paths = BTreeSet::new();
    let mut reports = Vec::new();
    let mut status = 0;
    while let Some(option) = arguments.next() {
        let (criterion, crate_root) = match option.as_str() {
            "--source" => (false, false),
            "--crate-root" => (false, true),
            "--criterion-source" => (true, true),
            _ => return Err("Native Rust reader expects explicit source options".into()),
        };
        let path = arguments
            .next()
            .ok_or("Native Rust reader source option has no path")?;
        if !paths.insert(path.clone()) {
            return Err("Native Rust reader source identity is duplicated".into());
        }
        let source = fs::read_to_string(&path)
            .map_err(|error| format!("Original Rust source could not be read: {error}"))?;
        if source.starts_with('\u{feff}') {
            return Err("Rust BOM source coordinates require an explicit native recipe".into());
        }
        let syntax = syn::parse_file(&source)
            .map_err(|error| format!("Original Rust syntax could not be parsed: {error}"))?;
        let analysis = Analysis::source(&syntax, criterion, crate_root);
        if !analysis.refusals.is_empty() {
            status = 2;
        } else if !analysis.findings.is_empty() && status == 0 {
            status = 1;
        }
        reports.push(json!({"path":path,"source":source,"findings":analysis.findings,"refusals":analysis.refusals}));
    }
    if reports.is_empty() {
        return Err("Native Rust reader source inventory is empty".into());
    }
    Ok((
        json!({"schema":"quality-ratchet.rust-ast.v1","files":reports}),
        status,
    ))
}

/// Preserve native finding/refusal status without presenting incomplete source as clean.
fn main() -> ExitCode {
    match run() {
        Ok((report, status)) => {
            println!("{report}");
            ExitCode::from(status)
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}
