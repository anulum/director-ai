// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — native Rust macro source boundaries
//! Admit explicit expression recipes and source-bound Criterion registrations.
use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens;
use std::collections::{BTreeMap, BTreeSet};
use syn::{punctuated::Punctuated, Macro, Path, Token};

/// Detect declaration syntax even inside nested macro token groups.
pub fn has_declaration_tokens(tokens: TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        TokenTree::Group(group) => has_declaration_tokens(group.stream()),
        TokenTree::Ident(name) => [
            "fn",
            "struct",
            "enum",
            "mod",
            "type",
            "const",
            "static",
            "trait",
            "impl",
            "union",
            "macro_rules",
            "use",
        ]
        .contains(&name.to_string().as_str()),
        _ => false,
    })
}

/// Require a known expression-producing recipe rather than silently skipping arbitrary tokens.
pub fn expression_recipe(node: &Macro, bindings: &[(String, String)]) -> Result<(), String> {
    let path = node.path.to_token_stream().to_string().replace(' ', "");
    let supported = [
        "assert",
        "assert_eq",
        "assert_ne",
        "debug_assert",
        "debug_assert_eq",
        "debug_assert_ne",
        "vec",
        "format",
        "format_args",
        "panic",
        "println",
        "eprintln",
        "print",
        "eprint",
        "matches",
        "env",
        "option_env",
        "concat",
        "file",
        "line",
        "column",
        "module_path",
        "stringify",
        "include_str",
        "include_bytes",
        "json",
        "serde_json::json",
        "log::error",
        "log::warn",
        "log::info",
        "log::debug",
        "log::trace",
        "wrap_pyfunction",
        "pyo3::wrap_pyfunction",
    ];
    if !supported.contains(&path.as_str()) {
        return Err(format!("Macro expression recipe is not qualified: {path}"));
    }
    let bound = if path == "json" {
        external_symbol(bindings, "json", "serde_json", "serde_json::*")
    } else if path == "wrap_pyfunction" {
        external_symbol(bindings, "wrap_pyfunction", "pyo3", "pyo3::prelude::*")
    } else if let Some((package, _)) = path.split_once("::") {
        namespace(bindings, package)
    } else {
        agrees(
            bindings,
            &path,
            &[
                format!("std::{path}"),
                format!("core::{path}"),
                format!("alloc::{path}"),
            ],
            trusted_globs(bindings),
        )
    };
    if !bound {
        return Err(format!("Macro expression import is not qualified: {path}"));
    }
    if has_declaration_tokens(node.tokens.clone()) {
        return Err("Macro arguments contain declaration syntax requiring native expansion".into());
    }
    Ok(())
}

/// Validate compact Criterion macro arguments against actual top-level functions and groups.
pub fn criterion_registration(
    node: &Macro,
    functions: &BTreeSet<String>,
    groups: &mut BTreeSet<String>,
) -> Result<(), String> {
    let paths = node
        .parse_body_with(Punctuated::<Path, Token![,]>::parse_terminated)
        .map_err(|_| "Criterion registration arguments are not qualified")?;
    let names: Result<Vec<_>, _> = paths
        .iter()
        .map(|path| {
            if path.leading_colon.is_some()
                || path.segments.len() != 1
                || !matches!(path.segments[0].arguments, syn::PathArguments::None)
            {
                Err("Criterion registration requires actual local identifier targets")
            } else {
                Ok(path.segments[0].ident.to_string())
            }
        })
        .collect();
    let names = names?;
    if node.path.is_ident("criterion_group") && names.len() >= 2 {
        if names.iter().skip(1).any(|name| !functions.contains(name)) {
            return Err("Criterion target is not an actual authored top-level function".into());
        }
        if !groups.insert(names[0].clone()) {
            return Err("Criterion group identity is duplicated".into());
        }
        Ok(())
    } else if node.path.is_ident("criterion_main")
        && !names.is_empty()
        && names.iter().all(|name| groups.contains(name))
    {
        Ok(())
    } else {
        Err("Criterion registration does not match its actual group/function source".into())
    }
}

/// Preserve each explicit imported symbol and its actual source package path.
fn imports(tree: &syn::UseTree, prefix: &str, result: &mut Vec<(String, String)>) {
    match tree {
        syn::UseTree::Path(node) => {
            let prefix = format!("{prefix}{}::", node.ident);
            imports(&node.tree, &prefix, result);
        }
        syn::UseTree::Name(node) => {
            result.push((node.ident.to_string(), format!("{prefix}{}", node.ident)))
        }
        syn::UseTree::Rename(node) => {
            result.push((node.rename.to_string(), format!("{prefix}{}", node.ident)))
        }
        syn::UseTree::Group(node) => {
            for item in &node.items {
                imports(item, prefix, result);
            }
        }
        syn::UseTree::Glob(_) => result.push(("*".into(), format!("{prefix}*"))),
    }
}

/// Require an actual unambiguous explicit native macro import from its owning package.
pub fn imported(file: &syn::File, symbol: &str, package: &str) -> bool {
    let mut bindings = Vec::new();
    for item in &file.items {
        if let syn::Item::Use(node) = item {
            imports(&node.tree, "", &mut bindings);
        }
    }
    let expected = format!("{package}::{symbol}");
    let selected: Vec<_> = bindings.iter().filter(|(name, _)| name == symbol).collect();
    !selected.is_empty() && selected.iter().all(|(_, path)| path == &expected)
}

/// Admit only the proven Syn comma-token type recipe in its actual type-macro context.
pub fn comma_type(node: &syn::TypeMacro, imported: bool) -> bool {
    imported && node.mac.path.is_ident("Token") && node.mac.tokens.to_string() == ","
}

/// Check whether a relative glob resolves to an inline module in this source.
fn local_glob(
    path: &str,
    scope: &[String],
    modules: &BTreeMap<Vec<String>, bool>,
    crate_root: bool,
) -> bool {
    let mut segments = path.trim_end_matches("::*").split("::").peekable();
    let mut owner = scope.to_vec();
    match segments.next() {
        Some("self") => {}
        Some("crate") if crate_root => owner.clear(),
        Some("super") => {
            if owner.pop().is_none() {
                return false;
            }
            while segments.next_if_eq(&"super").is_some() {
                if owner.pop().is_none() {
                    return false;
                }
            }
        }
        _ => return false,
    }
    owner.extend(segments.map(str::to_owned));
    (1..=owner.len()).all(|depth| modules.get(&owner[..depth]) == Some(&true))
}

/// Collect imports and distinguish proven inline module globs from unresolved ones.
pub fn source_bindings(file: &syn::File, crate_root: bool) -> Vec<(String, String)> {
    /// Native source bindings and the inline module paths this file owns.
    #[derive(Default)]
    struct Collector {
        /// Explicit bindings and wildcards whose original scope is unresolved.
        bindings: Vec<(String, String)>,
        /// Relative and external glob candidates and the modules that declare them.
        globs: Vec<(String, Vec<String>)>,
        /// Path of the current inline module from the source's top level.
        scope: Vec<String>,
        /// Whether every declaration of a module supplies its body in this file.
        modules: BTreeMap<Vec<String>, bool>,
    }

    impl<'ast> syn::visit::Visit<'ast> for Collector {
        /// Preserve original imports in the module that introduces them.
        fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
            let mut items = Vec::new();
            imports(&node.tree, "", &mut items);
            for (name, path) in items {
                if name == "*" && node.leading_colon.is_none() {
                    self.globs.push((path, self.scope.clone()));
                } else {
                    self.bindings.push((name, path));
                }
            }
        }

        /// Preserve explicit crate aliases which can shadow a qualified package root.
        fn visit_item_extern_crate(&mut self, node: &'ast syn::ItemExternCrate) {
            let name = node.rename.as_ref().map_or(&node.ident, |(_, alias)| alias);
            self.bindings
                .push((name.to_string(), node.ident.to_string()));
        }

        /// Record only inline children as modules whose contents are available here.
        fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
            self.bindings
                .push((node.ident.to_string(), "#authored-module".into()));
            self.scope.push(node.ident.to_string());
            let inline = node.content.is_some();
            self.modules
                .entry(self.scope.clone())
                .and_modify(|owned| *owned &= inline)
                .or_insert(inline);
            syn::visit::visit_item_mod(self, node);
            self.scope.pop();
        }
    }

    let mut collector = Collector::default();
    syn::visit::Visit::visit_file(&mut collector, file);
    for (path, scope) in collector.globs {
        if !local_glob(&path, &scope, &collector.modules, crate_root) {
            collector.bindings.push(("*".into(), path));
        }
    }
    collector.bindings
}

/// Check that every remaining wildcard names a supported compiler or package prelude.
fn trusted_globs(bindings: &[(String, String)]) -> bool {
    bindings
        .iter()
        .filter(|(name, _)| name == "*")
        .all(|(_, path)| {
            [
                "std::*",
                "core::*",
                "alloc::*",
                "pyo3::prelude::*",
                "serde_json::*",
                "serde::*",
                "thiserror::*",
                "wasm_bindgen::prelude::*",
                "wasm_bindgen_test::*",
            ]
            .contains(&path.as_str())
        })
}

/// Refuse explicit rebinding even when another scope supplies the expected import.
pub fn agrees(
    bindings: &[(String, String)],
    name: &str,
    expected: &[String],
    fallback: bool,
) -> bool {
    let selected: Vec<_> = bindings.iter().filter(|(alias, _)| alias == name).collect();
    if selected.is_empty() {
        fallback
    } else {
        selected.iter().all(|(_, path)| expected.contains(path))
    }
}

/// Require a qualified package root to retain its actual external identity.
fn namespace(bindings: &[(String, String)], package: &str) -> bool {
    let mut expected = vec![package.to_string()];
    if ["wasm_bindgen", "wasm_bindgen_test"].contains(&package) {
        expected.push(format!("{package}::{package}"));
    }
    agrees(bindings, package, &expected, trusted_globs(bindings))
}

/// Bind an external macro or procedural attribute to its explicit import or supported prelude.
fn external_symbol(bindings: &[(String, String)], symbol: &str, package: &str, glob: &str) -> bool {
    namespace(bindings, package)
        && agrees(
            bindings,
            symbol,
            &[format!("{package}::{symbol}")],
            bindings
                .iter()
                .any(|(name, path)| name == "*" && path == glob),
        )
}

/// Admit only known non-generating attributes and source-bound framework generation recipes.
pub fn attribute_recipe(meta: &syn::Meta, bindings: &[(String, String)]) -> bool {
    let name = meta.path().to_token_stream().to_string().replace(' ', "");
    if name == "derive" {
        let syn::Meta::List(list) = meta else {
            return false;
        };
        let Ok(paths) = list.parse_args_with(Punctuated::<Path, Token![,]>::parse_terminated)
        else {
            return false;
        };
        return !paths.is_empty()
            && paths.iter().all(|path| {
                let name = path.to_token_stream().to_string().replace(' ', "");
                if [
                    "Clone",
                    "Copy",
                    "Debug",
                    "Default",
                    "Eq",
                    "PartialEq",
                    "Ord",
                    "PartialOrd",
                    "Hash",
                ]
                .contains(&name.as_str())
                {
                    agrees(
                        bindings,
                        &name,
                        &[format!("std::{name}"), format!("core::{name}")],
                        trusted_globs(bindings),
                    )
                } else if ["Serialize", "Deserialize"].contains(&name.as_str()) {
                    external_symbol(bindings, &name, "serde", "serde::*")
                } else if name == "Error" {
                    external_symbol(bindings, &name, "thiserror", "thiserror::*")
                } else {
                    ["serde::Serialize", "serde::Deserialize", "thiserror::Error"]
                        .contains(&name.as_str())
                        && namespace(
                            bindings,
                            name.split_once("::")
                                .expect("Qualified derive has a package")
                                .0,
                        )
                }
            });
    }
    if ["pyfunction", "pyclass", "pymodule", "pymethods"].contains(&name.as_str()) {
        return external_symbol(bindings, &name, "pyo3", "pyo3::prelude::*");
    }
    if [
        "new",
        "getter",
        "setter",
        "staticmethod",
        "classmethod",
        "classattr",
        "pyo3",
    ]
    .contains(&name.as_str())
    {
        return external_symbol(bindings, "pymethods", "pyo3", "pyo3::prelude::*");
    }
    if name == "wasm_bindgen" {
        return external_symbol(bindings, &name, "wasm_bindgen", "wasm_bindgen::prelude::*");
    }
    if name == "wasm_bindgen_test" {
        return external_symbol(bindings, &name, "wasm_bindgen_test", "wasm_bindgen_test::*");
    }
    if name == "error" || name == "from" || name == "source" || name == "backtrace" {
        return external_symbol(bindings, "Error", "thiserror", "thiserror::*");
    }
    if name == "serde" {
        return external_symbol(bindings, "Serialize", "serde", "serde::*")
            || external_symbol(bindings, "Deserialize", "serde", "serde::*");
    }
    [
        "cfg",
        "test",
        "deny",
        "warn",
        "forbid",
        "inline",
        "repr",
        "cold",
        "must_use",
        "no_mangle",
        "export_name",
        "link",
        "link_name",
        "link_section",
        "target_feature",
        "deprecated",
        "non_exhaustive",
        "track_caller",
        "crate_name",
        "crate_type",
        "path",
        "macro_export",
        "global_allocator",
        "alloc_error_handler",
        "automatically_derived",
        "used",
        "should_panic",
        "instruction_set",
        "naked",
    ]
    .contains(&name.as_str())
}
