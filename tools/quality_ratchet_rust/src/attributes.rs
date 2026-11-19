// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — native Rust documentation attributes
//! Analyse native documentation values and nested quality directives.
use proc_macro2::Span;
use syn::{punctuated::Punctuated, Attribute, Expr, Lit, Meta, Token};

/// Recognise unconditional, explanatory Rustdoc string attributes.
pub fn documented(attributes: &[Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        if let Meta::NameValue(value) = &attribute.meta {
            if value.path.is_ident("doc") {
                if let Expr::Lit(value) = &value.value {
                    if let Lit::Str(text) = &value.lit {
                        return !text.value().trim().is_empty();
                    }
                }
            }
        }
        false
    })
}

/// Preserve actual masking locations and refuse documentation without a qualified literal.
pub fn inspect(
    attribute: &Attribute,
    bindings: &[(String, String)],
) -> (Vec<(Span, String)>, Vec<String>) {
    let mut findings = Vec::new();
    let mut refusals = Vec::new();
    inspect_meta(&attribute.meta, bindings, &mut findings, &mut refusals);
    (findings, refusals)
}

/// Descend through cfg_attr without treating conditional text as unconditional documentation.
fn inspect_meta(
    meta: &Meta,
    bindings: &[(String, String)],
    findings: &mut Vec<(Span, String)>,
    refusals: &mut Vec<String>,
) {
    use syn::spanned::Spanned;
    let path = meta.path();
    if path.is_ident("allow") || path.is_ident("expect") {
        findings.push((
            path.span(),
            "Lint relaxation or expectation directive is not allowed".into(),
        ));
    } else if path.is_ident("cfg_attr") {
        if let Meta::List(list) = meta {
            match list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                Ok(items) if items.len() >= 2 => {
                    for item in items.iter().skip(1) {
                        inspect_meta(item, bindings, findings, refusals);
                    }
                }
                _ => refusals.push("Conditional attribute grammar is not qualified".into()),
            }
        } else {
            refusals.push("Conditional attribute grammar is not qualified".into());
        }
    } else if path.is_ident("doc") {
        match meta {
            Meta::NameValue(value) => {
                if !matches!(&value.value, Expr::Lit(value) if matches!(&value.lit, Lit::Str(_))) {
                    refusals.push(
                        "Computed Rustdoc text requires a qualified original-input recipe".into(),
                    );
                }
            }
            Meta::List(list) => {
                match list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                    Ok(items) => {
                        for item in items {
                            if item.path().is_ident("hidden") {
                                findings.push((
                                    item.path().span(),
                                    "Rustdoc visibility masking directive is not allowed".into(),
                                ));
                            } else if ![
                                "alias",
                                "cfg",
                                "inline",
                                "no_inline",
                                "test",
                                "html_root_url",
                                "html_logo_url",
                                "html_favicon_url",
                                "html_playground_url",
                                "issue_tracker_base_url",
                            ]
                            .iter()
                            .any(|name| item.path().is_ident(name))
                            {
                                refusals.push("Rustdoc metadata has an unqualified member".into());
                            }
                        }
                    }
                    Err(_) => refusals.push("Rustdoc metadata grammar is not qualified".into()),
                }
            }
            Meta::Path(_) => refusals.push("Bare Rustdoc attribute is not qualified".into()),
        }
    } else if !crate::macros::attribute_recipe(meta, bindings) {
        refusals.push("Attribute or derive requires a qualified native package recipe".into());
    }
}
