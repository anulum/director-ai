// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — native Rust authored declarations
//! Bind every authored declaration and attribute to native source positions.
use crate::{attributes, macros};
use proc_macro2::Span;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use syn::{
    spanned::Spanned,
    visit::{self, Visit},
    Attribute, File, Item,
};

/// Original native findings and visible analysis refusals.
#[derive(Default)]
pub struct Analysis {
    /// Original declaration/directive findings at native character positions.
    pub findings: Vec<Value>,
    /// Unsupported source forms preventing complete qualification.
    pub refusals: Vec<String>,
    /// Whether the adapter qualified the actual Criterion benchmark source.
    criterion: bool,
    /// Actual explicit Syn token import for the qualified punctuation type.
    syn_token: bool,
    /// Actual authored top-level functions for generated benchmark registrations.
    functions: BTreeSet<String>,
    /// Actual previously registered groups in this same source.
    groups: BTreeSet<String>,
    /// Current module nesting, separating root benchmark registrations.
    nesting: usize,
    /// Actual explicit/glob imports and package shadows throughout the original source.
    bindings: Vec<(String, String)>,
}

impl Analysis {
    /// Traverse complete original syntax without filtering cfg, tests or private owners.
    pub fn source(file: &File, criterion: bool, crate_root: bool) -> Self {
        let bindings = macros::source_bindings(file, crate_root);
        let syn_token = macros::imported(file, "Token", "syn")
            && macros::agrees(&bindings, "Token", &["syn::Token".into()], false);
        let mut result = Self {
            criterion,
            bindings,
            syn_token,
            functions: file
                .items
                .iter()
                .filter_map(|item| {
                    if let Item::Fn(node) = item {
                        Some(node.sig.ident.to_string())
                    } else {
                        None
                    }
                })
                .collect(),
            ..Self::default()
        };
        if criterion
            && !(macros::imported(file, "criterion_group", "criterion")
                && macros::imported(file, "criterion_main", "criterion"))
        {
            result
                .refusals
                .push("Criterion macro imports differ from the qualified source package".into());
        }
        if !attributes::documented(&file.attrs) {
            result.findings.push(json!({"rule":"module-documentation","line":1,"column":1,"message":"Source file requires a Rustdoc module comment"}));
        }
        result.visit_file(file);
        result
    }

    /// Retain the actual native position of one declaration or directive finding.
    fn finding(&mut self, span: Span, rule: &str, message: String) {
        let position = span.start();
        self.findings.push(
            json!({"rule":rule,"line":position.line,"column":position.column+1,"message":message}),
        );
    }

    /// Require explanatory documentation on each authored named or positional member.
    fn declaration(&mut self, kind: &str, name: String, attributes: &[Attribute], span: Span) {
        if !attributes::documented(attributes) {
            self.finding(
                span,
                "declaration-documentation",
                format!("{kind} {name} requires a Rustdoc comment"),
            );
        }
    }
}

impl<'ast> Visit<'ast> for Analysis {
    /// Refuse opaque item syntax before descending into every supported original node.
    fn visit_item(&mut self, node: &'ast Item) {
        if matches!(node, Item::Verbatim(_)) {
            self.refusals
                .push("Native Rust item contains unparsed syntax".into());
        }
        visit::visit_item(self, node);
    }

    /// Refuse opaque trait member syntax while retaining its actual native descendants.
    fn visit_trait_item(&mut self, node: &'ast syn::TraitItem) {
        if matches!(node, syn::TraitItem::Verbatim(_)) {
            self.refusals
                .push("Native Rust trait member contains unparsed syntax".into());
        }
        visit::visit_trait_item(self, node);
    }

    /// Refuse opaque implementation syntax while retaining actual authored members.
    fn visit_impl_item(&mut self, node: &'ast syn::ImplItem) {
        if matches!(node, syn::ImplItem::Verbatim(_)) {
            self.refusals
                .push("Native Rust implementation member contains unparsed syntax".into());
        }
        visit::visit_impl_item(self, node);
    }

    /// Refuse opaque external ABI syntax while retaining actual foreign declarations.
    fn visit_foreign_item(&mut self, node: &'ast syn::ForeignItem) {
        if matches!(node, syn::ForeignItem::Verbatim(_)) {
            self.refusals
                .push("Native Rust foreign member contains unparsed syntax".into());
        }
        visit::visit_foreign_item(self, node);
    }

    /// Inspect native masking and computed documentation even on nested expressions.
    fn visit_attribute(&mut self, node: &'ast Attribute) {
        let (findings, refusals) = attributes::inspect(node, &self.bindings);
        for (span, message) in findings {
            self.finding(span, "source-directive", message);
        }
        self.refusals.extend(refusals);
        visit::visit_attribute(self, node);
    }

    /// Admit only source-bound generated Criterion registrations at the root benchmark owner.
    fn visit_item_macro(&mut self, node: &'ast syn::ItemMacro) {
        for attribute in &node.attrs {
            self.visit_attribute(attribute);
        }
        if self.criterion && self.nesting == 0 {
            if let Err(error) =
                macros::criterion_registration(&node.mac, &self.functions, &mut self.groups)
            {
                self.refusals.push(error);
            }
        } else {
            self.refusals
                .push("Item macro needs a qualified native generated-source recipe".into());
        }
    }

    /// Refuse declaration-producing trait macros independently of target visibility.
    fn visit_trait_item_macro(&mut self, _node: &'ast syn::TraitItemMacro) {
        self.refusals
            .push("Trait macro needs a qualified native generated-source recipe".into());
    }

    /// Refuse declaration-producing implementation macros independently of cfg.
    fn visit_impl_item_macro(&mut self, _node: &'ast syn::ImplItemMacro) {
        self.refusals
            .push("Implementation macro needs a qualified native generated-source recipe".into());
    }

    /// Refuse declaration-producing ABI macros independently of the target platform.
    fn visit_foreign_item_macro(&mut self, _node: &'ast syn::ForeignItemMacro) {
        self.refusals
            .push("Foreign macro needs a qualified native generated-source recipe".into());
    }

    /// Preserve the distinct type-macro context of the source-bound Syn token recipe.
    fn visit_type_macro(&mut self, node: &'ast syn::TypeMacro) {
        if !macros::comma_type(node, self.syn_token) {
            self.refusals
                .push("Rust type macro requires a qualified native source recipe".into());
        }
    }

    /// Require known expression-producing tokens and expose hidden declaration syntax.
    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        if let Err(error) = macros::expression_recipe(node, &self.bindings) {
            self.refusals.push(error);
        }
    }

    /// Document every actual function before traversing its source children.
    fn visit_item_fn(&mut self, n: &'ast syn::ItemFn) {
        self.declaration(
            "Function",
            n.sig.ident.to_string(),
            &n.attrs,
            n.sig.ident.span(),
        );
        visit::visit_item_fn(self, n);
    }

    /// Document every actual struct before traversing its source children.
    fn visit_item_struct(&mut self, n: &'ast syn::ItemStruct) {
        self.declaration("Struct", n.ident.to_string(), &n.attrs, n.ident.span());
        visit::visit_item_struct(self, n);
    }

    /// Document every actual enum before traversing its source children.
    fn visit_item_enum(&mut self, n: &'ast syn::ItemEnum) {
        self.declaration("Enum", n.ident.to_string(), &n.attrs, n.ident.span());
        visit::visit_item_enum(self, n);
    }

    /// Document every actual union before traversing its source children.
    fn visit_item_union(&mut self, n: &'ast syn::ItemUnion) {
        self.declaration("Union", n.ident.to_string(), &n.attrs, n.ident.span());
        visit::visit_item_union(self, n);
    }

    /// Document every actual trait before traversing its source children.
    fn visit_item_trait(&mut self, n: &'ast syn::ItemTrait) {
        self.declaration("Trait", n.ident.to_string(), &n.attrs, n.ident.span());
        visit::visit_item_trait(self, n);
    }

    /// Document every actual trait alias before traversing its source children.
    fn visit_item_trait_alias(&mut self, n: &'ast syn::ItemTraitAlias) {
        self.declaration("Trait alias", n.ident.to_string(), &n.attrs, n.ident.span());
        visit::visit_item_trait_alias(self, n);
    }

    /// Document every actual constant before traversing its source children.
    fn visit_item_const(&mut self, n: &'ast syn::ItemConst) {
        self.declaration("Constant", n.ident.to_string(), &n.attrs, n.ident.span());
        visit::visit_item_const(self, n);
    }

    /// Document every actual static before traversing its source children.
    fn visit_item_static(&mut self, n: &'ast syn::ItemStatic) {
        self.declaration("Static", n.ident.to_string(), &n.attrs, n.ident.span());
        visit::visit_item_static(self, n);
    }

    /// Document every actual type before traversing its source children.
    fn visit_item_type(&mut self, n: &'ast syn::ItemType) {
        self.declaration("Type", n.ident.to_string(), &n.attrs, n.ident.span());
        visit::visit_item_type(self, n);
    }

    /// Document each authored module and retain its actual nesting during traversal.
    fn visit_item_mod(&mut self, n: &'ast syn::ItemMod) {
        self.declaration("Module", n.ident.to_string(), &n.attrs, n.ident.span());
        self.nesting += 1;
        visit::visit_item_mod(self, n);
        self.nesting -= 1;
    }

    /// Document every actual trait function before traversing its source children.
    fn visit_trait_item_fn(&mut self, n: &'ast syn::TraitItemFn) {
        self.declaration(
            "Trait function",
            n.sig.ident.to_string(),
            &n.attrs,
            n.sig.ident.span(),
        );
        visit::visit_trait_item_fn(self, n);
    }

    /// Document every actual trait constant before traversing its source children.
    fn visit_trait_item_const(&mut self, n: &'ast syn::TraitItemConst) {
        self.declaration(
            "Trait constant",
            n.ident.to_string(),
            &n.attrs,
            n.ident.span(),
        );
        visit::visit_trait_item_const(self, n);
    }

    /// Document every actual trait type before traversing its source children.
    fn visit_trait_item_type(&mut self, n: &'ast syn::TraitItemType) {
        self.declaration("Trait type", n.ident.to_string(), &n.attrs, n.ident.span());
        visit::visit_trait_item_type(self, n);
    }

    /// Document every actual implementation function before traversing its source children.
    fn visit_impl_item_fn(&mut self, n: &'ast syn::ImplItemFn) {
        self.declaration(
            "Implementation function",
            n.sig.ident.to_string(),
            &n.attrs,
            n.sig.ident.span(),
        );
        visit::visit_impl_item_fn(self, n);
    }

    /// Document every actual implementation constant before traversing its source children.
    fn visit_impl_item_const(&mut self, n: &'ast syn::ImplItemConst) {
        self.declaration(
            "Implementation constant",
            n.ident.to_string(),
            &n.attrs,
            n.ident.span(),
        );
        visit::visit_impl_item_const(self, n);
    }

    /// Document every actual implementation type before traversing its source children.
    fn visit_impl_item_type(&mut self, n: &'ast syn::ImplItemType) {
        self.declaration(
            "Implementation type",
            n.ident.to_string(),
            &n.attrs,
            n.ident.span(),
        );
        visit::visit_impl_item_type(self, n);
    }

    /// Document every actual foreign function before traversing its source children.
    fn visit_foreign_item_fn(&mut self, n: &'ast syn::ForeignItemFn) {
        self.declaration(
            "Foreign function",
            n.sig.ident.to_string(),
            &n.attrs,
            n.sig.ident.span(),
        );
        visit::visit_foreign_item_fn(self, n);
    }

    /// Document every actual foreign static before traversing its source children.
    fn visit_foreign_item_static(&mut self, n: &'ast syn::ForeignItemStatic) {
        self.declaration(
            "Foreign static",
            n.ident.to_string(),
            &n.attrs,
            n.ident.span(),
        );
        visit::visit_foreign_item_static(self, n);
    }

    /// Document every actual foreign type before traversing its source children.
    fn visit_foreign_item_type(&mut self, n: &'ast syn::ForeignItemType) {
        self.declaration(
            "Foreign type",
            n.ident.to_string(),
            &n.attrs,
            n.ident.span(),
        );
        visit::visit_foreign_item_type(self, n);
    }

    /// Document every actual variant before traversing its source children.
    fn visit_variant(&mut self, n: &'ast syn::Variant) {
        self.declaration("Variant", n.ident.to_string(), &n.attrs, n.ident.span());
        visit::visit_variant(self, n);
    }

    /// Document every actual field before traversing its source children.
    fn visit_field(&mut self, n: &'ast syn::Field) {
        self.declaration(
            "Field",
            n.ident
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| "positional field".into()),
            &n.attrs,
            n.ident
                .as_ref()
                .map(|name| name.span())
                .unwrap_or_else(|| n.ty.span()),
        );
        visit::visit_field(self, n);
    }
}
