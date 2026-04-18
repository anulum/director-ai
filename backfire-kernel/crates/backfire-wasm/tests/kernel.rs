// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — backfire-wasm integration tests
//
// These run under `wasm-pack test --node` against a real WASM
// build. Each test drives ``WasmStreamingKernel`` through a sequence
// of (token, score) pairs and asserts on fields of the returned
// session object via ``js_sys::Reflect`` — no serde_json dependency
// is pulled into the WASM crate.

#![cfg(target_arch = "wasm32")]

use backfire_wasm::WasmStreamingKernel;
use js_sys::Reflect;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;

fn default_config() -> &'static str {
    r#"{
        "coherence_threshold": 0.6,
        "hard_limit": 0.5,
        "soft_limit": 0.7,
        "w_logic": 0.6,
        "w_fact": 0.4,
        "window_size": 3,
        "window_threshold": 0.55,
        "trend_window": 3,
        "trend_threshold": 0.2,
        "history_window": 5,
        "deadline_ms": 50,
        "logit_entropy_limit": 1.2
    }"#
}

fn get(obj: &JsValue, key: &str) -> JsValue {
    Reflect::get(obj, &JsValue::from_str(key)).unwrap()
}

fn is_halted(session: &JsValue) -> bool {
    get(session, "halted").as_bool().unwrap_or(false)
}

fn halt_reason(session: &JsValue) -> String {
    get(session, "halt_reason").as_string().unwrap_or_default()
}

#[wasm_bindgen_test]
fn construction_is_active() {
    let kernel = WasmStreamingKernel::new(default_config()).unwrap();
    assert!(kernel.is_active());
}

#[wasm_bindgen_test]
fn all_passing_tokens_do_not_halt() {
    let mut kernel = WasmStreamingKernel::new(default_config()).unwrap();
    for i in 0..5 {
        let session = kernel.process_token(&format!("t{i}"), 0.9);
        assert!(!is_halted(&session));
    }
    assert!(kernel.is_active());
}

#[wasm_bindgen_test]
fn sub_threshold_token_triggers_hard_halt() {
    let mut kernel = WasmStreamingKernel::new(default_config()).unwrap();
    kernel.process_token("a", 0.9);
    let session = kernel.process_token("b", 0.1);
    assert!(is_halted(&session));
    let reason = halt_reason(&session);
    assert!(reason.starts_with("hard_limit"), "reason={reason}");
    assert!(!kernel.is_active());
}

#[wasm_bindgen_test]
fn window_average_below_threshold_triggers_halt() {
    // hard_limit 0.5 leaves room for window avg to fall without any
    // individual sub-threshold score. With window_size=3 and
    // threshold 0.55 we need three consecutive scores averaging
    // below 0.55.
    let mut kernel = WasmStreamingKernel::new(default_config()).unwrap();
    kernel.process_token("a", 0.54);
    kernel.process_token("b", 0.54);
    let session = kernel.process_token("c", 0.54);
    assert!(is_halted(&session));
    assert!(halt_reason(&session).starts_with("window_avg"));
}

#[wasm_bindgen_test]
fn downward_trend_triggers_halt() {
    let mut kernel = WasmStreamingKernel::new(default_config()).unwrap();
    // Start high, keep high enough for window not to trip, then fall
    // sharply so the recent[0] - recent[-1] drop exceeds 0.2.
    kernel.process_token("a", 0.95);
    kernel.process_token("b", 0.85);
    let session = kernel.process_token("c", 0.55);
    assert!(is_halted(&session));
}

#[wasm_bindgen_test]
fn halt_is_irrevocable() {
    let mut kernel = WasmStreamingKernel::new(default_config()).unwrap();
    kernel.process_token("a", 0.1);
    assert!(!kernel.is_active());
    let session = kernel.process_token("b", 0.99);
    assert!(is_halted(&session));
}

#[wasm_bindgen_test]
fn malformed_config_returns_error() {
    assert!(WasmStreamingKernel::new("not json").is_err());
}
