// SPDX-License-Identifier: Apache-2.0
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
//
// Numeric wasm32 coverage is not claimed: the installed target has no
// profiler_builtins runtime (instrumented cargo check fails with E0463).
// These tests instead execute the public WASM API in Node, including config
// refusal, each halt decision and boundary, eviction, snapshots and nonfinite
// scores. Browser/Worker package tests cover the separate generated web entry.

//! Node integration tests for the public WebAssembly streaming API.
//! They check native decision semantics and JavaScript session snapshots.

#![cfg(target_arch = "wasm32")]

use backfire_wasm::WasmStreamingKernel;
use js_sys::Reflect;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;

/// Return a complete native configuration for the JavaScript-facing constructor.
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

/// Read a named property from a JavaScript session or event object.
fn get(obj: &JsValue, key: &str) -> JsValue {
    Reflect::get(obj, &JsValue::from_str(key)).unwrap()
}

/// Read the halted flag from a serialised session snapshot.
fn is_halted(session: &JsValue) -> bool {
    get(session, "halted").as_bool().unwrap_or(false)
}

/// Read the halt reason as a JavaScript string.
fn halt_reason(session: &JsValue) -> String {
    get(session, "halt_reason").as_string().unwrap_or_default()
}

/// Apply numeric overrides while retaining the other configuration fields.
fn config_with(changes: &[(&str, f64)]) -> String {
    let config = js_sys::JSON::parse(default_config()).unwrap();
    for (field, value) in changes {
        Reflect::set(
            &config,
            &JsValue::from_str(field),
            &JsValue::from_f64(*value),
        )
        .unwrap();
    }
    js_sys::JSON::stringify(&config)
        .unwrap()
        .as_string()
        .unwrap()
}

/// Read a session property as a JavaScript array.
fn array_field(session: &JsValue, field: &str) -> js_sys::Array {
    js_sys::Array::from(&get(session, field))
}

/// Check that the zero fallback drives both the halt decision and the event trace.
fn assert_nonfinite_score_becomes_zero(score: f64) {
    let mut kernel = WasmStreamingKernel::new(default_config()).unwrap();
    let session = kernel.process_token("unscored", score);
    assert!(is_halted(&session));
    assert!(!kernel.is_active());
    assert!(halt_reason(&session).starts_with("hard_limit (0.000"));
    assert_eq!(get(&session, "halt_index").as_f64(), Some(0.0));
    assert_eq!(
        array_field(&session, "coherence_history").get(0).as_f64(),
        Some(0.0)
    );
    let event = array_field(&session, "events").get(0);
    assert_eq!(get(&event, "coherence").as_f64(), Some(0.0));
    assert_eq!(get(&event, "halted").as_bool(), Some(true));
}

/// Check that a valid configuration starts an active stream.
#[wasm_bindgen_test]
fn construction_is_active() {
    let kernel = WasmStreamingKernel::new(default_config()).unwrap();
    assert!(kernel.is_active());
}

/// Keep the session active across several passing score windows.
#[wasm_bindgen_test]
fn all_passing_tokens_do_not_halt() {
    let mut kernel = WasmStreamingKernel::new(default_config()).unwrap();
    for i in 0..5 {
        let session = kernel.process_token(&format!("t{i}"), 0.9);
        assert!(!is_halted(&session));
    }
    assert!(kernel.is_active());
}

/// Halt on a low individual score and report the hard-limit reason.
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

/// Halt when three passing token scores fill a sub-threshold average window.
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

/// Check that a steep recent decline halts with the native fixture.
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

/// Keep the kernel halted after a subsequent high score.
#[wasm_bindgen_test]
fn halt_is_irrevocable() {
    let mut kernel = WasmStreamingKernel::new(default_config()).unwrap();
    kernel.process_token("a", 0.1);
    assert!(!kernel.is_active());
    let session = kernel.process_token("b", 0.99);
    assert!(is_halted(&session));
}

/// Reject non-JSON input at construction.
#[wasm_bindgen_test]
fn malformed_config_returns_error() {
    assert!(WasmStreamingKernel::new("not json").is_err());
}

/// Return native validation errors for invalid thresholds, weights and window sizes.
#[wasm_bindgen_test]
fn canonical_config_validation_rejects_invalid_parameters() {
    for (field, value, message) in [
        ("coherence_threshold", -0.1, "coherence_threshold"),
        ("coherence_threshold", 1.1, "coherence_threshold"),
        ("hard_limit", -0.1, "hard_limit"),
        ("hard_limit", 1.1, "hard_limit"),
        ("w_logic", 0.2, "w_logic + w_fact"),
        ("window_size", 0.0, "window_size"),
        ("trend_window", 0.0, "trend_window"),
        ("trend_window", 1.0, "trend_window"),
        ("trend_threshold", 0.0, "trend_threshold"),
        ("trend_threshold", -0.1, "trend_threshold"),
        ("deadline_ms", 0.0, "deadline_ms"),
    ] {
        let result = WasmStreamingKernel::new(&config_with(&[(field, value)]));
        let error = result.err().expect("invalid configuration was accepted");
        assert!(
            error.as_string().unwrap().contains(message),
            "field={field}"
        );
    }
}

/// Reject partial host configuration when required native fields are absent.
#[wasm_bindgen_test]
fn incomplete_studio_config_is_refused() {
    let error = WasmStreamingKernel::new(r#"{"threshold":0.65}"#)
        .err()
        .expect("missing required configuration was accepted");
    assert!(error.as_string().unwrap().contains("coherence_threshold"));
}

/// Check that NaN is recorded and evaluated as zero.
#[wasm_bindgen_test]
fn nan_score_matches_native_zero_score() {
    assert_nonfinite_score_becomes_zero(f64::NAN);
}

/// Check that positive infinity uses the zero fallback.
#[wasm_bindgen_test]
fn positive_infinity_matches_native_zero_score() {
    assert_nonfinite_score_becomes_zero(f64::INFINITY);
}

/// Check that negative infinity uses the zero fallback.
#[wasm_bindgen_test]
fn negative_infinity_matches_native_zero_score() {
    assert_nonfinite_score_becomes_zero(f64::NEG_INFINITY);
}

/// Allow the zero fallback when it equals the hard limit.
#[wasm_bindgen_test]
fn zero_hard_limit_keeps_native_strict_comparison_for_nonfinite_input() {
    let mut kernel = WasmStreamingKernel::new(&config_with(&[("hard_limit", 0.0)])).unwrap();
    let session = kernel.process_token("unscored", f64::NAN);
    assert!(!is_halted(&session));
    assert!(kernel.is_active());
    assert_eq!(
        array_field(&session, "coherence_history").get(0).as_f64(),
        Some(0.0)
    );
}

/// Keep an individual score equal to the hard limit.
#[wasm_bindgen_test]
fn hard_limit_equality_does_not_halt() {
    let mut kernel = WasmStreamingKernel::new(default_config()).unwrap();
    let session = kernel.process_token("boundary", 0.5);
    assert!(!is_halted(&session));
    assert!(kernel.is_active());
}

/// Keep a full window whose mean equals the configured floor.
#[wasm_bindgen_test]
fn window_threshold_equality_does_not_halt() {
    let mut kernel = WasmStreamingKernel::new(default_config()).unwrap();
    for token in ["a", "b", "c"] {
        assert!(!is_halted(&kernel.process_token(token, 0.55)));
    }
    assert!(kernel.is_active());
}

/// Keep a stream whose projected drop equals the configured tolerance.
#[wasm_bindgen_test]
fn trend_threshold_equality_does_not_halt() {
    let mut kernel = WasmStreamingKernel::new(&config_with(&[("trend_threshold", 0.25)])).unwrap();
    for (token, score) in [("a", 0.875), ("b", 0.75), ("c", 0.625)] {
        assert!(!is_halted(&kernel.process_token(token, score)));
    }
    assert!(kernel.is_active());
}

/// Halt once the oldest high scores have left the average window.
#[wasm_bindgen_test]
fn window_discards_old_high_scores_before_deciding() {
    let config = config_with(&[("window_threshold", 0.7), ("trend_threshold", 1.0)]);
    let mut kernel = WasmStreamingKernel::new(&config).unwrap();
    for score in [0.9, 0.9, 0.9, 0.6, 0.6] {
        assert!(!is_halted(&kernel.process_token("kept", score)));
    }
    let session = kernel.process_token("halted", 0.6);
    assert!(halt_reason(&session).starts_with("window_avg"));
    assert_eq!(get(&session, "halt_index").as_f64(), Some(5.0));
    assert_eq!(array_field(&session, "tokens").length(), 6);
}

/// Allow the stream when the fitted drop stays within the limit over the recent window.
#[wasm_bindgen_test]
fn trend_uses_recent_window_instead_of_entire_history() {
    let mut kernel = WasmStreamingKernel::new(&config_with(&[("trend_threshold", 0.25)])).unwrap();
    for score in [0.9, 0.8, 0.7, 0.6] {
        assert!(!is_halted(&kernel.process_token("kept", score)));
    }
    assert!(kernel.is_active());
}

/// Report the hard limit when all three checks could halt at the same token.
#[wasm_bindgen_test]
fn hard_limit_precedes_simultaneous_window_and_trend_halts() {
    let mut kernel = WasmStreamingKernel::new(&config_with(&[("window_threshold", 0.8)])).unwrap();
    kernel.process_token("a", 0.9);
    kernel.process_token("b", 0.9);
    let session = kernel.process_token("c", 0.1);
    assert!(halt_reason(&session).starts_with("hard_limit"));
}

/// Report the average first when both average and trend checks could halt.
#[wasm_bindgen_test]
fn window_halt_precedes_simultaneous_trend_halt() {
    let mut kernel = WasmStreamingKernel::new(&config_with(&[("window_threshold", 0.75)])).unwrap();
    kernel.process_token("a", 0.9);
    kernel.process_token("b", 0.7);
    let session = kernel.process_token("c", 0.5);
    assert!(halt_reason(&session).starts_with("window_avg"));
}

/// Compare snapshots before and after a token submitted to each halted stream.
#[wasm_bindgen_test]
fn every_halt_preserves_original_session_on_later_calls() {
    for scores in [vec![0.1], vec![0.54, 0.54, 0.54], vec![0.95, 0.85, 0.55]] {
        let mut kernel = WasmStreamingKernel::new(default_config()).unwrap();
        for score in scores {
            kernel.process_token("original", score);
        }
        assert!(!kernel.is_active());
        let original = js_sys::JSON::stringify(&kernel.get_session()).unwrap();
        let after = kernel.process_token("must-not-append", f64::NAN);
        assert_eq!(original, js_sys::JSON::stringify(&after).unwrap());
        assert!(!kernel.is_active());
    }
}

/// Check that an initial snapshot contains no tokens, scores or events.
#[wasm_bindgen_test]
fn empty_session_has_no_events_or_halt() {
    let kernel = WasmStreamingKernel::new(default_config()).unwrap();
    let session = kernel.get_session();
    assert!(!is_halted(&session));
    assert_eq!(get(&session, "halt_index").as_f64(), Some(0.0));
    for field in ["tokens", "coherence_history", "events"] {
        assert_eq!(array_field(&session, field).length(), 0);
    }
}

/// Check that an earlier snapshot stays fixed and Unicode tokens retain order.
#[wasm_bindgen_test]
fn snapshots_are_independent_and_events_preserve_token_order() {
    let mut kernel = WasmStreamingKernel::new(default_config()).unwrap();
    let first = kernel.process_token("Ž", 0.9);
    let second = kernel.process_token("中", 0.8);
    assert_eq!(array_field(&first, "tokens").length(), 1);
    assert_eq!(array_field(&second, "tokens").length(), 2);
    for (index, token, score) in [(0, "Ž", 0.9), (1, "中", 0.8)] {
        let event = array_field(&second, "events").get(index);
        assert_eq!(get(&event, "token").as_string().as_deref(), Some(token));
        assert_eq!(get(&event, "index").as_f64(), Some(f64::from(index)));
        assert_eq!(get(&event, "coherence").as_f64(), Some(score));
        assert_eq!(get(&event, "halted").as_bool(), Some(false));
    }
}

/// Record finite scores outside the unit interval without clamping them.
#[wasm_bindgen_test]
fn finite_scores_keep_native_values_without_clamping() {
    let mut kernel = WasmStreamingKernel::new(default_config()).unwrap();
    let passing = kernel.process_token("high", 1.25);
    assert!(!is_halted(&passing));
    let halted = kernel.process_token("low", -0.25);
    assert!(is_halted(&halted));
    let scores = array_field(&halted, "coherence_history");
    assert_eq!(scores.get(0).as_f64(), Some(1.25));
    assert_eq!(scores.get(1).as_f64(), Some(-0.25));
}

/// Check the shared fitted-drop decision and index with equal endpoints.
#[wasm_bindgen_test]
fn regression_detects_decline_with_equal_endpoints() {
    let config = config_with(&[
        ("window_size", 10.0),
        ("trend_window", 5.0),
        ("trend_threshold", 0.15),
    ]);
    let mut kernel = WasmStreamingKernel::new(&config).unwrap();
    for score in [0.8, 1.0, 0.8, 0.5, 0.8] {
        kernel.process_token("observed", score);
    }
    let session = kernel.get_session();
    assert!(is_halted(&session));
    assert!(halt_reason(&session).starts_with("downward_trend (0.2000"));
    assert_eq!(get(&session, "halt_index").as_f64(), Some(4.0));
}

/// Keep all five tokens when the fitted trend rises despite falling endpoints.
#[wasm_bindgen_test]
fn regression_does_not_halt_on_declining_endpoints_with_rising_trend() {
    let config = config_with(&[
        ("window_size", 10.0),
        ("trend_window", 5.0),
        ("trend_threshold", 0.15),
    ]);
    let mut kernel = WasmStreamingKernel::new(&config).unwrap();
    for score in [0.9, 0.5, 0.8, 1.0, 0.7] {
        assert!(!is_halted(&kernel.process_token("observed", score)));
    }
    assert!(kernel.is_active());
    assert_eq!(array_field(&kernel.get_session(), "tokens").length(), 5);
}
