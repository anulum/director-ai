// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — WebAssembly streaming halt kernel

#![deny(missing_docs)]
//! Stateful halt decisions for host-supplied coherence scores.
//!
//! This crate provides no scorer or clock. Nonfinite scores become zero,
//! matching the native streaming kernel; event timestamps remain zero.

use backfire_core::signals::trend_drop;
use backfire_types::{BackfireConfig, StreamSession, TokenEvent};
use std::collections::VecDeque;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
/// A single token stream with hard-limit, window and trend checks.
///
/// The first halt is permanent. Create another instance for a new stream.
pub struct WasmStreamingKernel {
    /// Validated native thresholds and window sizes for this stream.
    config: BackfireConfig,
    /// Session history and the first halt decision returned to the host.
    session: StreamSession,
    /// Latest scores used by the sliding-average check.
    window: VecDeque<f64>,
    /// True until the first halt locks the session.
    is_active: bool,
}

#[wasm_bindgen]
impl WasmStreamingKernel {
    /// Parse and validate the complete native `BackfireConfig` JSON schema.
    ///
    /// Returns a JavaScript error value for malformed or invalid configuration.
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<WasmStreamingKernel, JsValue> {
        let config = BackfireConfig::from_json(config_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        config
            .validate()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(WasmStreamingKernel {
            window: VecDeque::with_capacity(config.window_size),
            config,
            session: StreamSession::default(),
            is_active: true,
        })
    }

    /// Append one token and return a serialised session snapshot.
    ///
    /// The host supplies the coherence score, normally in `[0, 1]`. NaN and
    /// either infinity become zero before recording or gating; finite values
    /// are not clamped, matching the native streaming kernel. Checks run in
    /// hard-limit, window-average, downward-trend order. The trend uses the
    /// shared least-squares projected drop across the recent score window.
    /// Calls after a halt
    /// return the original snapshot without recording the supplied token.
    #[wasm_bindgen]
    pub fn process_token(&mut self, token: &str, score: f64) -> JsValue {
        if !self.is_active {
            return serde_wasm_bindgen::to_value(&self.session).unwrap();
        }

        let score = if score.is_finite() { score } else { 0.0 };
        let i = self.session.tokens.len();

        let mut event = TokenEvent {
            token: token.to_string(),
            index: i as u32,
            coherence: score,
            timestamp_s: 0.0,
            halted: false,
        };

        self.session.tokens.push(token.to_string());
        self.session.coherence_history.push(score);
        self.window.push_back(score);

        if self.window.len() > self.config.window_size {
            self.window.pop_front();
        }

        if score < self.config.hard_limit {
            self.halt(i, "hard_limit", score, self.config.hard_limit, &mut event);
            return serde_wasm_bindgen::to_value(&self.session).unwrap();
        }

        if self.window.len() >= self.config.window_size {
            let avg: f64 = self.window.iter().sum::<f64>() / self.window.len() as f64;
            if avg < self.config.window_threshold {
                self.halt(
                    i,
                    "window_avg",
                    avg,
                    self.config.window_threshold,
                    &mut event,
                );
                return serde_wasm_bindgen::to_value(&self.session).unwrap();
            }
        }

        if self.session.coherence_history.len() >= self.config.trend_window {
            let recent = &self.session.coherence_history
                [self.session.coherence_history.len() - self.config.trend_window..];
            let drop = trend_drop(recent);
            if drop > self.config.trend_threshold {
                self.halt(
                    i,
                    "downward_trend",
                    drop,
                    self.config.trend_threshold,
                    &mut event,
                );
                return serde_wasm_bindgen::to_value(&self.session).unwrap();
            }
        }

        self.session.events.push(event);
        serde_wasm_bindgen::to_value(&self.session).unwrap()
    }

    /// Return whether this stream can still accept tokens.
    #[wasm_bindgen]
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Return an independent serialised snapshot, including all token events.
    #[wasm_bindgen]
    pub fn get_session(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.session).unwrap()
    }
}

impl WasmStreamingKernel {
    /// Record the failing event and seal the session at its first halt.
    fn halt(&mut self, i: usize, reason_key: &str, val: f64, limit: f64, event: &mut TokenEvent) {
        event.halted = true;
        self.session.halted = true;
        self.session.halt_index = i as i32;

        let operator = if reason_key == "downward_trend" {
            ">"
        } else {
            "<"
        };
        self.session.halt_reason = format!("{reason_key} ({val:.4} {operator} {limit})");

        self.is_active = false;
        self.session.events.push(event.clone());
    }
}
