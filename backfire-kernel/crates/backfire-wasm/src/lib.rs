// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Backfire WASM Edge
// (C) 1998-2026 Miroslav Sotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────

use backfire_types::{BackfireConfig, StreamSession, TokenEvent};
use std::collections::VecDeque;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmStreamingKernel {
    config: BackfireConfig,
    session: StreamSession,
    window: VecDeque<f64>,
    is_active: bool,
}

#[wasm_bindgen]
impl WasmStreamingKernel {
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<WasmStreamingKernel, JsValue> {
        let config = BackfireConfig::from_json(config_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(WasmStreamingKernel {
            window: VecDeque::with_capacity(config.window_size),
            config,
            session: StreamSession::default(),
            is_active: true,
        })
    }

    #[wasm_bindgen]
    pub fn process_token(&mut self, token: &str, score: f64) -> JsValue {
        let i = self.session.tokens.len();

        let mut event = TokenEvent {
            token: token.to_string(),
            index: i as u32,
            coherence: score,
            timestamp_s: 0.0, // Assuming JS manages real timestamps
            halted: false,
        };

        if !self.is_active {
            event.halted = true;
            return serde_wasm_bindgen::to_value(&self.session).unwrap();
        }

        self.session.tokens.push(token.to_string());
        self.session.coherence_history.push(score);
        self.window.push_back(score);

        if self.window.len() > self.config.window_size {
            self.window.pop_front();
        }

        // Check 1: Hard limit
        if score < self.config.hard_limit {
            self.halt(i, "hard_limit", score, self.config.hard_limit, &mut event);
            return serde_wasm_bindgen::to_value(&self.session).unwrap();
        }

        // Check 2: Sliding window average
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

        // Check 3: Downward trend
        if self.session.coherence_history.len() >= self.config.trend_window {
            let recent = &self.session.coherence_history
                [self.session.coherence_history.len() - self.config.trend_window..];
            let drop = recent[0] - recent[recent.len() - 1];
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

    #[wasm_bindgen]
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    #[wasm_bindgen]
    pub fn get_session(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.session).unwrap()
    }
}

impl WasmStreamingKernel {
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
