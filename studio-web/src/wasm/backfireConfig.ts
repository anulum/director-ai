// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI Studio — shared configuration for the WASM lesson and kernel.

/** Complete JSON configuration accepted by the generated kernel. */
export interface BackfireConfig {
  /** Coherence floor for approval by the native scorer. */
  readonly coherence_threshold: number;
  /** Per-token score below which streaming stops immediately. */
  readonly hard_limit: number;
  /** Coherence level below which the native logic records a warning. */
  readonly soft_limit: number;
  /** Logical divergence weight in the composite scorer. */
  readonly w_logic: number;
  /** Factual divergence weight in the composite scorer. */
  readonly w_fact: number;
  /** Number of recent scores used for the sliding mean. */
  readonly window_size: number;
  /** Halt threshold for the sliding coherence mean. */
  readonly window_threshold: number;
  /** Number of recent scores used to estimate downward trend. */
  readonly trend_window: number;
  /** Maximum downward change permitted by the trend check. */
  readonly trend_threshold: number;
  /** Coherence history capacity in the native scorer configuration. */
  readonly history_window: number;
  /** Native scoring deadline field in milliseconds. */
  readonly deadline_ms: number;
  /** Native logit entropy limit field in bits per token. */
  readonly logit_entropy_limit: number;
}

/** Defaults from the native BackfireConfig. */
export const DEFAULT_BACKFIRE_CONFIG: BackfireConfig = Object.freeze({
  coherence_threshold: 0.6,
  hard_limit: 0.5,
  soft_limit: 0.7,
  w_logic: 0.6,
  w_fact: 0.4,
  window_size: 10,
  window_threshold: 0.55,
  trend_window: 5,
  trend_threshold: 0.15,
  history_window: 5,
  deadline_ms: 50,
  logit_entropy_limit: 1.2,
});
