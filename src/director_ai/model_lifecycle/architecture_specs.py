# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Architecture Specifications

"""Records that specify a transformer trained from scratch.

:class:`TransformerArchitectureSpec` fixes a model's shape, its norm,
activation and position encoding, and the serving wire profile it must
convert to. Width transfer is described by
:class:`MaximalUpdateParametrisationSpec`, whose empirical check is a
:class:`MupValidationReceipt`; YaRN position scaling by
:class:`YarnRopeScalingSpec`. The rules are in :mod:`.architecture_contracts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ArchitectureKind = Literal[
    "encoder_only", "causal_decoder", "encoder_decoder", "dynamics_transformer"
]
GgufWireProfile = Literal["llama", "qwen2", "qwen3", "modern-bert", "onnx_only_custom"]
DocumentMaskMode = Literal["block_diagonal_reset_rope", "unpacked_single_sequence"]
PoolFundingKind = Literal[
    "local_owned_hardware",
    "monthly_cloud_credit",
    "granted_free_credit_or_tier",
    "per_run_owner_approved_paid",
]

ARCHITECTURE_KINDS: frozenset[str] = frozenset(
    {"encoder_only", "causal_decoder", "encoder_decoder", "dynamics_transformer"}
)
GGUF_WIRE_PROFILES: frozenset[str] = frozenset(
    {"llama", "qwen2", "qwen3", "modern-bert", "onnx_only_custom"}
)
DOCUMENT_MASK_MODES: frozenset[str] = frozenset(
    {"block_diagonal_reset_rope", "unpacked_single_sequence"}
)
POOL_FUNDING_KINDS: frozenset[str] = frozenset(
    {
        "local_owned_hardware",
        "monthly_cloud_credit",
        "granted_free_credit_or_tier",
        "per_run_owner_approved_paid",
    }
)
ACTIVATIONS: frozenset[str] = frozenset({"swiglu", "geglu", "gelu_approx", "relu2"})
NORM_KINDS: frozenset[str] = frozenset({"rmsnorm", "qknorm_rmsnorm", "layernorm"})
POSITIONAL_ENCODINGS: frozenset[str] = frozenset(
    {"rope", "yarn_rope", "alibi", "continuous_time_fourier"}
)
ROPE_ENCODINGS: frozenset[str] = frozenset({"rope", "yarn_rope"})
"""The position encodings that need a ``rope_theta``."""

MAX_SEED = 2**63
"""Seeds lie in ``[0, 2**63)``."""


@dataclass(frozen=True, slots=True)
class YarnRopeScalingSpec:
    """YaRN rotary position scaling parameters aligned with GGUF conversion."""

    schema: Literal["yarn-rope-scaling.v1"]
    factor: float
    original_max_position_embeddings: int
    attention_factor: float | None
    extrapolation_factor: float | None
    beta_fast: float
    beta_slow: float


@dataclass(frozen=True, slots=True)
class MupValidationReceipt:
    """Empirical multi-width coordinate-check receipt for width-transfer runs."""

    schema: Literal["mup-validation-receipt.v1"]
    proxy_widths_tested: tuple[int, ...]
    optimal_base_lr: float
    max_optimal_lr_drift_ratio: float
    coordinate_check_steps: int
    dedup_report_digest: str
    verified_at_utc: str


@dataclass(frozen=True, slots=True)
class MaximalUpdateParametrisationSpec:
    """Explicit width-transfer scaling exponents and empirical receipt digest."""

    schema: Literal["mup-spec.v1"]
    base_d_model: int
    target_d_model: int
    input_scale: float
    output_zero_init: bool
    attention_logit_scale_mode: Literal["standard_inv_sqrt_d", "mup_inv_d"]
    hidden_init_std_scale_power: float
    embedding_lr_scale_power: float
    hidden_matrix_lr_scale_power: float
    norm_and_bias_lr_scale_power: float
    readout_lr_scale_power: float
    proxy_validation_receipt_digest: str | None


@dataclass(frozen=True, slots=True)
class TransformerArchitectureSpec:
    """From-scratch transformer specification locked to serving wire profiles."""

    schema: Literal["transformer-architecture-spec.v1"]
    family: Literal["language", "dynamics"]
    architecture_kind: ArchitectureKind
    gguf_wire_profile: GgufWireProfile
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_hidden_dim: int
    activation: Literal["swiglu", "geglu", "gelu_approx", "relu2"]
    norm_kind: Literal["rmsnorm", "qknorm_rmsnorm", "layernorm"]
    norm_eps: float
    linear_bias: bool
    qkv_bias: bool
    positional_encoding: Literal[
        "rope", "yarn_rope", "alibi", "continuous_time_fourier"
    ]
    rope_theta: float | None
    local_rope_theta: float | None
    rope_scaling: YarnRopeScalingSpec | None
    local_attention_window: int | None
    global_attn_every_n_layers: int | None
    max_context_length: int
    vocab_or_channel_dim: int
    output_dim: int
    tie_word_embeddings: bool
    document_mask_mode: DocumentMaskMode
    init_std: float
    init_seed: int
    mup_spec: MaximalUpdateParametrisationSpec | None
