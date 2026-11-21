# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Architecture Validators

"""Validators of transformer architecture specs and their wire profiles.

A spec trained from scratch must convert to the serving profile it names,
so each profile fixes what its converter can express, as in ``llama.cpp``
at commit ``8887a48f``:

* ``llama``: a causal decoder with RMSNorm, SwiGLU, no biases,
  ``head_dim * n_heads == d_model`` and no local attention;
* ``qwen2``: the same, except that the query, key and value projections
  carry biases;
* ``qwen3``: a causal decoder with query-key RMSNorm, SwiGLU, no biases, an
  even ``head_dim`` (``head_dim * n_heads`` may differ from ``d_model``) and
  no local attention;
* ``modern-bert``: an encoder with LayerNorm, GeGLU or SwiGLU, RoPE, no
  biases, as many key-value heads as heads, ``head_dim * n_heads ==
  d_model``, a local attention window and unpacked single sequences;
* ``onnx_only_custom``: no converter lock.

The 1/d attention-logit scale of maximal-update width transfer is
expressible only in ``llama`` and ``onnx_only_custom``. Width transfer on
any pool other than owned hardware needs its empirical receipt.
"""

from __future__ import annotations

from ._checks import (
    require_bool,
    require_choice,
    require_int,
)
from .architecture_specs import (
    ACTIVATIONS,
    ARCHITECTURE_KINDS,
    DOCUMENT_MASK_MODES,
    GGUF_WIRE_PROFILES,
    MAX_SEED,
    NORM_KINDS,
    POOL_FUNDING_KINDS,
    POSITIONAL_ENCODINGS,
    ROPE_ENCODINGS,
    TransformerArchitectureSpec,
)
from .errors import ArchitectureSpecError
from .width_and_position_contracts import (
    optional_count,
    positive_float,
    validate_mup_spec,
    validate_yarn_rope_scaling,
)

E = ArchitectureSpecError

_DECODER: dict[str, object] = {
    "architecture_kind": "causal_decoder",
    "activation": "swiglu",
}
_NO_BIAS: dict[str, object] = {"qkv_bias": False, "linear_bias": False}
_NO_LOCAL: dict[str, object] = {
    "local_attention_window": None,
    "global_attn_every_n_layers": None,
}
PROFILE_LOCKS: dict[str, dict[str, object]] = {
    "llama": {**_DECODER, "norm_kind": "rmsnorm", **_NO_BIAS, **_NO_LOCAL},
    "qwen2": {**_DECODER, "norm_kind": "rmsnorm", **_NO_BIAS, **_NO_LOCAL}
    | {"qkv_bias": True},
    "qwen3": {**_DECODER, "norm_kind": "qknorm_rmsnorm", **_NO_BIAS}
    | {"local_attention_window": None},
    "modern-bert": {
        "architecture_kind": "encoder_only",
        "norm_kind": "layernorm",
        "positional_encoding": "rope",
        **_NO_BIAS,
        "document_mask_mode": "unpacked_single_sequence",
    },
    "onnx_only_custom": {},
}
"""The fields each wire profile fixes, with their required values."""

FULL_WIDTH_PROFILES: frozenset[str] = frozenset({"llama", "qwen2", "modern-bert"})
"""Profiles whose converters need ``head_dim * n_heads == d_model``."""

MUP_INV_D_PROFILES: frozenset[str] = frozenset({"llama", "onnx_only_custom"})
"""Profiles that can express the 1/d attention-logit scale."""


def _check_shape(spec: TransformerArchitectureSpec) -> None:
    """Check the spec's field types and the numeric rules of every profile."""
    require_choice(spec.schema, "schema", {"transformer-architecture-spec.v1"}, E)
    require_choice(spec.family, "family", {"language", "dynamics"}, E)
    require_choice(spec.architecture_kind, "architecture_kind", ARCHITECTURE_KINDS, E)
    require_choice(spec.gguf_wire_profile, "gguf_wire_profile", GGUF_WIRE_PROFILES, E)
    for name in (
        "d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim", "ffn_hidden_dim",
        "max_context_length", "vocab_or_channel_dim", "output_dim",
    ):  # fmt: skip
        require_int(getattr(spec, name), name, E, minimum=1)
    if spec.n_heads % spec.n_kv_heads:
        raise E("n_heads must be a multiple of n_kv_heads")
    require_choice(spec.activation, "activation", ACTIVATIONS, E)
    require_choice(spec.norm_kind, "norm_kind", NORM_KINDS, E)
    require_choice(
        spec.positional_encoding, "positional_encoding", POSITIONAL_ENCODINGS, E
    )
    require_choice(
        spec.document_mask_mode, "document_mask_mode", DOCUMENT_MASK_MODES, E
    )
    for name in ("linear_bias", "qkv_bias", "tie_word_embeddings"):
        require_bool(getattr(spec, name), name, E)
    positive_float(spec.norm_eps, "norm_eps")
    positive_float(spec.init_std, "init_std")
    if require_int(spec.init_seed, "init_seed", E, minimum=0) >= MAX_SEED:
        raise E("init_seed must be below 2**63")
    window = optional_count(spec.local_attention_window, "local_attention_window")
    if (
        optional_count(spec.global_attn_every_n_layers, "global_attn_every_n_layers")
        and window is None
    ):
        raise E("global_attn_every_n_layers needs a local_attention_window")


def _check_positions(spec: TransformerArchitectureSpec) -> None:
    """Check the rotary bases and YaRN scaling against the position encoding."""
    if spec.positional_encoding in ROPE_ENCODINGS:
        positive_float(spec.rope_theta, "rope_theta")
    elif spec.rope_theta is not None:
        raise E("rope_theta is only for rotary position encodings")
    if spec.local_rope_theta is not None:
        if spec.local_attention_window is None:
            raise E("local_rope_theta needs a local_attention_window")
        positive_float(spec.local_rope_theta, "local_rope_theta")
    if spec.positional_encoding == "yarn_rope":
        if spec.rope_scaling is None:
            raise E("yarn_rope needs rope_scaling")
        validate_yarn_rope_scaling(
            spec.rope_scaling, max_context_length=spec.max_context_length
        )
    elif spec.rope_scaling is not None:
        raise E("rope_scaling is only for yarn_rope")


def _check_profile(spec: TransformerArchitectureSpec) -> None:
    """Check the fields the spec's wire profile fixes."""
    profile = spec.gguf_wire_profile
    for name, value in PROFILE_LOCKS[profile].items():
        if getattr(spec, name) != value:
            raise E(f"wire profile {profile!r} needs {name} == {value!r}")
    if profile in FULL_WIDTH_PROFILES and spec.head_dim * spec.n_heads != spec.d_model:
        raise E(f"wire profile {profile!r} needs head_dim * n_heads == d_model")
    if profile == "qwen3" and spec.head_dim % 2:
        raise E("wire profile 'qwen3' needs an even head_dim")
    if profile == "modern-bert":
        if spec.activation not in {"geglu", "swiglu"}:
            raise E("wire profile 'modern-bert' needs geglu or swiglu")
        if spec.n_kv_heads != spec.n_heads:
            raise E("wire profile 'modern-bert' needs n_kv_heads == n_heads")
        if spec.local_attention_window is None:
            raise E("wire profile 'modern-bert' needs a local_attention_window")


def check_transformer_architecture_shape(
    spec: TransformerArchitectureSpec,
) -> TransformerArchitectureSpec:
    """Check every rule of a spec that does not depend on its funding pool.

    Parameters
    ----------
    spec : TransformerArchitectureSpec
        The spec to check.

    Returns
    -------
    TransformerArchitectureSpec
        The same spec.

    Raises
    ------
    ArchitectureSpecError
        If a field is malformed, a rotary or YaRN rule fails, or the wire
        profile cannot express the spec.
    MalformedDigestError
        If a width-transfer digest is malformed.
    """
    if not isinstance(spec, TransformerArchitectureSpec):
        raise E("a TransformerArchitectureSpec is required")
    _check_shape(spec)
    _check_positions(spec)
    _check_profile(spec)
    if spec.mup_spec is not None:
        validate_mup_spec(spec.mup_spec, d_model=spec.d_model)
        if (
            spec.mup_spec.attention_logit_scale_mode == "mup_inv_d"
            and spec.gguf_wire_profile not in MUP_INV_D_PROFILES
        ):
            raise E(f"wire profile {spec.gguf_wire_profile!r} cannot express mup_inv_d")
    return spec


def validate_transformer_architecture_spec(
    spec: TransformerArchitectureSpec, *, funding_kind: str
) -> TransformerArchitectureSpec:
    """Check a spec, including the width-transfer receipt its funding pool needs.

    Parameters
    ----------
    spec : TransformerArchitectureSpec
        The spec to check.
    funding_kind : str
        The funding kind of the pool that pays for the run.

    Returns
    -------
    TransformerArchitectureSpec
        The same spec.

    Raises
    ------
    ArchitectureSpecError
        If the shape check fails, the funding kind is unknown, or width
        transfer outside owned hardware has no receipt digest.
    MalformedDigestError
        If a width-transfer digest is malformed.
    """
    check_transformer_architecture_shape(spec)
    funding = require_choice(funding_kind, "funding_kind", POOL_FUNDING_KINDS, E)
    if (
        spec.mup_spec is not None
        and funding != "local_owned_hardware"
        and spec.mup_spec.proxy_validation_receipt_digest is None
    ):
        raise E("width transfer outside owned hardware needs its validation receipt")
    return spec
