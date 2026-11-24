# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Architecture Tests

"""Transformer architecture specs, their wire profiles, YaRN and width transfer."""

from __future__ import annotations

from dataclasses import replace

import pytest

from director_ai.model_lifecycle import (
    POOL_FUNDING_KINDS,
    ArchitectureSpecError,
    MalformedDigestError,
    MaximalUpdateParametrisationSpec,
    MupValidationReceipt,
    TransformerArchitectureSpec,
    YarnRopeScalingSpec,
    canonical_json_bytes,
    check_transformer_architecture_shape,
    from_canonical_json_bytes,
    validate_mup_spec,
    validate_mup_validation_receipt,
    validate_transformer_architecture_spec,
    validate_yarn_rope_scaling,
)
from tests.model_lifecycle_support import UTC, digest_of

LOCAL = "local_owned_hardware"
CLOUD = "monthly_cloud_credit"


def spec(**changes: object) -> TransformerArchitectureSpec:
    """Return a valid ``llama`` language spec with ``changes`` applied."""
    base = TransformerArchitectureSpec(
        schema="transformer-architecture-spec.v1",
        family="language",
        architecture_kind="causal_decoder",
        gguf_wire_profile="llama",
        d_model=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        head_dim=64,
        ffn_hidden_dim=1408,
        activation="swiglu",
        norm_kind="rmsnorm",
        norm_eps=1e-5,
        linear_bias=False,
        qkv_bias=False,
        positional_encoding="rope",
        rope_theta=10000.0,
        local_rope_theta=None,
        rope_scaling=None,
        local_attention_window=None,
        global_attn_every_n_layers=None,
        max_context_length=4096,
        vocab_or_channel_dim=32000,
        output_dim=32000,
        tie_word_embeddings=True,
        document_mask_mode="block_diagonal_reset_rope",
        init_std=0.02,
        init_seed=7,
        mup_spec=None,
    )
    return replace(base, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


QWEN2 = {"gguf_wire_profile": "qwen2", "qkv_bias": True}
QWEN3 = {"gguf_wire_profile": "qwen3", "norm_kind": "qknorm_rmsnorm", "head_dim": 128}
MODERN_BERT = {
    "gguf_wire_profile": "modern-bert",
    "architecture_kind": "encoder_only",
    "norm_kind": "layernorm",
    "activation": "geglu",
    "n_kv_heads": 8,
    "local_attention_window": 128,
    "global_attn_every_n_layers": 3,
    "local_rope_theta": 10000.0,
    "rope_theta": 160000.0,
    "document_mask_mode": "unpacked_single_sequence",
}
CUSTOM_DYNAMICS = {
    "gguf_wire_profile": "onnx_only_custom",
    "family": "dynamics",
    "architecture_kind": "dynamics_transformer",
    "positional_encoding": "continuous_time_fourier",
    "rope_theta": None,
    "activation": "relu2",
    "norm_kind": "layernorm",
    "linear_bias": True,
    "head_dim": 40,
}


def yarn(**changes: object) -> YarnRopeScalingSpec:
    """Return a valid YaRN scaling from 1024 positions with ``changes`` applied."""
    base = YarnRopeScalingSpec("yarn-rope-scaling.v1", 4.0, 1024, None, 1.0, 32.0, 1.0)
    return replace(base, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


def mup(**changes: object) -> MaximalUpdateParametrisationSpec:
    """Return a valid width-transfer spec to ``d_model`` 512 with ``changes`` applied."""
    base = MaximalUpdateParametrisationSpec(
        "mup-spec.v1",
        128,
        512,
        1.0,
        True,
        "standard_inv_sqrt_d",
        -0.5,
        0.0,
        -1.0,
        0.0,
        -1.0,
        digest_of("receipt"),
    )
    return replace(base, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


def receipt(**changes: object) -> MupValidationReceipt:
    """Return a valid coordinate-check receipt with ``changes`` applied."""
    base = MupValidationReceipt(
        "mup-validation-receipt.v1",
        (64, 128, 256),
        0.003,
        0.1,
        200,
        digest_of("dedup"),
        UTC,
    )
    return replace(base, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


class TestProfiles:
    """Each wire profile accepts what its converter expresses and nothing else."""

    @pytest.mark.parametrize(
        "changes",
        [{}, QWEN2, QWEN3, MODERN_BERT, CUSTOM_DYNAMICS],
        ids=["llama", "qwen2", "qwen3", "modern-bert", "custom"],
    )
    def test_each_profile_has_a_valid_spec(self, changes: dict[str, object]) -> None:
        """A spec within each profile's locks validates on every funding kind."""
        item = spec(**changes)
        for funding in sorted(POOL_FUNDING_KINDS):
            assert (
                validate_transformer_architecture_spec(item, funding_kind=funding)
                == item
            )

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({**QWEN2, "qkv_bias": False}, "qkv_bias"),
            ({**QWEN3, "norm_kind": "rmsnorm"}, "norm_kind"),
            ({**MODERN_BERT, "qkv_bias": True}, "qkv_bias"),
            (
                {
                    **MODERN_BERT,
                    "local_attention_window": None,
                    "local_rope_theta": None,
                    "global_attn_every_n_layers": None,
                },
                "local_attention_window",
            ),
            ({"qkv_bias": True}, "qkv_bias"),
            ({"linear_bias": True}, "linear_bias"),
            ({"norm_kind": "layernorm"}, "norm_kind"),
            ({"activation": "geglu"}, "activation"),
            ({"architecture_kind": "encoder_only"}, "architecture_kind"),
            ({"local_attention_window": 256}, "local_attention_window"),
            ({"head_dim": 32}, "head_dim \\* n_heads"),
            ({**QWEN2, "head_dim": 32}, "head_dim \\* n_heads"),
            ({**QWEN3, "head_dim": 127}, "even head_dim"),
            ({**QWEN3, "local_attention_window": 64}, "local_attention_window"),
            ({**MODERN_BERT, "activation": "relu2"}, "geglu or swiglu"),
            ({**MODERN_BERT, "n_kv_heads": 4}, "n_kv_heads == n_heads"),
            ({**MODERN_BERT, "head_dim": 32}, "head_dim \\* n_heads"),
            (
                {
                    **MODERN_BERT,
                    "positional_encoding": "alibi",
                    "rope_theta": None,
                    "local_rope_theta": None,
                },
                "positional_encoding",
            ),
            (
                {**MODERN_BERT, "architecture_kind": "causal_decoder"},
                "architecture_kind",
            ),
            (
                {**MODERN_BERT, "document_mask_mode": "block_diagonal_reset_rope"},
                "document_mask_mode",
            ),
        ],
        ids=[
            "M-B2-ARCH-1-qwen2-bias",
            "M-B2-ARCH-2-qwen3-norm",
            "M-B2-ARCH-3-bert-bias",
            "M-B2-ARCH-3-bert-window",
            "llama-qkv",
            "llama-linear",
            "llama-norm",
            "llama-act",
            "llama-kind",
            "llama-window",
            "llama-width",
            "qwen2-width",
            "qwen3-odd",
            "qwen3-window",
            "bert-act",
            "bert-kv",
            "bert-width",
            "bert-position",
            "bert-kind",
            "bert-mask",
        ],
    )
    def test_profile_locks(self, changes: dict[str, object], fragment: str) -> None:
        """A field outside its profile's lock is refused by name."""
        with pytest.raises(ArchitectureSpecError, match=fragment):
            check_transformer_architecture_shape(spec(**changes))

    def test_qwen3_width_may_differ_from_d_model(self) -> None:
        """Only qwen3 and custom profiles allow ``head_dim * n_heads != d_model``."""
        item = spec(**QWEN3)
        assert item.head_dim * item.n_heads != item.d_model
        check_transformer_architecture_shape(item)


class TestNumericRules:
    """Shape, seed, rotary and YaRN rules that hold for every profile."""

    @pytest.mark.parametrize(
        "changes",
        [
            {"d_model": 0},
            {"n_layers": 0},
            {"n_heads": 0},
            {"n_kv_heads": 0},
            {"head_dim": 0},
            {"ffn_hidden_dim": 0},
            {"max_context_length": 0},
            {"vocab_or_channel_dim": 0},
            {"output_dim": 0},
            {"n_layers": 8.0},
            {"n_kv_heads": 3},
            {"norm_eps": 0.0},
            {"norm_eps": 1},
            {"init_std": -0.02},
            {"init_std": float("nan")},
            {"init_seed": -1},
            {"init_seed": 2**63},
            {"init_seed": True},
            {"linear_bias": 0},
            {"tie_word_embeddings": None},
            {"schema": "transformer-architecture-spec.v2"},
            {"family": "spiking"},
            {"architecture_kind": "mixture"},
            {"gguf_wire_profile": "gemma"},
            {"activation": "gelu"},
            {"norm_kind": "batchnorm"},
            {"positional_encoding": "learned"},
            {"document_mask_mode": "packed"},
            {"local_attention_window": 0},
            {"global_attn_every_n_layers": 2},
        ],
    )
    def test_numeric_and_choice_refusals(self, changes: dict[str, object]) -> None:
        """Each malformed field is refused."""
        with pytest.raises(ArchitectureSpecError):
            check_transformer_architecture_shape(
                spec(**{"gguf_wire_profile": "onnx_only_custom", **changes})
            )

    def test_the_largest_seed_passes(self) -> None:
        """Seeds lie in ``[0, 2**63)``."""
        check_transformer_architecture_shape(spec(init_seed=2**63 - 1))
        check_transformer_architecture_shape(spec(init_seed=0))

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"rope_theta": None}, "rope_theta"),
            ({"rope_theta": 0.0}, "rope_theta"),
            ({"positional_encoding": "alibi"}, "only for rotary"),
            ({"local_rope_theta": 10000.0}, "needs a local_attention_window"),
            (
                {"local_attention_window": 64, "local_rope_theta": 0.0},
                "local_rope_theta",
            ),
            ({"rope_scaling": yarn()}, "only for yarn_rope"),
            ({"positional_encoding": "yarn_rope"}, "needs rope_scaling"),
        ],
    )
    def test_rotary_rules(self, changes: dict[str, object], fragment: str) -> None:
        """Rotary bases and YaRN scaling match the position encoding."""
        with pytest.raises(ArchitectureSpecError, match=fragment):
            check_transformer_architecture_shape(
                spec(**{"gguf_wire_profile": "onnx_only_custom", **changes})
            )

    def test_yarn_extends_a_rotary_model(self) -> None:
        """A YaRN spec with its scaling validates; its parts are checked."""
        item = spec(positional_encoding="yarn_rope", rope_scaling=yarn())
        check_transformer_architecture_shape(item)
        assert validate_yarn_rope_scaling(yarn(), max_context_length=4096) == yarn()

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"factor": 1.0}, "factor must exceed 1"),
            ({"factor": 4}, "factor"),
            ({"original_max_position_embeddings": 4096}, "below max_context_length"),
            ({"original_max_position_embeddings": 0}, "original_max"),
            ({"beta_fast": 1.0}, "beta_fast must exceed beta_slow"),
            ({"beta_fast": 0.5}, "beta_fast must exceed beta_slow"),
            ({"beta_slow": 0.0}, "beta_slow"),
            ({"attention_factor": float("inf")}, "attention_factor"),
            ({"extrapolation_factor": 1}, "extrapolation_factor"),
            ({"schema": "yarn.v1"}, "schema"),
        ],
        ids=[
            "factor-one",
            "factor-int",
            "original-full",
            "original-zero",
            "M-B2-ARCH-6-equal",
            "M-B2-ARCH-6-below",
            "slow-zero",
            "attention",
            "extrapolation",
            "schema",
        ],
    )
    def test_yarn_refusals(self, changes: dict[str, object], fragment: str) -> None:
        """Each YaRN rule is refused by name, inside a spec too."""
        with pytest.raises(ArchitectureSpecError, match=fragment):
            validate_yarn_rope_scaling(yarn(**changes), max_context_length=4096)
        with pytest.raises(ArchitectureSpecError, match=fragment):
            check_transformer_architecture_shape(
                spec(positional_encoding="yarn_rope", rope_scaling=yarn(**changes))
            )

    def test_a_scaling_of_another_type_is_refused(self) -> None:
        """YaRN scaling must be its own record."""
        with pytest.raises(ArchitectureSpecError):
            validate_yarn_rope_scaling(receipt(), max_context_length=4096)  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it


class TestWidthTransfer:
    """Maximal-update width transfer and its receipt."""

    @pytest.mark.parametrize("profile", ["llama", "onnx_only_custom"])
    def test_mup_inv_d_on_the_profiles_that_express_it(self, profile: str) -> None:
        """The 1/d logit scale is accepted where the converter has it."""
        item = spec(
            gguf_wire_profile=profile,
            mup_spec=mup(attention_logit_scale_mode="mup_inv_d"),
        )
        check_transformer_architecture_shape(item)

    @pytest.mark.parametrize(
        "changes",
        [QWEN2, QWEN3, MODERN_BERT],
        ids=["qwen2", "M-B2-ARCH-4-qwen3", "bert"],
    )
    def test_mup_inv_d_elsewhere_is_refused(self, changes: dict[str, object]) -> None:
        """M-B2-ARCH-4: other profiles cannot express the 1/d logit scale."""
        item = spec(**changes, mup_spec=mup(attention_logit_scale_mode="mup_inv_d"))
        with pytest.raises(ArchitectureSpecError, match="cannot express mup_inv_d"):
            check_transformer_architecture_shape(item)
        check_transformer_architecture_shape(replace(item, mup_spec=mup()))

    @pytest.mark.parametrize("funding", sorted(POOL_FUNDING_KINDS - {LOCAL}))
    def test_m_b2_arch_5_paid_width_transfer_needs_its_receipt(
        self, funding: str
    ) -> None:
        """M-B2-ARCH-5: outside owned hardware, width transfer needs a receipt."""
        item = spec(mup_spec=mup(proxy_validation_receipt_digest=None))
        assert validate_transformer_architecture_spec(item, funding_kind=LOCAL) == item
        with pytest.raises(ArchitectureSpecError, match="validation receipt"):
            validate_transformer_architecture_spec(item, funding_kind=funding)
        with_receipt = spec(mup_spec=mup())
        validate_transformer_architecture_spec(with_receipt, funding_kind=funding)

    def test_an_unknown_funding_kind_is_refused(self) -> None:
        """The funding kind is one of the four pool kinds."""
        with pytest.raises(ArchitectureSpecError, match="funding_kind"):
            validate_transformer_architecture_spec(spec(), funding_kind="donated")

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"target_d_model": 256}, ArchitectureSpecError),
            ({"base_d_model": 0}, ArchitectureSpecError),
            ({"input_scale": 0.0}, ArchitectureSpecError),
            ({"output_zero_init": 1}, ArchitectureSpecError),
            ({"attention_logit_scale_mode": "inv_d"}, ArchitectureSpecError),
            ({"readout_lr_scale_power": float("nan")}, ArchitectureSpecError),
            ({"hidden_init_std_scale_power": -1}, ArchitectureSpecError),
            ({"schema": "mup.v1"}, ArchitectureSpecError),
            ({"proxy_validation_receipt_digest": "x"}, MalformedDigestError),
        ],
    )
    def test_mup_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Each malformed width-transfer field is refused, inside a spec too."""
        with pytest.raises(error):
            validate_mup_spec(mup(**changes), d_model=512)
        with pytest.raises(error):
            check_transformer_architecture_shape(spec(mup_spec=mup(**changes)))

    def test_a_mup_of_another_type_is_refused(self) -> None:
        """Width transfer must be its own record."""
        with pytest.raises(ArchitectureSpecError):
            validate_mup_spec(yarn(), d_model=512)  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"proxy_widths_tested": (128,)}, ArchitectureSpecError),
            ({"proxy_widths_tested": (128, 128)}, ArchitectureSpecError),
            ({"proxy_widths_tested": (256, 128)}, ArchitectureSpecError),
            ({"proxy_widths_tested": [64, 128]}, ArchitectureSpecError),
            ({"proxy_widths_tested": (0, 128)}, ArchitectureSpecError),
            ({"optimal_base_lr": 0.0}, ArchitectureSpecError),
            ({"max_optimal_lr_drift_ratio": -0.1}, ArchitectureSpecError),
            ({"coordinate_check_steps": 0}, ArchitectureSpecError),
            ({"verified_at_utc": "yesterday"}, ArchitectureSpecError),
            ({"schema": "receipt.v1"}, ArchitectureSpecError),
            ({"dedup_report_digest": "x"}, MalformedDigestError),
        ],
    )
    def test_receipt_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """A receipt tests at least two increasing widths; each field is checked."""
        with pytest.raises(error):
            validate_mup_validation_receipt(receipt(**changes))

    def test_a_receipt_validates_and_another_type_is_refused(self) -> None:
        """A valid receipt passes, with a zero drift too; other records do not."""
        assert validate_mup_validation_receipt(receipt()) == receipt()
        validate_mup_validation_receipt(receipt(max_optimal_lr_drift_ratio=0.0))
        with pytest.raises(ArchitectureSpecError):
            validate_mup_validation_receipt(yarn())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it


class TestCodec:
    """Architecture records decode through the codec with their checks."""

    @pytest.mark.parametrize(
        "record",
        [
            spec(positional_encoding="yarn_rope", rope_scaling=yarn(), mup_spec=mup()),
            spec(**MODERN_BERT),
            receipt(),
        ],
        ids=["yarn-mup", "bert", "receipt"],
    )
    def test_round_trip(self, record: object) -> None:
        """A spec with nested parts, and a receipt, decode to equal records."""
        raw = canonical_json_bytes(record)
        assert from_canonical_json_bytes(type(record), raw) == record

    def test_the_codec_runs_the_shape_check_but_not_the_funding_rule(self) -> None:
        """The funding rule needs the pool, so it stays with the full validator."""
        unreceipted = spec(mup_spec=mup(proxy_validation_receipt_digest=None))
        raw = canonical_json_bytes(unreceipted)
        decoded = from_canonical_json_bytes(TransformerArchitectureSpec, raw)
        with pytest.raises(ArchitectureSpecError, match="validation receipt"):
            validate_transformer_architecture_spec(decoded, funding_kind=CLOUD)
        with pytest.raises(ArchitectureSpecError, match="qkv_bias"):
            from_canonical_json_bytes(
                TransformerArchitectureSpec, canonical_json_bytes(spec(qkv_bias=True))
            )

    def test_a_spec_of_another_type_is_refused(self) -> None:
        """The shape check checks its own record type."""
        with pytest.raises(ArchitectureSpecError):
            check_transformer_architecture_shape(receipt())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it
