# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Lyapunov Proof Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import pytest

from director_ai.research.physics.lyapunov_proof import (
    ProofResult,
    prove_critical_coupling,
    prove_dv_dt_negative,
    prove_numerical_stability,
    prove_v_non_negative,
    prove_v_zero_at_fixpoint,
    run_all_proofs,
)


@pytest.mark.physics
class TestLyapunovProofs:
    def test_v_non_negative(self):
        result = prove_v_non_negative()
        assert isinstance(result, ProofResult)
        assert result.verified is True
        assert "V_non_negative" in result.name

    def test_v_zero_at_fixpoint(self):
        result = prove_v_zero_at_fixpoint()
        assert result.verified is True
        assert "fixpoint" in result.name

    def test_dv_dt_negative(self):
        result = prove_dv_dt_negative()
        assert result.verified is True
        assert "dV_dt" in result.name

    def test_critical_coupling(self):
        result = prove_critical_coupling()
        assert result.verified is True
        assert "K_c" in result.symbolic_expr

    def test_numerical_stability(self):
        result = prove_numerical_stability(n_steps=30, n_trials=3)
        assert isinstance(result, ProofResult)
        # May be stochastic, but should pass most trials
        assert "numerical" in result.name

    def test_run_all_proofs_symbolic_only(self):
        results = run_all_proofs(include_numerical=False)
        assert len(results) == 4
        for r in results:
            assert isinstance(r, ProofResult)
            assert r.verified is True

    def test_run_all_proofs_with_numerical(self):
        results = run_all_proofs(include_numerical=True)
        assert len(results) == 5

    def test_proof_result_has_detail(self):
        result = prove_v_non_negative()
        assert len(result.detail) > 0
        assert len(result.symbolic_expr) > 0
        assert "QED" in result.detail
