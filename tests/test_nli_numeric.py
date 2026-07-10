# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI numeric mapping contracts

"""Contract tests for the NLI numeric mapping helpers.

``director_ai.core.scoring._nli_numeric`` owns the logits-to-score maths:
softmax, divergence and confidence mapping, label-index resolution, and
the shared float-list reducers. These tests pin the re-export surface on
``nli.py`` and the mapping semantics on both class layouts.
"""

from __future__ import annotations

import numpy as np
import pytest

import director_ai.core.scoring._nli_numeric as nli_numeric
import director_ai.core.scoring.nli as nli_mod


class TestReExportSurface:
    def test_nli_module_re_exports_the_numeric_helpers(self):
        for name in nli_numeric.__all__:
            assert getattr(nli_mod, name) is getattr(nli_numeric, name)


class TestProbabilityMappings:
    def test_softmax_rows_are_distributions_and_match_reference(self):
        x = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, -3.0]])
        out = nli_numeric._softmax_np(x)
        np.testing.assert_allclose(out.sum(axis=1), [1.0, 1.0], rtol=1e-12)
        ref = np.exp(x - x.max(axis=1, keepdims=True))
        ref = ref / ref.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(out, ref, rtol=1e-12)

    def test_divergence_two_class_is_one_minus_supported(self):
        probs = np.array([[0.2, 0.8], [0.9, 0.1]])
        assert nli_numeric._probs_to_divergence(probs) == pytest.approx([0.2, 0.9])

    def test_divergence_three_class_uses_label_indices(self):
        probs = np.array([[0.6, 0.3, 0.1]])
        default = nli_numeric._probs_to_divergence(probs)
        assert default == pytest.approx([0.1 + 0.5 * 0.3])
        swapped = nli_numeric._probs_to_divergence(probs, label_indices=(0, 1))
        assert swapped == pytest.approx([0.6 + 0.5 * 0.3])

    def test_confidence_is_high_for_one_hot_and_low_for_uniform(self):
        one_hot = np.array([[1.0, 0.0, 0.0]])
        uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
        high = nli_numeric._probs_to_confidence(one_hot)[0]
        low = nli_numeric._probs_to_confidence(uniform)[0]
        assert high == pytest.approx(1.0, abs=1e-6)
        assert low == pytest.approx(0.0, abs=1e-6)


class TestLabelIndexResolution:
    def test_missing_config_falls_back_to_default_layout(self):
        assert nli_numeric._resolve_label_indices(object()) == (2, 1)

    def test_custom_id2label_is_honoured(self):
        class Config:
            id2label = {0: "CONTRADICTION", 1: "entailment", 2: "Neutral"}

        class Model:
            config = Config()

        assert nli_numeric._resolve_label_indices(Model()) == (0, 2)


class TestFloatReducers:
    def test_reducers_handle_empty_inputs(self):
        assert nli_numeric._sum_float_list([]) == 0.0
        assert nli_numeric._mean_float([]) == 0.0
        assert nli_numeric._count_below_threshold([], 0.5) == 0
        assert nli_numeric._weighted_sum_float([], []) == 0.0

    def test_reducers_compute_expected_values(self):
        values = [0.2, 0.4, 0.9]
        assert nli_numeric._sum_float_list(values) == pytest.approx(1.5)
        assert nli_numeric._mean_float(values) == pytest.approx(0.5)
        assert nli_numeric._count_below_threshold(values, 0.5) == 2
        assert nli_numeric._weighted_sum_float(
            values, [1.0, 0.5, 2.0]
        ) == pytest.approx(0.2 + 0.2 + 1.8)
