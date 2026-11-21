# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Adapter Merge Tests

"""M-J4-3: merges share one tokeniser and a set identity; the result is checked."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from director_ai.model_lifecycle import (
    AdapterMergeError,
    LanguageFamilyContract,
    ModelArtefactEnvelope,
    merge_identity,
    record_digest,
    validate_adapter_merge,
    validate_merge_result,
)
from tests.model_lifecycle_support import digest_of, edge, envelope
from tests.test_model_lifecycle_families import language_contract

CONTRACT = language_contract(architecture_spec_digest=digest_of("arch"))


def parent(
    label: str, contract: LanguageFamilyContract = CONTRACT, **changes: Any
) -> ModelArtefactEnvelope:
    """Return a language artefact named ``label`` bound to ``contract``."""
    content = digest_of(label)
    base = envelope(
        content_digest=content,
        lineage_edges=(edge(digest_of("base"), content),),
        family_contract_digest=record_digest(contract),
    )
    return replace(base, **changes)


PARENTS = (parent("a"), parent("b"))
CONTRACTS = (CONTRACT, CONTRACT)
IDENTITY = (digest_of("tokeniser"), digest_of("arch"), digest_of("embedding"), False)


class TestParents:
    """The parents: language, bound, distinct, one kind, one identity."""

    def test_a_compatible_merge_returns_its_identity(self) -> None:
        """Two bound adapters with one tokeniser and a set identity pass."""
        assert merge_identity(PARENTS, CONTRACTS) == IDENTITY
        weights_ok = validate_adapter_merge(
            PARENTS, CONTRACTS, method="linear", weights=(0.5, 0.5)
        )
        assert weights_ok == IDENTITY

    def test_m_j4_3_a_cross_tokeniser_merge_is_refused(self) -> None:
        """Parents with different tokenisers are never merged."""
        other = language_contract(
            architecture_spec_digest=digest_of("arch"),
            tokeniser_envelope_digest=digest_of("tok-2"),
        )
        with pytest.raises(AdapterMergeError, match="M-J4-3"):
            merge_identity((PARENTS[0], parent("b", other)), (CONTRACT, other))

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"architecture_spec_digest": None}, "architecture and embedding identity"),
            ({"embedding_space_digest": None}, "architecture and embedding identity"),
        ],
        ids=["unset-architecture", "unset-embedding"],
    )
    def test_an_unset_identity_is_refused_even_when_equal(
        self, changes: dict[str, object], fragment: str
    ) -> None:
        """Two unset identities prove nothing."""
        unset = replace(CONTRACT, **changes)  # type: ignore[arg-type]
        with pytest.raises(AdapterMergeError, match=fragment):
            merge_identity((parent("a", unset), parent("b", unset)), (unset, unset))

    @pytest.mark.parametrize(
        "changes",
        [
            {"architecture_spec_digest": digest_of("arch-2")},
            {"embedding_space_digest": digest_of("emb-2")},
            {"tie_word_embeddings": True},
        ],
        ids=["architecture", "embedding", "tie"],
    )
    def test_a_different_identity_is_refused(self, changes: dict[str, object]) -> None:
        """The architecture, embedding space and tie are equal across parents."""
        other = replace(CONTRACT, **changes)  # type: ignore[arg-type]
        with pytest.raises(AdapterMergeError, match="differ in architecture"):
            merge_identity((PARENTS[0], parent("b", other)), (CONTRACT, other))

    @pytest.mark.parametrize(
        ("parents", "contracts", "fragment"),
        [
            ((parent("a"),), (CONTRACT,), "at least two"),
            (PARENTS, (CONTRACT,), "one family contract per parent"),
            ((parent("a"), parent("a")), CONTRACTS, "parent content digests"),
            (
                (parent("a"), parent("b", artefact_kind="safetensors_weights")),
                CONTRACTS,
                "all adapters or all full",
            ),
            (
                (
                    parent("a", artefact_kind="word_patch"),
                    parent("b", artefact_kind="word_patch"),
                ),
                CONTRACTS,
                "all adapters or all full",
            ),
            (
                (parent("a"), parent("b", family_contract_digest=digest_of("x"))),
                CONTRACTS,
                "not the one it names",
            ),
        ],
        ids=[
            "one-parent",
            "contract-count",
            "duplicate",
            "mixed-kinds",
            "word-patch",
            "unbound",
        ],
    )
    def test_parent_refusals(
        self, parents: tuple[Any, ...], contracts: tuple[Any, ...], fragment: str
    ) -> None:
        """Each parent rule is refused by name."""
        with pytest.raises(AdapterMergeError, match=fragment):
            merge_identity(parents, contracts)

    def test_full_weights_merge_too(self) -> None:
        """Two full-weight artefacts are one mergeable kind."""
        weights = (
            parent("a", artefact_kind="safetensors_weights"),
            parent("b", artefact_kind="safetensors_weights"),
        )
        assert merge_identity(weights, CONTRACTS) == IDENTITY

    def test_only_language_artefacts_merge(self) -> None:
        """A dynamics artefact is not merged by this rule."""
        dynamics = parent(
            "b", family="dynamics", family_contract_schema="dynamics-family-contract.v1"
        )
        with pytest.raises(AdapterMergeError, match="only language"):
            merge_identity((PARENTS[0], dynamics), CONTRACTS)


class TestWeights:
    """One finite positive weight per parent; linear weights sum exactly to 1."""

    def test_linear_weights_sum_as_declared_decimals(self) -> None:
        """0.1 + 0.2 + 0.7 is exactly 1; 0.5 + 0.5000000000000001 is not, though floats round it to 1."""
        three = (parent("a"), parent("b"), parent("c"))
        assert validate_adapter_merge(
            three, (CONTRACT,) * 3, method="linear", weights=(0.1, 0.2, 0.7)
        )
        assert sum((0.5, 0.5000000000000001)) == 1.0
        with pytest.raises(AdapterMergeError, match="sum exactly to 1"):
            validate_adapter_merge(
                PARENTS, CONTRACTS, method="linear", weights=(0.5, 0.5000000000000001)
            )

    def test_ties_weights_have_no_sum_rule(self) -> None:
        """TIES and DARE-TIES weights are scalings."""
        for method in ("ties", "dare_ties"):
            assert validate_adapter_merge(
                PARENTS, CONTRACTS, method=method, weights=(1.0, 2.0)
            )

    @pytest.mark.parametrize(
        ("method", "weights", "fragment"),
        [
            ("linear", (0.5, 0.4), "sum exactly to 1"),
            ("linear", (1.0, 1e-60), "more precision"),
            ("ties", (1.0,), "one weight per parent"),
            ("ties", (1.0, 0.0), "above zero"),
            ("ties", (1.0, -1.0), "above zero"),
            ("ties", (1.0, float("nan")), "weight"),
            ("slerp", (0.5, 0.5), "method"),
        ],
    )
    def test_weight_refusals(
        self, method: str, weights: tuple[float, ...], fragment: str
    ) -> None:
        """Each weight rule is refused by name."""
        with pytest.raises(AdapterMergeError, match=fragment):
            validate_adapter_merge(PARENTS, CONTRACTS, method=method, weights=weights)


class TestResult:
    """The merged artefact is new and claims the parents' identity."""

    def merged(
        self, contract: LanguageFamilyContract = CONTRACT, **changes: Any
    ) -> ModelArtefactEnvelope:
        """Return a merged artefact bound to ``contract``."""
        return parent("merged", contract, **changes)

    def test_a_faithful_result_passes(self) -> None:
        """A new artefact with the parents' identity passes."""
        item = self.merged()
        assert (
            validate_merge_result(
                item, CONTRACT, parents=PARENTS, parent_contracts=CONTRACTS
            )
            == item
        )

    @pytest.mark.parametrize(
        "changes",
        [
            {"tokeniser_envelope_digest": digest_of("tok-2")},
            {"architecture_spec_digest": digest_of("arch-2")},
            {"embedding_space_digest": digest_of("emb-2")},
            {"tie_word_embeddings": True},
        ],
        ids=["tokeniser", "architecture", "embedding", "tie"],
    )
    def test_a_result_claiming_another_identity_is_refused(
        self, changes: dict[str, object]
    ) -> None:
        """The result may not claim another tokeniser, architecture, embedding or tie."""
        other = replace(CONTRACT, **changes)  # type: ignore[arg-type]
        with pytest.raises(AdapterMergeError, match="claims another"):
            validate_merge_result(
                self.merged(other), other, parents=PARENTS, parent_contracts=CONTRACTS
            )

    def test_a_result_that_is_a_parent_is_refused(self) -> None:
        """The merge produces a new artefact."""
        with pytest.raises(AdapterMergeError, match="new artefact"):
            validate_merge_result(
                PARENTS[0], CONTRACT, parents=PARENTS, parent_contracts=CONTRACTS
            )

    def test_the_result_is_bound_and_language(self) -> None:
        """The result names its own contract and is a language artefact."""
        with pytest.raises(AdapterMergeError, match="not the one it names"):
            validate_merge_result(
                self.merged(family_contract_digest=digest_of("x")),
                CONTRACT,
                parents=PARENTS,
                parent_contracts=CONTRACTS,
            )
        dynamics = self.merged(
            family="dynamics", family_contract_schema="dynamics-family-contract.v1"
        )
        with pytest.raises(AdapterMergeError, match="only language"):
            validate_merge_result(
                dynamics, CONTRACT, parents=PARENTS, parent_contracts=CONTRACTS
            )
        with pytest.raises(AdapterMergeError, match="M-J4-3"):
            other = language_contract(
                architecture_spec_digest=digest_of("arch"),
                tokeniser_envelope_digest=digest_of("t"),
            )
            validate_merge_result(
                self.merged(),
                CONTRACT,
                parents=(PARENTS[0], parent("b", other)),
                parent_contracts=(CONTRACT, other),
            )
