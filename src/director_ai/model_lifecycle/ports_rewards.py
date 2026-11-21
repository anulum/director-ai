# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Verifiable-Reward Ports

"""Ports of verifiable-reward training: verifiers, curricula and self-play.

A verifier runs a networkless, deterministic check on a group of rollouts.
A curriculum scheduler samples prompts whose empirical pass rate lies in the
zone of proximal development. A self-play synthesizer admits a synthetic
problem only with a constructive witness and a sufficient mutant kill rate.
Adapters outside the kernel implement them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from .rlvr_records import VerifiableRewardOutcome


@runtime_checkable
class VerifiableRewardPort(Protocol):
    """Runs networkless deterministic code, math, proof, or SBY RTL verifiers."""

    def verify_rollout_group(
        self,
        *,
        prompt_id: str,
        problem_bundle_digest: str,
        candidate_responses: Sequence[str],
        verifier_family: Literal[
            "code_execution",
            "symbolic_math",
            "lean_or_smt_proof",
            "hardware_rtl_sby",
        ],
        max_wall_ms_per_rollout: int,
        scratch_dir: Path,
    ) -> tuple[VerifiableRewardOutcome, ...]:
        """Verify each response of one prompt's rollout group.

        Parameters
        ----------
        prompt_id : str
            The prompt.
        problem_bundle_digest : str
            SHA-256 of the problem bundle.
        candidate_responses : Sequence[str]
            The group's responses.
        verifier_family : str
            The verifier family.
        max_wall_ms_per_rollout : int
            The wall-clock cap per rollout.
        scratch_dir : Path
            A scratch directory for the verifier.

        Returns
        -------
        tuple[VerifiableRewardOutcome, ...]
            One outcome per response.
        """
        ...


@runtime_checkable
class ZpdCurriculumSchedulerPort(Protocol):
    """Samples prompts in the empirical pass-rate zone of proximal development."""

    def sample_curriculum_batch(
        self,
        *,
        archive_digest: str,
        batch_size: int,
        group_size_g: int,
        empirical_pass_rates: Mapping[str, float],
        seed: int,
    ) -> tuple[str, ...]:
        """Sample a batch of prompt ids by their empirical pass rates.

        Parameters
        ----------
        archive_digest : str
            SHA-256 of the prompt archive.
        batch_size : int
            Prompts per batch.
        group_size_g : int
            Rollouts per prompt.
        empirical_pass_rates : Mapping[str, float]
            Prompt id to its observed pass rate.
        seed : int
            The sampling seed.

        Returns
        -------
        tuple[str, ...]
            The sampled prompt ids.
        """
        ...


@runtime_checkable
class SelfPlayProblemSynthesizerPort(Protocol):
    """Admits synthetic problems only with constructive witnesses and mutant kills."""

    def verify_and_admit_synthetic_problem(
        self,
        *,
        template_family_id: str,
        problem_spec_bytes: bytes,
        constructive_witness_bytes: bytes,
        mutant_kill_rate_floor: float = 0.80,
    ) -> tuple[bool, str]:
        """Admit a synthetic problem, or refuse it, with the reason.

        Parameters
        ----------
        template_family_id : str
            The problem's template family.
        problem_spec_bytes : bytes
            The problem specification.
        constructive_witness_bytes : bytes
            A witness that the problem is solvable.
        mutant_kill_rate_floor : float
            The fewest mutants the problem's checks must kill, 0.80 by default.

        Returns
        -------
        tuple[bool, str]
            Whether the problem is admitted, and why.
        """
        ...
