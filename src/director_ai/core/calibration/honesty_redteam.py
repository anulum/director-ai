# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — adversarial strip-resistance battery for the honesty seal

"""Red-team the verifiable-honesty seal — a survivor is a release blocker (WS-1).

Signing a claim is necessary but not sufficient: an attacker does not strip the
badge, they show a *plausible-wrong* one. DIRECTOR-AI owns the adversarial layer
of the SCPN-STUDIO verifiable-honesty contract — given the published
``scpn_studio_platform.seal`` (``seal`` / ``verify`` over an ``Ed25519`` signed
``HonestyEnvelope`` yielding ``Verdict.{VERIFIED,STRIPPED,FORGED,UNGRADED}``),
this battery mounts the strip / forge / tamper / replay / near-threshold attacks
and asserts every one is caught (verdict ≠ ``VERIFIED``). A **survivor** — an
attack that still verifies — is a **release blocker**.

The headline attack is the decision-value-separation case the fleet converged on
(SPO + DIRECTOR-AI): a render that keeps the *verdict* honest but inflates the
*margin* (``validated`` with a rendered ``margin=2σ`` over a true ``0.1σ``). The
defence — ratified — is that the margin lives **inside the signed unit** and is
recomputed, so inflating the rendered margin breaks the signature over
``canonical(unit)`` → ``FORGED``. This battery proves the seal enforces it.

The seal SDK is an optional dependency (the ``honesty`` extra; it needs
Python ≥ 3.12 and ``cryptography``). When it is absent the battery raises a clear
error rather than silently passing — a security gate must never no-op.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

try:
    from scpn_studio_platform.seal import (
        Ed25519Signer,
        Keyring,
        Verdict,
        seal,
        verify,
    )

    _SEAL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the honesty extra
    _SEAL_AVAILABLE = False

__all__ = [
    "AttackOutcome",
    "BatteryReport",
    "HonestyRedteamError",
    "assert_zero_survivors",
    "run_strip_resistance_battery",
]

#: Hint shown when the optional seal SDK is missing.
_EXTRA_HINT = (
    "the honesty seal is unavailable; install the 'honesty' extra "
    "(scpn-studio-platform>=0.10, Python>=3.12)"
)

#: A regrade re-derives the grade from the unit (the pure-function grade the
#: envelope is verified against).
Regrade = Callable[[Mapping[str, Any]], str]


class HonestyRedteamError(RuntimeError):
    """Raised when the strip-resistance battery finds a surviving attack."""


@dataclass(frozen=True)
class AttackOutcome:
    """The verdict an attack produced, and whether it survived (still verified)."""

    attack: str
    verdict: str
    survived: bool

    def to_dict(self) -> dict[str, object]:
        """Serialise for the published battery report."""
        return {
            "attack": self.attack,
            "verdict": self.verdict,
            "survived": self.survived,
        }


@dataclass(frozen=True)
class BatteryReport:
    """The whole battery: every attack's outcome and the survivor verdict.

    ``all_blocked`` is the release-gate boolean — true only when every attack was
    caught (no survivor). The measurable is ``survivors`` (target: empty).
    """

    outcomes: tuple[AttackOutcome, ...]
    survivors: tuple[str, ...]
    all_blocked: bool

    def to_dict(self) -> dict[str, object]:
        """Serialise the battery report for publication + the CI gate."""
        return {
            "all_blocked": self.all_blocked,
            "survivors": list(self.survivors),
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


def run_strip_resistance_battery(
    *,
    unit: Mapping[str, Any],
    true_grade: str,
    regrade: Regrade,
    inflated_grade: str = "validated",
    margin_field: str = "margin",
) -> BatteryReport:
    """Mount the adversarial battery against a freshly sealed unit.

    Seals ``unit`` (graded ``true_grade`` by ``regrade``) with a fresh Ed25519
    key, then runs each attack and verifies it. Returns a :class:`BatteryReport`;
    a non-empty ``survivors`` means the seal failed to catch an attack — a release
    blocker. ``inflated_grade`` is the over-claim a strip attempts;
    ``margin_field`` is the signed value the margin-forge attack inflates.

    Raises :class:`HonestyRedteamError` if the seal SDK (the ``honesty`` extra) is
    not installed — a security gate must be loud, never a silent pass.
    """
    if not _SEAL_AVAILABLE:
        raise HonestyRedteamError(_EXTRA_HINT)
    if inflated_grade == true_grade:
        raise ValueError(
            "inflated_grade must over-claim true_grade; equal grades make the "
            "strip attack vacuous"
        )

    signer = Ed25519Signer.generate("redteam-key")
    keyring = Keyring()
    keyring.add(signer.key_id, signer.verifier())
    envelope = seal(
        dict(unit),
        signer=signer,
        grader={"id": "director-ai/redteam@1"},
        verifiability_mode="recompute",
        exactness_class="bit-exact",
    ).to_dict()

    # A second, independently-valid envelope of the same unit (regrades to
    # true_grade) for the replay attack: a genuinely-signed envelope still cannot
    # be rendered at a grade above what its unit recomputes to.
    foreign = seal(
        dict(unit),
        signer=signer,
        grader={"id": "director-ai/redteam@1"},
        verifiability_mode="recompute",
        exactness_class="bit-exact",
    ).to_dict()

    inflated_unit = copy.deepcopy(dict(unit))
    if margin_field in inflated_unit:
        inflated_unit[margin_field] = float(inflated_unit[margin_field]) * 20.0
    tampered_margin = copy.deepcopy(envelope)
    tampered_margin["unit"] = inflated_unit

    tampered_unit = copy.deepcopy(envelope)
    tampered_unit["unit"] = {**dict(unit), "claim": "a swapped claim"}

    # Each attack -> (envelope_or_none, rendered_grade, keyring) to verify.
    attacks: tuple[tuple[str, Any, str | None, Any], ...] = (
        # Strip: render a higher grade than the unit recomputes to.
        ("strip-overclaim", envelope, inflated_grade, keyring),
        # Tamper: mutate the signed unit -> digest/signature mismatch.
        ("tamper-unit", tampered_unit, true_grade, keyring),
        # Near-threshold margin forge (decision-value separation): inflate the
        # signed margin while keeping the verdict -> breaks sig over canonical(unit).
        ("near-threshold-margin-forge", tampered_margin, true_grade, keyring),
        # Replay: a different (lower-graded) unit's valid envelope rendered AS the
        # inflated grade — a genuinely-signed envelope cannot launder a higher claim.
        ("replay-foreign-unit", foreign, inflated_grade, keyring),
        # Unknown key: the signer is not in the keyring.
        ("unknown-key", envelope, true_grade, Keyring()),
        # Stripped entirely: no envelope at all.
        ("missing-envelope", None, true_grade, keyring),
    )

    outcomes: list[AttackOutcome] = []
    for name, env, rendered, kr in attacks:
        verdict = verify(env, rendered, keyring=kr, regrade=regrade)
        verdict_value = str(getattr(verdict, "value", verdict))
        survived = verdict == Verdict.VERIFIED
        outcomes.append(AttackOutcome(name, verdict_value, survived))

    survivors = tuple(o.attack for o in outcomes if o.survived)
    return BatteryReport(
        outcomes=tuple(outcomes),
        survivors=survivors,
        all_blocked=not survivors,
    )


def assert_zero_survivors(report: BatteryReport) -> None:
    """Raise if any attack survived — the release-blocker enforcement.

    A surviving attack means the verifiable-honesty seal admitted a stripped,
    forged, or replayed render as ``VERIFIED``; the portal must not ship.
    """
    if report.survivors:
        raise HonestyRedteamError(
            f"strip-resistance battery has survivors (release blocker): "
            f"{', '.join(report.survivors)}"
        )
