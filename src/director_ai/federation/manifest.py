# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — schema-A studio capability manifest producer

"""Build the Director-AI schema-A studio capability manifest.

This is the federation-gate artifact the SCPN-STUDIO keeper and the Director-AI
Tier-B portal consume — the schema-A manifest (locked contract era ``v1``)
carrying the studio's ``verbs``, ``evidence_types``, federated ``ui_module``, and
a deterministic ``content_digest``. It is the one federation seam Director-AI did
not yet ship; the dedicated portal reads it (never the Python internals) so the
codebase evolves freely behind a stable, versioned contract.

The producer is deliberately self-contained — it emits a schema-A-conformant
mapping directly, with no dependency on the (not-yet-published)
``scpn-studio-platform`` SDK. When that SDK lands, ``build_manifest`` can delegate
to its ``CapabilityManifest`` builder without changing the emitted contract (the
JSON shape is the contract, the SDK is one way to produce it).

Verbs are the **guardrail** domain's, grounded in shipped capabilities: response
scoring (the 5-tier scorer), atomic verification (VerifiedScorer), the
contradiction-driven streaming ``halt``, conformal ``calibrate``, prompt-injection
detection, multi-backend ``benchmark``, sealed evidence ``replay``, and PII
``redact``. ``safety_tier`` is honest per the repo's own boundary: the
response-level scorer and verifier are ``production``-validated, the opt-in
streaming halt is ``research`` (not a sole production gate).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version

__all__ = [
    "CONTRACT_ERA",
    "StudioManifest",
    "UiModule",
    "Verb",
    "build_manifest",
]

#: The locked studio network-contract era this manifest pins (see the v1 contract).
CONTRACT_ERA = "v1"

#: SYNAPSE wire-protocol axis, versioned independently of the contract era.
PROTOCOL_VERSION = "1"

#: Director-AI runs the free local-first profile; tenant identity is ignored.
TRANSPORT_PROFILE = "local-first"

#: SemVer range of the platform SDK this manifest targets once it is published.
PLATFORM_SDK = ">=0.1,<0.2"

#: Studio slug; stable identity across versions.
STUDIO = "director-ai"


@dataclass(frozen=True)
class Verb:
    """One capability verb the studio exposes, with its contract attributes.

    Mirrors the schema-A verb shape (contract §2.3): the ``verb`` name, its
    ``safety_tier`` (research/certified/production), the ``side_effect`` class,
    a ``timing`` class, the evidence schemas it ``produces``, the compute
    ``backends`` it runs on, and an optional ``fidelity`` for compute verbs.
    """

    verb: str
    safety_tier: str
    side_effect: str
    timing_class: str
    produces: tuple[str, ...]
    backends: tuple[str, ...]
    fidelity: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Render the verb as a sorted-key-ready schema-A mapping."""
        payload: dict[str, object] = {
            "verb": self.verb,
            "safety_tier": self.safety_tier,
            "side_effect": self.side_effect,
            "timing": {"class": self.timing_class},
            "produces": list(self.produces),
            "backends": list(self.backends),
        }
        if self.fidelity is not None:
            payload["fidelity"] = self.fidelity
        return payload


@dataclass(frozen=True)
class UiModule:
    """The federated UI panel descriptor (Module Federation 2.x).

    ``remote_entry`` is a host-relative path so the descriptor survives any
    hosting decision (subdomain vs path under the Institute) without a rewrite;
    the hub resolves it against the deployed portal origin.
    """

    remote_entry: str
    exposes: tuple[str, ...]
    federation: str = "module-federation-2"

    def to_dict(self) -> dict[str, object]:
        """Render the ui_module as a schema-A mapping."""
        return {
            "remote_entry": self.remote_entry,
            "exposes": list(self.exposes),
            "federation": self.federation,
        }


# The guardrail domain's verb registry — each grounded in a shipped capability.
_VERBS: tuple[Verb, ...] = (
    Verb(
        verb="score",
        safety_tier="production",
        side_effect="read-only",
        timing_class="interactive",
        produces=("studio.response-score.v1",),
        backends=("python", "rust"),
        fidelity="ml-surrogate",
    ),
    Verb(
        verb="validate",
        safety_tier="production",
        side_effect="read-only",
        timing_class="batch",
        produces=("studio.verification.v1",),
        backends=("python", "rust"),
        fidelity="ml-surrogate",
    ),
    Verb(
        verb="halt",
        safety_tier="research",
        side_effect="read-only",
        timing_class="realtime",
        produces=("studio.streaming-halt.v1",),
        backends=("python", "rust"),
    ),
    Verb(
        verb="calibrate",
        safety_tier="research",
        side_effect="read-only",
        timing_class="batch",
        produces=("studio.calibration.v1",),
        backends=("python",),
    ),
    Verb(
        verb="detect-injection",
        safety_tier="research",
        side_effect="read-only",
        timing_class="interactive",
        produces=("studio.injection-scan.v1",),
        backends=("python",),
    ),
    Verb(
        verb="benchmark",
        safety_tier="research",
        side_effect="simulated",
        timing_class="batch",
        produces=("studio.backend-benchmark.v1",),
        backends=("python", "rust"),
    ),
    Verb(
        verb="replay",
        safety_tier="research",
        side_effect="read-only",
        timing_class="batch",
        produces=("studio.evidence-replay.v1",),
        backends=("python",),
    ),
    Verb(
        verb="redact",
        safety_tier="research",
        side_effect="read-only",
        timing_class="interactive",
        produces=("studio.redaction.v1",),
        backends=("python",),
    ),
)

_UI_MODULE = UiModule(
    remote_entry="/studio/remoteEntry.js",
    exposes=("./DirectorAIStudioPanel",),
)


def _studio_version() -> str:
    """Return the installed distribution version, or a source-tree sentinel.

    The version is an environment-dependent stamp (installed dist vs a source
    checkout); the ``--check`` drift gate excludes it so the check is env-stable,
    while ``content_digest`` covers the verb/evidence contract that must not drift.
    """
    try:
        return version("director-ai")
    except PackageNotFoundError:
        return "0+unknown"


@dataclass(frozen=True)
class StudioManifest:
    """The schema-A capability manifest for the Director-AI studio.

    Serialises (via :meth:`to_dict`) to the deterministic, sorted-key JSON the
    federation gate consumes. ``content_digest`` is computed over the contract
    body (every field except the digest itself and the environment-dependent
    ``studio_version``), so any verb, evidence-type, or ui_module change moves the
    digest and trips the drift gate, while a version bump alone does not.
    """

    verbs: tuple[Verb, ...]
    ui_module: UiModule
    studio_version: str = field(default_factory=_studio_version)

    @property
    def evidence_types(self) -> tuple[str, ...]:
        """Sorted, de-duplicated set of evidence schemas the verbs produce."""
        seen: set[str] = set()
        for verb in self.verbs:
            seen.update(verb.produces)
        return tuple(sorted(seen))

    def _contract_body(self) -> dict[str, object]:
        """Return the digest-covered contract fields (no digest, no version)."""
        return {
            "contract_era": CONTRACT_ERA,
            "protocol_version": PROTOCOL_VERSION,
            "transport_profile": TRANSPORT_PROFILE,
            "studio": STUDIO,
            "platform_sdk": PLATFORM_SDK,
            "enumeration": "language-agnostic",
            "evidence_types": list(self.evidence_types),
            "verbs": [verb.to_dict() for verb in self.verbs],
            "ui_module": self.ui_module.to_dict(),
        }

    def content_digest(self) -> str:
        """Return the deterministic ``sha256:`` digest of the contract body."""
        canonical = json.dumps(
            self._contract_body(),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Render the full schema-A manifest (contract body + digest + version)."""
        payload = self._contract_body()
        payload["content_digest"] = self.content_digest()
        payload["studio_version"] = self.studio_version
        return payload


def build_manifest() -> StudioManifest:
    """Build the Director-AI schema-A studio capability manifest."""
    return StudioManifest(verbs=_VERBS, ui_module=_UI_MODULE)
