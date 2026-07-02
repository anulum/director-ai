# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — STUDIO architecture-map.v2 federation extension

"""Build the Director-AI STUDIO architecture-map.v2 extension.

The schema-A capability manifest is the minimal discovery contract. The
``architecture-map.v2`` extension gives SCPN-STUDIO a hub-ready topology view:
runtime backends, wired capabilities, exposed interfaces, pipeline stages, wire
formats, cross-repo adapters, and explicit ownership boundaries. The entries are
authored from shipped Director-AI surfaces and deliberately stay additive to the
schema-A contract, so older schema-A consumers can continue reading the nested
``schema_a`` block from the federation envelope.
"""

from __future__ import annotations

from .manifest import build_manifest

__all__ = [
    "ARCHITECTURE_MAP_VERSION",
    "build_architecture_map_extension",
    "build_federation_document",
]

#: Version tag accepted by the SCPN-STUDIO federation hub.
ARCHITECTURE_MAP_VERSION = "architecture-map.v2"


def _backends() -> list[dict[str, object]]:
    """Return the runtime and build-target backends exposed by Director-AI."""
    return [
        {
            "name": "python",
            "language": "python",
            "role": "reference-orchestrator",
            "status": "runtime-active",
            "dispatch_order": 0,
        },
        {
            "name": "rust",
            "language": "rust",
            "role": "scoring-accelerator",
            "status": "runtime-active",
            "dispatch_order": 1,
        },
        {
            "name": "onnx",
            "language": "onnx",
            "role": "nli-inference-export",
            "status": "build-available",
            "dispatch_order": 2,
        },
    ]


def _capabilities() -> list[dict[str, object]]:
    """Return capability rows that correspond to shipped schema-A verbs."""
    return [
        {
            "name": "coherence-scorer",
            "verb": "score",
            "domain": "guardrail",
            "status": "wired",
            "safety_tier": "production",
        },
        {
            "name": "verified-scorer",
            "verb": "validate",
            "domain": "guardrail",
            "status": "wired",
            "safety_tier": "production",
        },
        {
            "name": "streaming-halt",
            "verb": "halt",
            "domain": "runtime",
            "status": "wired",
            "safety_tier": "research",
        },
        {
            "name": "conformal-calibration",
            "verb": "calibrate",
            "domain": "calibration",
            "status": "wired",
            "safety_tier": "research",
        },
        {
            "name": "prompt-injection-detection",
            "verb": "detect-injection",
            "domain": "guardrail",
            "status": "wired",
            "safety_tier": "research",
        },
        {
            "name": "backend-benchmark",
            "verb": "benchmark",
            "domain": "evaluation",
            "status": "wired",
            "safety_tier": "research",
        },
        {
            "name": "evidence-replay",
            "verb": "replay",
            "domain": "audit",
            "status": "wired",
            "safety_tier": "research",
        },
        {
            "name": "pii-redaction",
            "verb": "redact",
            "domain": "privacy",
            "status": "wired",
            "safety_tier": "research",
        },
    ]


def _interfaces() -> list[dict[str, object]]:
    """Return discoverable entry points for host portals and integrators."""
    return [
        {
            "kind": "python_api",
            "entry": "director_ai.guard.guard",
            "contract": "runtime guard orchestration",
        },
        {
            "kind": "cli",
            "entry": "director-ai",
            "contract": "local-first operator entry point",
        },
        {
            "kind": "rest",
            "entry": "/v1/review",
            "contract": "response review surface",
        },
        {
            "kind": "grpc",
            "entry": "director.DirectorService",
            "contract": "service integration surface",
        },
        {
            "kind": "ui_module",
            "entry": "/studio/remoteEntry.js",
            "contract": "Module Federation 2 DirectorAIStudioPanel",
        },
        {
            "kind": "artifact",
            "entry": "docs/_generated/studio_manifest.json",
            "contract": "SCPN-STUDIO federation envelope",
        },
    ]


def _pipeline_stages() -> list[dict[str, object]]:
    """Return the STUDIO pipeline stages exposed for architecture routing."""
    return [
        {
            "stage": "candidate-generation",
            "inputs": ["prompt:text"],
            "outputs": ["candidate:text"],
            "processing_model": "LLMGenerator/provider boundary",
        },
        {
            "stage": "response-verification",
            "inputs": ["prompt:text", "candidate:text"],
            "outputs": ["studio.verification.v1"],
            "processing_model": "CoherenceScorer plus VerifiedScorer",
        },
        {
            "stage": "evidence-feedback",
            "inputs": ["studio.verification.v1"],
            "outputs": ["recall.correctness.feedback.v1"],
            "processing_model": "RemanentiaCorrectnessClient",
        },
        {
            "stage": "operator-surface",
            "inputs": ["studio.verification.v1", "studio.response-score.v1"],
            "outputs": ["studio.federation-envelope.v2"],
            "processing_model": "SCPN-STUDIO schema-A hub adapter",
        },
    ]


def _wire_formats() -> list[dict[str, object]]:
    """Return named wire formats and the local surfaces that ground them."""
    return [
        {
            "name": "studio.federation-envelope.v2",
            "schema_ref": "docs/_generated/studio_manifest.json",
            "producer": "tools/emit_studio_manifest.py",
        },
        {
            "name": "studio.response-score.v1",
            "schema_ref": "src/director_ai/federation/manifest.py",
            "producer": "score verb",
        },
        {
            "name": "studio.verification.v1",
            "schema_ref": "src/director_ai/core/scoring/verified_scorer.py",
            "producer": "validate verb",
        },
        {
            "name": "studio.streaming-halt.v1",
            "schema_ref": "src/director_ai/core/runtime/streaming.py",
            "producer": "halt verb",
        },
        {
            "name": "studio.backend-benchmark.v1",
            "schema_ref": "benchmarks/",
            "producer": "benchmark verb",
        },
        {
            "name": "recall.correctness.feedback.v1",
            "schema_ref": "src/director_ai/core/calibration/recall_correctness_client.py",
            "producer": "evidence-feedback stage",
        },
    ]


def _cross_repo() -> list[dict[str, object]]:
    """Return cross-repo federation edges the STUDIO hub can route."""
    return [
        {
            "sibling": "SCPN-STUDIO",
            "adapter": "schema-A federation envelope",
            "wire_format": "studio.federation-envelope.v2",
        },
        {
            "sibling": "SYNAPSE-CHANNEL",
            "adapter": "VerifiedScorer verdict and release receipt evidence",
            "wire_format": "studio.verification.v1",
        },
        {
            "sibling": "REMANENTIA",
            "adapter": "RemanentiaCorrectnessClient recall feedback",
            "wire_format": "recall.correctness.feedback.v1",
        },
        {
            "sibling": "DIRECTOR-CLASS-AI",
            "adapter": "shared verification lineage",
            "wire_format": "studio.verification.v1",
        },
    ]


def _boundaries() -> dict[str, list[str]]:
    """Return explicit ownership boundaries for hub consumers."""
    return {
        "bounded": [
            "response-level verification",
            "atomic claim verification",
            "streaming oversight research signal",
            "prompt injection scanning",
            "PII redaction",
        ],
        "closed": [
            "long-term memory storage",
            "coordination bus ownership",
            "source-of-truth billing",
            "unsourced benchmark claims",
        ],
    }


def build_architecture_map_extension() -> dict[str, object]:
    """Return the hub-ready ``architecture-map.v2`` federation extension."""
    return {
        "version": ARCHITECTURE_MAP_VERSION,
        "backends": _backends(),
        "capabilities": _capabilities(),
        "interfaces": _interfaces(),
        "pipeline_stages": _pipeline_stages(),
        "wire_formats": _wire_formats(),
        "cross_repo": _cross_repo(),
        "boundaries": _boundaries(),
    }


def build_federation_document() -> dict[str, object]:
    """Return the SCPN-STUDIO federation envelope for Director-AI.

    The envelope keeps the original schema-A capability manifest intact under
    ``schema_a`` and adds the topology extension under ``architecture_map``.
    """
    return {
        "schema_a": build_manifest().to_dict(),
        "architecture_map": build_architecture_map_extension(),
    }
