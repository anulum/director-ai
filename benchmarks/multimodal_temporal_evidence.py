# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — multimodal temporal evidence packet

"""Generate local evidence for multimodal and temporal guardrails.

The packet checks the production-relevant R12 primitives without model
downloads:

* image claims map to allow, warn, or halt guard decisions;
* caption and metadata grounding add evidence references without raw payloads;
* video frame similarity streams halt on temporal consistency collapse;
* hash-bag image/text checks run through the dependency-free fallback path.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks._common import save_results
from benchmarks._provenance import resolve_git_sha
from director_ai.core.guard_control import RiskEnvelope
from director_ai.core.multimodal_guard import (
    HashBagCrossModalVerifier,
    HashBagImageEncoder,
    MultimodalCheckRequest,
    MultimodalGuard,
    MultimodalVerifierAdapter,
)


@dataclass(frozen=True)
class _Verdict:
    label: str
    similarity: float
    reason: str


class _ConstantGuard:
    def __init__(self, label: str, similarity: float) -> None:
        self._label = label
        self._similarity = similarity

    def check(self, _claim) -> _Verdict:
        return _Verdict(
            label=self._label,
            similarity=self._similarity,
            reason=f"{self._label}:{self._similarity}",
        )


def _envelope() -> RiskEnvelope:
    return RiskEnvelope(
        action_category="multimodal",
        reversibility="reversible",
        domain="regulated",
        calibrated_threshold=0.5,
        no_go_threshold=0.85,
    )


def run_image_claim_probe() -> dict[str, Any]:
    """Return image claim allow/halt/grounding evidence."""
    cases = [
        (
            "supported",
            MultimodalVerifierAdapter(
                image_guard=_ConstantGuard("consistent", 0.95),
                enabled_modalities=("image",),
                benchmarked_modalities=("image",),
            ),
            MultimodalCheckRequest(
                modality="image",
                claim_text="The image shows an approved label.",
                media_ref="media://image-supported",
                image_bytes=b"raw image payload",
            ),
            "allow",
        ),
        (
            "hallucinated",
            MultimodalVerifierAdapter(
                image_guard=_ConstantGuard("hallucinated", 0.05),
                enabled_modalities=("image",),
                benchmarked_modalities=("image",),
            ),
            MultimodalCheckRequest(
                modality="image",
                claim_text="The image shows a medical device.",
                media_ref="media://image-halt",
                image_bytes=b"raw image payload",
            ),
            "halt",
        ),
        (
            "caption_conflict",
            MultimodalVerifierAdapter(
                image_guard=_ConstantGuard("consistent", 0.94),
                caption_score_fn=lambda _caption, _claim: 0.12,
                enabled_modalities=("image",),
                benchmarked_modalities=("image",),
            ),
            MultimodalCheckRequest(
                modality="image",
                claim_text="The image shows an approved label.",
                media_ref="media://image-caption",
                image_bytes=b"raw image payload",
                caption_text="Caption says the label is missing.",
            ),
            "halt",
        ),
    ]
    records = []
    raw_payload_leaked = False
    for name, adapter, request, expected in cases:
        result = adapter.check(
            request,
            risk_envelope=_envelope(),
            policy_id="policy.multimodal.regulated",
        )
        serialised = json.dumps(result.to_dict(), sort_keys=True)
        raw_payload_leaked = raw_payload_leaked or "raw image payload" in serialised
        raw_payload_leaked = raw_payload_leaked or request.claim_text in serialised
        records.append(
            {
                "case": name,
                "decision": result.guard_decision.decision,
                "expected_decision": expected,
                "verdict": result.signal.verdict,
                "risk_score": round(result.guard_decision.risk_score, 4),
                "evidence_refs": list(result.guard_decision.evidence_refs),
                "matched": result.guard_decision.decision == expected,
            }
        )
    return {
        "name": "image_claim_paths",
        "records": records,
        "raw_payload_leaked": raw_payload_leaked,
        "passed": bool(
            all(record["matched"] for record in records) and not raw_payload_leaked
        ),
    }


def run_video_temporal_probe() -> dict[str, Any]:
    """Return video temporal consistency halt evidence."""
    adapter = MultimodalVerifierAdapter(
        enabled_modalities=("video",),
        benchmarked_modalities=("video",),
    )
    request = MultimodalCheckRequest(
        modality="video",
        claim_text="The object remains in view.",
        media_ref="media://video-temporal",
        frame_similarities=(0.9, 0.1, 0.0, 0.0),
    )
    result = adapter.check(
        request,
        risk_envelope=_envelope(),
        policy_id="policy.multimodal.regulated",
    )
    serialised = json.dumps(result.to_dict(), sort_keys=True)
    return {
        "name": "video_temporal_consistency",
        "decision": result.guard_decision.decision,
        "verdict": result.signal.verdict,
        "evidence_refs": list(result.guard_decision.evidence_refs),
        "claim_text_leaked": request.claim_text in serialised,
        "passed": bool(
            result.guard_decision.decision == "halt"
            and result.signal.verdict == "temporal_inconsistent"
            and "media://video-temporal#frame:3" in result.guard_decision.evidence_refs
            and request.claim_text not in serialised
        ),
    }


def run_hashbag_fallback_probe() -> dict[str, Any]:
    """Return dependency-free hash-bag encoder/verifier evidence."""
    encoder = HashBagImageEncoder(dim=64)
    verifier = HashBagCrossModalVerifier(dim=64)
    guard = MultimodalGuard(
        encoder=encoder,
        verifier=verifier,
        hallucination_threshold=0.05,
        consistency_threshold=0.1,
    )
    verdict = guard.check(
        type(
            "Claim",
            (),
            {
                "image_bytes": b"cat",
                "text_claim": "cat",
            },
        )()
    )
    return {
        "name": "hashbag_dependency_free",
        "label": verdict.label,
        "similarity": round(verdict.similarity, 4),
        "passed": bool(verdict.similarity >= 0.0 and verdict.label != "hallucinated"),
    }


def run_multimodal_temporal_evidence() -> dict[str, Any]:
    """Return the complete local R12 multimodal temporal evidence packet."""
    image = run_image_claim_probe()
    video = run_video_temporal_probe()
    hashbag = run_hashbag_fallback_probe()
    passed = bool(image["passed"] and video["passed"] and hashbag["passed"])
    return {
        "schema_version": "director-ai.multimodal-temporal-evidence.v1",
        "benchmark": "multimodal_temporal_evidence",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": resolve_git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "acceptance": {
            "passed": passed,
            "checks": {
                "image_claim_paths": bool(image["passed"]),
                "video_temporal_consistency": bool(video["passed"]),
                "hashbag_dependency_free": bool(hashbag["passed"]),
            },
            "limits": {
                "local_only": True,
                "external_vision_nli_benchmark_included": False,
                "real_video_model_included": False,
            },
        },
        "probes": {
            "image_claim_paths": image,
            "video_temporal_consistency": video,
            "hashbag_dependency_free": hashbag,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Director-AI multimodal temporal evidence packet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to benchmarks/results/.",
    )
    args = parser.parse_args(argv)

    payload = run_multimodal_temporal_evidence()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results saved to {args.output}")
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        save_results(payload, f"multimodal_temporal_evidence_{stamp}.json")
    print(json.dumps(payload["acceptance"], indent=2))
    return 0 if payload["acceptance"]["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
