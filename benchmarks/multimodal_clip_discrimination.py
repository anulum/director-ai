# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CLIP vs hash-bag multimodal discrimination benchmark

"""Image-grounding discrimination: real CLIP backend vs the FNV hash-bag baseline.

Generates solid-colour images with Pillow and pairs each with a matched caption
("a solid red image") and mismatched captions (the other colours). A real
image-grounding backend should score the matched caption clearly above the
mismatched ones; the byte hash-bag baseline cannot relate image bytes to text
semantics and shows no separation. The gap between the two is the upgrade CLIP
buys.

Pillow + open_clip are the ``[multimodal]`` extra, not core deps, so without them
the CLIP stage records ``available: false`` (with an install hint) rather than a
fabricated number. The hash-bag stage always runs (dependency-free).

Run::

    pip install "director-ai[multimodal]"
    python -m benchmarks.multimodal_clip_discrimination
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
_COLOURS = {
    "red": (220, 20, 20),
    "green": (20, 200, 20),
    "blue": (20, 20, 220),
    "yellow": (230, 220, 20),
}


def separation(matched: list[float], mismatched: list[float]) -> dict:
    """Mean matched / mismatched similarity, their gap, and pairwise AUC."""
    if not matched or not mismatched:
        return {"n_matched": len(matched), "n_mismatched": len(mismatched)}
    mean_m = sum(matched) / len(matched)
    mean_x = sum(mismatched) / len(mismatched)
    wins = sum(1 for m in matched for x in mismatched if m > x)
    ties = sum(1 for m in matched for x in mismatched if m == x)
    total = len(matched) * len(mismatched)
    auc = (wins + 0.5 * ties) / total
    return {
        "n_matched": len(matched),
        "n_mismatched": len(mismatched),
        "mean_matched": round(mean_m, 4),
        "mean_mismatched": round(mean_x, 4),
        "gap": round(mean_m - mean_x, 4),
        "pairwise_auc": round(auc, 4),
    }


def _png_bytes(colour: tuple[int, int, int]) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), colour).save(buf, format="PNG")
    return buf.getvalue()


def _score_backend(verify, encode) -> dict:
    """Collect matched/mismatched image-vs-caption scores for a backend."""
    matched: list[float] = []
    mismatched: list[float] = []
    for colour, rgb in _COLOURS.items():
        embedding = encode(_png_bytes(rgb))
        for caption_colour in _COLOURS:
            score = verify(embedding, f"a solid {caption_colour} image")
            (matched if caption_colour == colour else mismatched).append(score)
    return separation(matched, mismatched)


def _run_clip() -> dict:
    try:
        import open_clip  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        return {
            "available": False,
            "reason": f"{exc}. Install with: pip install director-ai[multimodal]",
        }
    from director_ai.core.multimodal_guard.encoders import TorchCLIPImageEncoder
    from director_ai.core.multimodal_guard.factory import _default_clip_loader
    from director_ai.core.multimodal_guard.verifier import TorchCLIPCrossModalVerifier

    model, preprocess, tokenizer, dim = _default_clip_loader(
        "ViT-B-32", "openai", "cpu"
    )
    encoder = TorchCLIPImageEncoder(model=model, preprocess=preprocess, dim=dim)
    verifier = TorchCLIPCrossModalVerifier(model=model, tokenizer=tokenizer, dim=dim)
    result = _score_backend(verifier.verify, encoder.encode)
    result["available"] = True
    return result


def _run_hashbag() -> dict:
    from director_ai.core.multimodal_guard.encoders import HashBagImageEncoder
    from director_ai.core.multimodal_guard.verifier import HashBagCrossModalVerifier

    encoder = HashBagImageEncoder(dim=512)
    verifier = HashBagCrossModalVerifier(dim=512)

    # The hash-bag baseline needs raw image bytes; reuse the PNG generator only if
    # Pillow is present, else fall back to deterministic synthetic byte payloads.
    try:
        from PIL import Image  # noqa: F401

        byte_source = _png_bytes
    except ImportError:

        def byte_source(rgb: tuple[int, int, int]) -> bytes:
            return bytes(rgb) * 512

    matched: list[float] = []
    mismatched: list[float] = []
    for colour, rgb in _COLOURS.items():
        embedding = encoder.encode(byte_source(rgb))
        for caption_colour in _COLOURS:
            score = verifier.verify(embedding, f"a solid {caption_colour} image")
            (matched if caption_colour == colour else mismatched).append(score)
    return {"available": True, **separation(matched, mismatched)}


def run_benchmark() -> dict:
    return {
        "benchmark": "multimodal_clip_discrimination",
        "colours": sorted(_COLOURS),
        "clip": _run_clip(),
        "hashbag_baseline": _run_hashbag(),
        "note": (
            "Matched = correct-colour caption, mismatched = other-colour captions. "
            "A semantic backend separates them (gap > 0, AUC -> 1); the byte "
            "hash-bag cannot relate image bytes to text and shows no separation."
        ),
    }


def main() -> None:
    result = run_benchmark()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "multimodal_clip_discrimination.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nMultimodal image-grounding discrimination (CLIP vs hash-bag):")
    for name in ("clip", "hashbag_baseline"):
        stage = result[name]
        if stage.get("available"):
            print(
                f"  {name:18} gap={stage['gap']:+.4f}  AUC={stage['pairwise_auc']:.4f} "
                f"(matched {stage['mean_matched']:.3f} vs {stage['mean_mismatched']:.3f})"
            )
        else:
            print(f"  {name:18} unavailable: {stage.get('reason')}")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
