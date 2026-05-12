# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Immutable model revision policy for external model artefacts."""

from __future__ import annotations

import re

DEFAULT_NLI_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
DEFAULT_NLI_MODEL_REVISION = "0430e3509dbd28d2dff7a117c0eae25359ff3e80"

MODEL_REVISION_REGISTRY: dict[str, str] = {
    DEFAULT_NLI_MODEL: DEFAULT_NLI_MODEL_REVISION,
    "lytang/MiniCheck-DeBERTa-v3-Large": "2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb",
    "distilbert-base-uncased": "12040accade4e8a0f71eabdb258fecc2e7e948be",
    "bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli": (
        "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
    ),
    "roberta-large-mnli": "2a8f12d27941090092df78e4ba6f0928eb5eac98",
    "microsoft/deberta-v3-small": "a36c739020e01763fe789b4b85e2df55d6180012",
    "distilroberta-base": "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b",
}

_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_EXPLICIT_PATH_PREFIXES = ("/", "./", "../", "~")


def _is_explicit_local_reference(model_name: str) -> bool:
    value = model_name.strip()
    if not value:
        return False
    if value.startswith(_EXPLICIT_PATH_PREFIXES):
        return True
    return bool(_URI_SCHEME_RE.match(value))


def _requires_remote_revision(model_name: str) -> bool:
    value = model_name.strip()
    if _is_explicit_local_reference(value):
        return False
    return "/" in value or value in MODEL_REVISION_REGISTRY


def resolve_model_revision(model_name: str, revision: str | None = None) -> str | None:
    """Resolve the immutable revision for a model load.

    Existing local paths and explicit path-like references are not Hub artefacts
    and therefore return ``None``. Remote repository IDs must either be present
    in ``MODEL_REVISION_REGISTRY`` or supply an explicit immutable revision.
    """

    if revision is not None:
        return revision

    pinned = MODEL_REVISION_REGISTRY.get(model_name)
    if pinned is not None:
        return pinned

    if _requires_remote_revision(model_name):
        raise ValueError(
            f"remote model {model_name!r} requires an explicit immutable revision",
        )

    return None
