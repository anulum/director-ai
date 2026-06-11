# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — airgap install example tests

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

from director_ai.core.config import DirectorConfig
from director_ai.core.scoring.nli import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "requirements" / "airgap_full_stack.toml"
DOC_PATH = ROOT / "docs-site" / "deployment" / "airgap.md"
SCRIPT_PATH = ROOT / "scripts" / "airgap_full_stack_example.sh"


def _load_manifest() -> dict:
    return tomllib.loads(MANIFEST_PATH.read_text())


def test_airgap_manifest_points_to_real_files() -> None:
    manifest = _load_manifest()
    policy = manifest["policy"]

    assert (ROOT / policy["doc"]).is_file()
    assert (ROOT / policy["script"]).is_file()
    assert (ROOT / policy["lock_file"]).is_file()
    assert "rust/" in policy["rust_wheel_dir"]


def test_airgap_script_has_valid_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT_PATH)], check=True)


def test_airgap_model_revisions_match_config_and_registry() -> None:
    manifest = _load_manifest()
    config = DirectorConfig()
    models = {model["key"]: model for model in manifest["models"]}

    assert models["nli"]["revision"] == config.nli_model_revision
    assert models["nli"]["revision"] == MODEL_REGISTRY[models["nli"]["name"]]
    assert models["embedding"]["revision"] == config.embedding_model_revision
    assert models["reranker"]["revision"] == config.reranker_model_revision


def test_airgap_docs_cover_install_inputs_and_checks() -> None:
    manifest = _load_manifest()
    text = DOC_PATH.read_text()
    script = SCRIPT_PATH.read_text()

    for extra in manifest["policy"]["install_extras"]:
        assert extra in text

    for model in manifest["models"]:
        assert model["name"] in text
        assert model["revision"] in text
        assert model["local_dir"] in text

    for filename in manifest["onnx"]["required_files"]:
        assert filename in text
        assert filename in script

    for command in manifest["checks"]["commands"]:
        assert re.sub(r"\s+", " ", command) in re.sub(r"\s+", " ", text + script)
