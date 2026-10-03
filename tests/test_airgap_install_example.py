# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — airgap install example tests

"""Check offline-install manifests, shell syntax and documented model inputs."""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

from director_ai.core.config import DirectorConfig
from director_ai.core.scoring.nli import MODEL_REGISTRY
from tools.ci_workflow_inventory import object_mapping

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "requirements" / "airgap_full_stack.toml"
DOC_PATH = ROOT / "docs-site" / "deployment" / "airgap.md"
SCRIPT_PATH = ROOT / "scripts" / "airgap_full_stack_example.sh"


def _load_manifest() -> dict[str, object]:
    """Read the maintained offline-install manifest as decoded TOML.

    Returns
    -------
    dict[str, object]
        String-keyed root retaining policy, model and verification values.

    Raises
    ------
    OSError
        If the manifest cannot be read.
    tomllib.TOMLDecodeError
        If the manifest contains invalid TOML.
    ValueError
        If its decoded root is not a string-keyed mapping.
    """
    return object_mapping(tomllib.loads(MANIFEST_PATH.read_text()))


def test_airgap_manifest_points_to_real_files() -> None:
    """Resolve the manifest's maintained guide, script and lock paths."""
    manifest = _load_manifest()
    policy = object_mapping(manifest["policy"])

    doc = policy["doc"]
    assert isinstance(doc, str)
    assert (ROOT / doc).is_file()
    script = policy["script"]
    assert isinstance(script, str)
    assert (ROOT / script).is_file()
    lock_file = policy["lock_file"]
    assert isinstance(lock_file, str)
    assert (ROOT / lock_file).is_file()
    rust_wheel_dir = policy["rust_wheel_dir"]
    assert isinstance(rust_wheel_dir, str)
    assert "rust/" in rust_wheel_dir


def test_airgap_script_has_valid_shell_syntax() -> None:
    """Parse the offline-install shell script without installing packages."""
    subprocess.run(["bash", "-n", str(SCRIPT_PATH)], check=True)


def test_airgap_model_revisions_match_config_and_registry() -> None:
    """Match declared model revisions to public defaults and the registry."""
    manifest = _load_manifest()
    config = DirectorConfig()
    model_records = manifest["models"]
    assert isinstance(model_records, list)
    models: dict[str, dict[str, object]] = {}
    for record in model_records:
        model = object_mapping(record)
        key = model["key"]
        assert isinstance(key, str)
        models[key] = model

    assert models["nli"]["revision"] == config.nli_model_revision
    nli_name = models["nli"]["name"]
    assert isinstance(nli_name, str)
    assert models["nli"]["revision"] == MODEL_REGISTRY[nli_name]
    assert models["embedding"]["revision"] == config.embedding_model_revision
    assert models["reranker"]["revision"] == config.reranker_model_revision


def test_airgap_docs_cover_install_inputs_and_checks() -> None:
    """Match install extras, model inputs and checks to the guide and script."""
    manifest = _load_manifest()
    text = DOC_PATH.read_text()
    script = SCRIPT_PATH.read_text()

    policy = object_mapping(manifest["policy"])
    install_extras = policy["install_extras"]
    assert isinstance(install_extras, list)
    for extra in install_extras:
        assert isinstance(extra, str)
        assert extra in text

    model_records = manifest["models"]
    assert isinstance(model_records, list)
    for record in model_records:
        model = object_mapping(record)
        name = model["name"]
        assert isinstance(name, str)
        assert name in text
        revision = model["revision"]
        assert isinstance(revision, str)
        assert revision in text
        local_dir = model["local_dir"]
        assert isinstance(local_dir, str)
        assert local_dir in text

    required_files = object_mapping(manifest["onnx"])["required_files"]
    assert isinstance(required_files, list)
    for filename in required_files:
        assert isinstance(filename, str)
        assert filename in text
        assert filename in script

    commands = object_mapping(manifest["checks"])["commands"]
    assert isinstance(commands, list)
    for command in commands:
        assert isinstance(command, str)
        assert re.sub(r"\s+", " ", command) in re.sub(r"\s+", " ", text + script)
