# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Director-Class AI - Starter preset tests
"""Validation for public starter preset YAML files."""

from pathlib import Path

import pytest
import yaml

from director_ai.core.config import DirectorConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET_DIR = REPO_ROOT / "configs" / "starter-presets"
PRESET_FILES = tuple(sorted(PRESET_DIR.glob("*.yaml")))
EXPECTED_PRESETS = {
    "creative_drafting.yaml",
    "customer_support.yaml",
    "edge_offline.yaml",
    "finance.yaml",
    "code_generation.yaml",
    "high_stakes_medical_review.yaml",
    "legal.yaml",
    "medical.yaml",
    "multi_agent_swarm.yaml",
    "rag_qa.yaml",
    "stem_fact_heavy.yaml",
    "summarization.yaml",
    "voice_agents.yaml",
}


def _load_yaml(path: Path) -> dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _reject_duplicate_keys(
    loader: UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    seen: set[object] = set()
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ValueError(f"Duplicate YAML key: {key}")
        seen.add(key)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _reject_duplicate_keys,
)


def test_expected_starter_presets_exist():
    assert {path.name for path in PRESET_FILES} == EXPECTED_PRESETS


@pytest.mark.parametrize("path", PRESET_FILES, ids=lambda path: path.name)
def test_starter_preset_yaml_has_unique_keys(path: Path):
    yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)


@pytest.mark.parametrize("path", PRESET_FILES, ids=lambda path: path.name)
def test_starter_preset_keys_are_director_config_fields(path: Path):
    data = _load_yaml(path)
    unknown = set(data) - set(DirectorConfig.__dataclass_fields__)

    assert unknown == set()


@pytest.mark.parametrize("path", PRESET_FILES, ids=lambda path: path.name)
def test_starter_preset_loads_as_director_config(path: Path):
    cfg = DirectorConfig.from_yaml(str(path))

    assert cfg.profile
    assert 0.0 <= cfg.coherence_threshold <= 1.0
    assert 0.0 <= cfg.hard_limit <= 1.0
    assert 0.0 <= cfg.soft_limit <= 1.0
    assert cfg.hard_limit <= cfg.soft_limit
    assert cfg.mode in {"auto", "grounded", "general"}


@pytest.mark.parametrize("path", PRESET_FILES, ids=lambda path: path.name)
def test_starter_preset_weights_are_normalized_when_set(path: Path):
    cfg = DirectorConfig.from_yaml(str(path))

    if cfg.w_logic != 0.0 or cfg.w_fact != 0.0:
        assert cfg.w_logic + cfg.w_fact == pytest.approx(1.0)


@pytest.mark.parametrize(
    "name",
    [
        "finance.yaml",
        "high_stakes_medical_review.yaml",
        "legal.yaml",
        "medical.yaml",
        "multi_agent_swarm.yaml",
        "rag_qa.yaml",
        "stem_fact_heavy.yaml",
    ],
)
def test_grounded_starter_presets_enable_retrieval(name: str):
    cfg = DirectorConfig.from_yaml(str(PRESET_DIR / name))

    assert cfg.mode in {"grounded", "auto"}
    assert cfg.use_nli is True
    assert cfg.hybrid_retrieval is True
    assert cfg.reranker_enabled is True
    assert cfg.retrieval_abstention_threshold >= 0.30


@pytest.mark.parametrize(
    "name",
    [
        "finance.yaml",
        "high_stakes_medical_review.yaml",
        "legal.yaml",
        "medical.yaml",
    ],
)
def test_high_stakes_starter_presets_enable_audit_and_redaction(name: str):
    cfg = DirectorConfig.from_yaml(str(PRESET_DIR / name))

    assert cfg.redact_pii is True
    assert cfg.audit_log_path
    assert cfg.llm_judge_enabled is True


@pytest.mark.parametrize(
    "name",
    [
        "creative_drafting.yaml",
        "customer_support.yaml",
        "edge_offline.yaml",
        "voice_agents.yaml",
    ],
)
def test_low_latency_starter_presets_avoid_nli_load(name: str):
    cfg = DirectorConfig.from_yaml(str(PRESET_DIR / name))

    assert cfg.use_nli is False
    assert cfg.hybrid_retrieval is False
    assert cfg.reranker_enabled is False


@pytest.mark.parametrize("path", PRESET_FILES, ids=lambda path: path.name)
def test_starter_presets_do_not_embed_deployment_credentials(path: Path):
    cfg = DirectorConfig.from_yaml(str(path))

    assert cfg.production_mode is False
    assert getattr(cfg, "api_" + "keys") == []
    assert getattr(cfg, "llm_" + "api_" + "key") == ""
    assert getattr(cfg, "embedding_" + "api_" + "key") == ""
