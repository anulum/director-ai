# DirectorConfig

Dataclass-based configuration with environment variable, YAML file, and named profile loaders. All fields have sensible defaults.

## Loading Configuration

=== "Environment Variables"

    ```python
    from director_ai.core.config import DirectorConfig

    config = DirectorConfig.from_env()
    ```

    All fields map to `DIRECTOR_<FIELD>` env vars:

    ```bash
    DIRECTOR_COHERENCE_THRESHOLD=0.6
    DIRECTOR_HARD_LIMIT=0.5
    DIRECTOR_USE_NLI=true
    DIRECTOR_NLI_MODEL=lytang/MiniCheck-DeBERTa-L
    DIRECTOR_LLM_PROVIDER=openai
    DIRECTOR_VECTOR_BACKEND=chroma
    DIRECTOR_METRICS_ENABLED=true
    ```

=== "YAML File"

    ```python
    config = DirectorConfig.from_yaml("config.yaml")
    ```

    ```yaml
    coherence_threshold: 0.6
    hard_limit: 0.5
    use_nli: true
    nli_model: "lytang/MiniCheck-DeBERTa-L"
    vector_backend: chroma
    chroma_persist_dir: "./data/chroma"
    metrics_enabled: true
    ```

=== "Named Profile"

    ```python
    config = DirectorConfig.from_profile("thorough")
    ```

## Profiles

| Profile | NLI | Threshold | Candidates | Metrics | Use Case |
|---------|-----|-----------|------------|---------|----------|
| `fast` | Off | 0.5 | 1 | Off | Development, prototyping |
| `thorough` | On | 0.6 | 3 | On | Production default |
| `research` | On | 0.7 | 5 | On | Evaluation, benchmarking |
| `medical` | On | 0.75 | 3 | On | Healthcare applications |
| `finance` | On | 0.65 | 3 | On | Financial services |
| `legal` | On | 0.7 | 3 | On | Legal document review |
| `creative` | Off | 0.4 | 1 | Off | Creative writing (low halt rate) |
| `summarization` | On | 0.15 | 1 | On | Document summarization |

## Building Components from Config

```python
config = DirectorConfig.from_profile("thorough")

# Build a scorer with all config settings applied
scorer = config.build_scorer(store=my_store)

# Build a complete agent
agent = config.build_agent(store=my_store)
```

## Field Reference

### Scoring

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `coherence_threshold` | `float` | `0.6` | Minimum coherence to approve |
| `hard_limit` | `float` | `0.5` | Safety kernel emergency stop |
| `soft_limit` | `float` | `0.6` | Warning zone threshold |
| `use_nli` | `bool` | `False` | Enable DeBERTa NLI model |
| `nli_model` | `str` | `"yaxili96/FactCG-DeBERTa-v3-Large"` | HuggingFace model ID |
| `max_candidates` | `int` | `3` | LLM candidates to generate |
| `history_window` | `int` | `5` | Rolling history size |
| `scorer_backend` | `str` | `"deberta"` | `deberta`, `onnx`, `minicheck`, `hybrid`, `lite`, `rust` |

### LLM Provider

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm_provider` | `str` | `"mock"` | `mock`, `openai`, `anthropic`, `huggingface`, `local` |
| `llm_api_url` | `str` | `""` | API endpoint (for `local` provider) |
| `llm_api_key` | `str` | `""` | API key (for cloud providers) |
| `llm_model` | `str` | `""` | Model name |
| `llm_temperature` | `float` | `0.8` | Sampling temperature |
| `llm_max_tokens` | `int` | `512` | Maximum tokens per response |

### LLM-as-Judge

!!! warning "Privacy"
    When enabled, user prompts and responses are sent to the configured external LLM.
    Do not enable in privacy-sensitive deployments without user consent.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm_judge_enabled` | `bool` | `False` | Enable LLM escalation |
| `llm_judge_confidence_threshold` | `float` | `0.3` | Softmax margin for escalation |
| `llm_judge_provider` | `str` | `""` | `"openai"`, `"anthropic"`, or `"local"` |
| `llm_judge_model` | `str` | `""` | Model name for judge |
| `privacy_mode` | `bool` | `False` | Redact PII before sending |

### Vector Store

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vector_backend` | `str` | `"memory"` | `"memory"` or `"chroma"` |
| `embedding_model` | `str` | `"BAAI/bge-large-en-v1.5"` | Embedding model ID |
| `chroma_collection` | `str` | `"director_ai"` | ChromaDB collection name |
| `chroma_persist_dir` | `str` | `""` | Persistence directory (empty = in-memory) |
| `reranker_enabled` | `bool` | `False` | Enable cross-encoder reranking |
| `reranker_model` | `str` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Reranker model |

### Server

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `server_host` | `str` | `"0.0.0.0"` | Bind address |
| `server_port` | `int` | `8080` | Port |
| `server_workers` | `int` | `1` | Uvicorn worker count |
| `cors_origins` | `str` | `""` | CORS allowed origins |
| `metrics_enabled` | `bool` | `False` | Enable Prometheus metrics |
| `log_level` | `str` | `"INFO"` | Logging level |
| `log_json` | `bool` | `False` | Structured JSON logging |

### Caching

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cache_size` | `int` | `1024` | LRU cache max entries |
| `cache_ttl` | `float` | `300.0` | Cache entry TTL (seconds) |
| `redis_url` | `str` | `""` | Redis URL for distributed caching |

## Full API

::: director_ai.core.config.DirectorConfig
