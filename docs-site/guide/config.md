# Configuration

`DirectorConfig` is a dataclass with environment variable, YAML file, and named profile loaders. All fields have sensible defaults.

## Loading

=== "Environment Variables"

    ```bash
    export DIRECTOR_COHERENCE_THRESHOLD=0.6
    export DIRECTOR_USE_NLI=true
    export DIRECTOR_VECTOR_BACKEND=chroma
    export DIRECTOR_METRICS_ENABLED=true
    ```

    ```python
    from director_ai.core.config import DirectorConfig
    config = DirectorConfig.from_env()
    ```

=== "YAML File"

    ```yaml
    # config.yaml
    coherence_threshold: 0.6
    hard_limit: 0.5
    use_nli: true
    nli_model: "lytang/MiniCheck-DeBERTa-L"
    vector_backend: chroma
    chroma_persist_dir: "./data/chroma"
    metrics_enabled: true
    ```

    ```python
    config = DirectorConfig.from_yaml("config.yaml")
    ```

=== "Named Profile"

    ```python
    config = DirectorConfig.from_profile("thorough")
    ```

=== "Direct Construction"

    ```python
    config = DirectorConfig(
        coherence_threshold=0.7,
        use_nli=True,
        scorer_backend="onnx",
    )
    ```

## Profiles

| Profile | NLI | Threshold | Candidates | Metrics | Use Case |
|---------|-----|-----------|------------|---------|----------|
| `fast` | Off | 0.5 | 1 | Off | Development, low latency |
| `thorough` | On | 0.6 | 3 | On | Production default |
| `research` | On | 0.7 | 5 | On | Evaluation, benchmarking |
| `medical` | On | 0.30 | 3 | On | Healthcare (measured on PubMedQA) |
| `finance` | On | 0.30 | 3 | On | Financial services (measured on FinanceBench) |
| `legal` | On | 0.30 | 3 | On | Legal document review (not yet measured) |
| `creative` | Off | 0.4 | 1 | Off | Creative writing (low halt rate) |
| `customer_support` | Off | 0.55 | 1 | Off | Support agents |
| `summarization` | On | 0.15 | 1 | On | Document summarization |
| `lite` | Off | 0.5 | 1 | Off | Zero-dependency fast path |

## Building Components

```python
config = DirectorConfig.from_profile("thorough")

# Build scorer with all config applied
scorer = config.build_scorer(store=my_store)

# Build vector store from config
store = config.build_store()
```

## Combining Profile + Overrides

```python
config = DirectorConfig.from_profile("medical")
config.nli_model = "lytang/MiniCheck-DeBERTa-L"
config.cache_size = 4096
```

Or combine YAML + env vars (env vars take precedence):

```python
config = DirectorConfig.from_yaml("config.yaml")
# DIRECTOR_USE_NLI=true overrides use_nli in YAML
```

## Key Field Groups

### Scoring

`coherence_threshold`, `hard_limit`, `soft_limit`, `use_nli`, `nli_model`, `scorer_backend`, `max_candidates`, `history_window`

### LLM Provider

`llm_provider` (`mock` | `openai` | `anthropic` | `huggingface` | `local`), `llm_api_key`, `llm_model`, `llm_temperature`, `llm_max_tokens`

### Vector Store

`vector_backend` (`memory` | `chroma`), `embedding_model`, `chroma_collection`, `chroma_persist_dir`, `reranker_enabled`

### Server

`server_host`, `server_port`, `server_workers`, `cors_origins`, `rate_limit_rpm`, `api_keys`

### Caching

`cache_size`, `cache_ttl`, `redis_url`

### Observability

`metrics_enabled`, `log_level`, `log_json`, `otel_enabled`

See [DirectorConfig API Reference](../api/config.md) for the complete field table.
