# DirectorConfig

Dataclass-based configuration with env var, YAML, and profile loaders.

## Fields

| Field | Default | Description |
|-------|---------|-------------|
| **Scoring** | | |
| `coherence_threshold` | `0.6` | Minimum coherence to approve |
| `hard_limit` | `0.5` | Safety kernel emergency stop floor |
| `soft_limit` | `0.6` | Warning zone floor (>= hard_limit) |
| `use_nli` | `False` | Load DeBERTa NLI model |
| `nli_model` | `"yaxili96/FactCG-DeBERTa-v3-Large"` | HF model ID |
| `max_candidates` | `3` | LLM candidates to generate |
| `history_window` | `5` | Rolling history size |
| `w_logic` / `w_fact` | `0.0` | Divergence weights (0 = class default; must sum to 1.0 when set) |
| **LLM** | | |
| `llm_provider` | `"mock"` | `mock`, `openai`, `anthropic`, `huggingface`, `local` |
| `llm_api_key` | `""` | API key (redacted in `to_dict()`) |
| `llm_model` | `""` | Model name |
| `llm_temperature` | `0.8` | Sampling temperature [0, 2] |
| `llm_max_tokens` | `512` | Max tokens per response |
| **Vector store** | | |
| `vector_backend` | `"memory"` | `"memory"` or `"chroma"` |
| `reranker_enabled` | `False` | Cross-encoder reranking |
| **Server** | | |
| `server_host` | `"0.0.0.0"` | Bind address |
| `server_port` | `8080` | Port |
| `server_workers` | `1` | Uvicorn workers |
| **Batch / Auth / Rate** | | |
| `batch_max_concurrency` | `4` | Max concurrent batch items |
| `rate_limit_rpm` | `0` | Requests/minute (0 = disabled) |
| `api_keys` | `[]` | Allowed API keys (empty = no auth) |
| **Observability** | | |
| `metrics_enabled` | `True` | Prometheus metrics |
| `log_level` | `"INFO"` | Logging level |
| `log_json` | `False` | Structured JSON logs |
| `audit_log_path` | `""` | SQLite audit path (empty = disabled) |
| `tenant_routing` | `False` | Multi-tenant scorer isolation |

## Profiles via `from_profile(name)`

| Profile | threshold | hard_limit | NLI | w_logic/w_fact |
|---------|-----------|------------|-----|----------------|
| `fast` | 0.50 | 0.50 | off | default |
| `thorough` | 0.60 | 0.50 | on | default |
| `research` | 0.70 | 0.50 | on | default |
| `medical` | 0.75 | 0.55 | on | 0.5 / 0.5 |
| `finance` | 0.70 | 0.50 | on | 0.4 / 0.6 |
| `legal` | 0.68 | 0.45 | on | 0.6 / 0.4 |
| `creative` | 0.40 | 0.30 | off | 0.7 / 0.3 |
| `customer_support` | 0.55 | 0.40 | off | 0.5 / 0.5 |

## Loaders

- **`from_yaml(path)`** -- YAML or JSON file. Unrecognised keys ignored.
- **`from_env(prefix="DIRECTOR_")`** -- e.g. `DIRECTOR_COHERENCE_THRESHOLD=0.7`.
- **`to_dict()`** -- serializes to dict; `llm_api_key`/`api_keys` redacted.
- **`configure_logging()`** -- applies `log_level` + `log_json`.

## Example

```python
from director_ai.core.config import DirectorConfig

cfg = DirectorConfig.from_profile("medical")
cfg.configure_logging()

# Or from env / YAML
cfg = DirectorConfig.from_env()
cfg = DirectorConfig.from_yaml("director.yaml")
```
