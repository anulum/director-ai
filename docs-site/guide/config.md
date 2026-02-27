# Configuration

## DirectorConfig

Dataclass-based configuration with env var, YAML, and profile support.

```python
from director_ai.core.config import DirectorConfig

# From environment variables (DIRECTOR_COHERENCE_THRESHOLD=0.7)
config = DirectorConfig.from_env()

# From YAML file
config = DirectorConfig.from_yaml("config.yaml")

# From named profile
config = DirectorConfig.from_profile("thorough")
```

## Profiles

| Profile | NLI | Threshold | Candidates | Metrics |
|---------|-----|-----------|------------|---------|
| `fast` | Off | 0.5 | 1 | Off |
| `thorough` | On | 0.6 | 3 | On |
| `research` | On | 0.7 | 5 | On |

## Environment Variables

All fields can be set via `DIRECTOR_<FIELD>` env vars:

```bash
DIRECTOR_COHERENCE_THRESHOLD=0.6
DIRECTOR_HARD_LIMIT=0.5
DIRECTOR_SOFT_LIMIT=0.65
DIRECTOR_USE_NLI=true
DIRECTOR_NLI_MODEL=lytang/MiniCheck-DeBERTa-L
DIRECTOR_LLM_PROVIDER=openai
DIRECTOR_VECTOR_BACKEND=chroma
DIRECTOR_METRICS_ENABLED=true
```

## YAML Configuration

```yaml
coherence_threshold: 0.6
hard_limit: 0.5
soft_limit: 0.65
use_nli: true
nli_model: "lytang/MiniCheck-DeBERTa-L"
vector_backend: chroma
chroma_persist_dir: "./data/chroma"
metrics_enabled: true
log_level: INFO
```
