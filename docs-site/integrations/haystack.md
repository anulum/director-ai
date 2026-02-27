# Haystack

```bash
pip install director-ai[haystack]
```

## Pipeline Component

```python
from director_ai.integrations.haystack import DirectorAIChecker
from haystack import Pipeline

checker = DirectorAIChecker(
    facts={"capital": "Paris is the capital of France."},
    threshold=0.6,
    filter_rejected=True,  # Remove rejected replies
)

pipeline = Pipeline()
pipeline.add_component("checker", checker)

result = pipeline.run({
    "checker": {
        "query": "What is the capital?",
        "replies": ["Paris is the capital.", "Berlin is the capital."],
    }
})

print(result["checker"]["replies"])   # Only approved replies
print(result["checker"]["scores"])    # Score details for all
print(result["checker"]["approved"])  # [True, False]
```
