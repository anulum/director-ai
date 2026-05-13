# Differentially Private Score Release

::: director_ai.core.federated_privacy.score_release.DifferentialPrivacyScoreReleaser

::: director_ai.core.federated_privacy.score_release.PrivacyScoreRelease

## Boundary

`DifferentialPrivacyScoreReleaser` adds calibrated Laplace noise to coherence
scores before they are exposed outside the runtime trust boundary. It is for
dashboards, analytics exports, cross-tenant aggregates, and public summaries
where an exact score could leak membership information about the retrieval
corpus.

It is not used for internal enforcement decisions. Halt and approval logic
should continue to use raw scores inside the protected runtime.

```python
from director_ai.core import DifferentialPrivacyScoreReleaser
from director_ai.core.federated_privacy import PrivacyAccountant

accountant = PrivacyAccountant(max_epsilon=5.0)
releaser = DifferentialPrivacyScoreReleaser(
    epsilon=0.25,
    sensitivity=1.0,
    accountant=accountant,
)

release = releaser.release_score(
    score.raw_score,
    label="public-dashboard",
    tenant_id="tenant-a",
    threshold=0.6,
)
publish(release.to_dict())
```

The release payload omits the raw score and carries mechanism, epsilon,
sensitivity, and budget metadata. Use `PrivacyAccountant` to cap cumulative
privacy loss across score disclosures.
