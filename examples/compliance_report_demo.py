# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""EU AI Act Article 15 compliance report demo."""

import tempfile
import time
from pathlib import Path

from director_ai import Article15Report, AuditEntry, AuditLog, ComplianceReporter

db_path = Path(tempfile.mkdtemp()) / "demo_audit.db"
log = AuditLog(db_path)

now = time.time()

# Simulate 30 days of production traffic across two models
for day in range(30):
    ts = now - (30 - day) * 86400
    for _ in range(20):
        # GPT-4o: 90% approval rate
        log.log(
            AuditEntry(
                prompt="Customer question",
                response="Product answer",
                model="gpt-4o",
                provider="openai",
                score=0.78 + day * 0.002,
                approved=day % 10 != 0,
                verdict_confidence=0.85,
                task_type="qa",
                domain="support",
                latency_ms=18.0,
                timestamp=ts,
            )
        )
    for _ in range(10):
        # Claude: 95% approval rate
        log.log(
            AuditEntry(
                prompt="Technical question",
                response="Technical answer",
                model="claude-sonnet-4-20250514",
                provider="anthropic",
                score=0.82 + day * 0.001,
                approved=day % 20 != 0,
                verdict_confidence=0.90,
                task_type="qa",
                domain="engineering",
                latency_ms=22.0,
                timestamp=ts,
            )
        )

print(f"Logged {log.count()} audit entries")

# Generate Article 15 report
reporter = ComplianceReporter(log, drift_window_days=7, drift_threshold=0.05)
report: Article15Report = reporter.generate_report(since=now - 30 * 86400)

# Print summary
print(f"\n{'=' * 60}")
print(f"Total interactions: {report.total_interactions:,}")
print(f"Hallucination rate: {report.overall_hallucination_rate:.2%}")
print(f"  95% CI: ± {report.overall_hallucination_rate_ci:.2%}")
print(f"Drift detected: {report.drift_detected}")
print(f"Incidents blocked: {report.incident_count}")

print("\nPer-model breakdown:")
for m in report.model_metrics:
    print(f"  {m.model}: {m.hallucination_rate:.2%} ({m.total_requests} requests)")

# Export Markdown report
md = report.to_markdown()
report_path = Path(tempfile.mkdtemp()) / "article15_report.md"
report_path.write_text(md)
print(f"\nArticle 15 report saved to: {report_path}")
print(f"Report length: {len(md.splitlines())} lines")

log.close()
