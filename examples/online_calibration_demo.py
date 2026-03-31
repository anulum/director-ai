# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Online calibration demo — improve the guardrail from production feedback."""

import tempfile
from pathlib import Path

from director_ai import CalibrationReport, FeedbackStore, OnlineCalibrator

# Create a temporary database for this demo
db_path = Path(tempfile.mkdtemp()) / "demo_feedback.db"
store = FeedbackStore(db_path)

# Simulate collecting feedback over time
corrections = [
    # (prompt, response, guardrail_approved, human_approved, score, domain)
    ("What is our policy?", "30-day returns.", True, True, 0.82, "support"),
    ("Refund timeline?", "60-day refund.", True, False, 0.61, "support"),
    ("Price of widget?", "$29.99.", True, True, 0.88, "sales"),
    ("Is widget in stock?", "Yes, 1000 units.", True, True, 0.75, "sales"),
    ("Warranty terms?", "Lifetime warranty.", True, False, 0.55, "support"),
    ("Return after 90 days?", "No returns.", False, False, 0.28, "support"),
    ("Shipping cost?", "Free worldwide.", True, False, 0.52, "sales"),
    ("Product materials?", "100% organic.", True, True, 0.91, "sales"),
    ("Cancel order?", "Cannot cancel.", False, True, 0.42, "support"),
    ("Delivery time?", "Same day.", True, False, 0.58, "sales"),
    ("Size options?", "S, M, L, XL.", True, True, 0.85, "sales"),
    ("Color options?", "Red, blue.", True, True, 0.79, "sales"),
    ("Gift wrapping?", "Available.", True, True, 0.72, "sales"),
    ("Bulk discount?", "10% off 100+.", True, True, 0.81, "sales"),
    ("International ship?", "US only.", True, True, 0.77, "sales"),
    ("Track order?", "Email link.", True, True, 0.83, "support"),
    ("Change address?", "Before shipping.", True, True, 0.76, "support"),
    ("Payment methods?", "Card only.", True, True, 0.80, "sales"),
    ("Crypto accepted?", "Yes, Bitcoin.", True, False, 0.53, "sales"),
    ("Store locations?", "Online only.", True, True, 0.87, "sales"),
    ("Phone support?", "24/7 hotline.", True, False, 0.56, "support"),
    ("Email support?", "Within 24h.", True, True, 0.74, "support"),
    ("Live chat?", "Available.", True, True, 0.78, "support"),
    ("FAQ page?", "See website.", True, True, 0.71, "support"),
    ("Privacy policy?", "GDPR compliant.", True, True, 0.84, "support"),
]

for prompt, response, g_approved, h_approved, score, domain in corrections:
    store.report(prompt, response, g_approved, h_approved, score, domain)

print(f"Total corrections: {store.count()}")
print(f"Disagreements: {len(store.get_disagreements())}")

# Calibrate
calibrator = OnlineCalibrator(store, min_corrections=20)
report: CalibrationReport = calibrator.calibrate()

print("\n--- Calibration Report ---")
print(f"Corrections: {report.correction_count}")
print(f"Accuracy: {report.current_accuracy:.1%}")
print(f"TPR: {report.tpr:.3f}")
print(f"TNR: {report.tnr:.3f}")
print(f"FPR: {report.fpr:.3f} ± {report.fpr_ci:.3f}")
print(f"FNR: {report.fnr:.3f} ± {report.fnr_ci:.3f}")
if report.optimal_threshold is not None:
    print(f"Optimal threshold: {report.optimal_threshold}")

# Per-domain calibration
print("\n--- Support Domain ---")
support = calibrator.calibrate(domain="support")
print(f"FPR: {support.fpr:.3f}, FNR: {support.fnr:.3f}")

print("\n--- Sales Domain ---")
sales = calibrator.calibrate(domain="sales")
print(f"FPR: {sales.fpr:.3f}, FNR: {sales.fnr:.3f}")

# Export training data
data = store.export_training_data()
print(f"\nExported {len(data)} training examples")
print(f"First: {data[0]}")

store.close()
print(f"\nDatabase at: {db_path}")
