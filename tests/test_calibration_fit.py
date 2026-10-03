# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — calibration fit benchmark tests

"""Exercise seeded calibration fitting and its existing campaign CLI outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import calibration_fit


def _campaign(path: Path, n_per_class: int = 400) -> str:
    """Write the seeded score campaign used by the calibration fitting tests.

    Parameters
    ----------
    path : pathlib.Path
        Test directory receiving the campaign JSON file.
    n_per_class : int
        Number of seeded examples for each right and hallucinated class.

    Returns
    -------
    str
        Path to the written campaign artefact.
    """
    rng = np.random.default_rng(7)
    good = np.clip(rng.normal(0.82, 0.12, n_per_class), 0, 1)
    bad = np.clip(rng.normal(0.18, 0.12, n_per_class), 0, 1)
    rows = [{"coherence": float(c), "label": "right"} for c in good]
    rows += [{"coherence": float(c), "label": "hallucinated"} for c in bad]
    payload = {"campaign": "test", "rows": rows}
    dest = path / "campaign.json"
    dest.write_text(json.dumps(payload), encoding="utf-8")
    return str(dest)


class TestLoadCampaign:
    """Check score and ground-truth extraction from calibration campaigns."""

    def test_maps_grounded_labels_to_binary(self, tmp_path: Path) -> None:
        """Map right, hallucinated and case-insensitive correct labels to binary values."""
        payload = {
            "rows": [
                {"coherence": 0.9, "label": "right"},
                {"coherence": 0.1, "label": "hallucinated"},
                {"coherence": 0.8, "label": "CORRECT"},
            ]
        }
        path = tmp_path / "c.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        scores, labels = calibration_fit._load_campaign_rows(path)
        assert scores == [0.9, 0.1, 0.8]
        assert labels == [1, 0, 1]

    def test_rejects_artefact_without_rows(self, tmp_path: Path) -> None:
        """Reject a campaign that contains no per-sample calibration rows."""
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"rows": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="no per-sample rows"):
            calibration_fit._load_campaign_rows(path)


class TestFitCalibration:
    """Check fitted maps, sample rows and explicit provenance."""

    def test_fit_reduces_ece_and_emits_maps_rows_and_provenance(
        self, tmp_path: Path
    ) -> None:
        """Fit both maps and retain all seeded rows, error metrics and commit provenance."""
        campaign = _campaign(tmp_path)
        payload = calibration_fit.fit_calibration(campaign, git_sha="a" * 40)

        assert payload["schema_version"] == calibration_fit.SCHEMA_VERSION
        assert payload["n_samples"] == 800
        # Both calibrators reduce ECE below the raw score's.
        assert payload["isotonic"]["ece"] <= payload["raw"]["ece"]
        assert payload["platt"]["ece"] <= payload["raw"]["ece"]
        # Maps are serialised for re-load.
        assert payload["isotonic"]["x_thresholds"]
        assert "a" in payload["platt"] and "b" in payload["platt"]
        # Per-sample rows carry raw + both calibrated probabilities.
        assert len(payload["rows"]) == 800
        assert set(payload["rows"][0]) == {"coherence", "grounded", "isotonic", "platt"}
        # Provenance stamp records the supplied commit.
        assert payload["provenance"]["git_sha"] == "a" * 40


class TestMain:
    """Check explicit and default calibration CLI output destinations."""

    def test_main_writes_explicit_out_path(self, tmp_path: Path) -> None:
        """Write the fitted campaign and supplied commit to the requested output file."""
        campaign = _campaign(tmp_path)
        out = tmp_path / "fit.json"
        rc = calibration_fit.main(
            ["--campaign", campaign, "--out", str(out), "--git-sha", "b" * 40]
        )
        assert rc == 0
        written = json.loads(out.read_text())
        assert written["provenance"]["git_sha"] == "b" * 40
        assert written["n_samples"] == 800

    def test_main_default_path_uses_save_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pass the fitted sample count to the default benchmark result writer."""
        campaign = _campaign(tmp_path)
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            calibration_fit,
            "save_results",
            lambda payload, filename: captured.update(
                filename=filename, n=payload["n_samples"]
            ),
        )
        assert calibration_fit.main(["--campaign", campaign]) == 0
        assert captured["filename"] == "calibration_fit.json"
        assert captured["n"] == 800
