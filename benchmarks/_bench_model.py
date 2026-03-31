# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — _bench_model
"""Standalone model benchmarker — called from run_jarvislabs.sh."""

import json
import logging
import pathlib
import sys
import time

import torch

sys.path.insert(0, sys.argv[3])  # WORKDIR
import benchmarks.aggrefact_eval as ae
from benchmarks._load_aggrefact_patch import _load_aggrefact_local

ae._load_aggrefact = _load_aggrefact_local
from benchmarks.aggrefact_eval import _BinaryNLIPredictor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

name = sys.argv[1]
model_path = sys.argv[2]
out_path = sys.argv[4]

rows = _load_aggrefact_local()
pred = _BinaryNLIPredictor(model_name=model_path)
by_ds: dict = {}
t0 = time.perf_counter()
for i, row in enumerate(rows):
    doc, claim = row.get("doc", ""), row.get("claim", "")
    lbl, ds = row.get("label"), row.get("dataset", "unknown")
    if lbl is None or not doc or not claim:
        continue
    prob = pred.score(doc, claim)
    by_ds.setdefault(ds, []).append((int(lbl), float(prob)))
    if (i + 1) % 2000 == 0:
        elapsed = time.perf_counter() - t0
        eta = (len(rows) - i - 1) * elapsed / (i + 1) / 60
        logging.info("%s: %d/%d (%.0f min remaining)", name, i + 1, len(rows), eta)

pathlib.Path(out_path).write_text(json.dumps(by_ds))
elapsed = time.perf_counter() - t0
logging.info("Saved %s (%.1f min)", name, elapsed / 60)
del pred.model, pred
torch.cuda.empty_cache()
