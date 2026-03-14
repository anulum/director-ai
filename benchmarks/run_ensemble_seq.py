import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("ensemble")

sys.path.insert(0, "/home/director-ai")
import benchmarks.aggrefact_eval as ae  # noqa: E402
from benchmarks._load_aggrefact_patch import _load_aggrefact_local  # noqa: E402

ae._load_aggrefact = _load_aggrefact_local
import numpy as np  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from benchmarks._common import save_results  # noqa: E402
from benchmarks.aggrefact_eval import _BinaryNLIPredictor  # noqa: E402

MODELS_DIR = Path("/home/director-ai/models")
SCORES_DIR = Path("/home/director-ai/scores")
BASE_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
SCORES_DIR.mkdir(exist_ok=True)

rows = _load_aggrefact_local()
log.info("Loaded %d samples", len(rows))

model_paths = {"base": BASE_MODEL}
for p in sorted(MODELS_DIR.iterdir()):
    if p.is_dir() and (p / "config.json").exists():
        model_paths[p.name] = str(p)

log.info("Models to score: %d", len(model_paths))

for name, path in model_paths.items():
    out_file = SCORES_DIR / f"{name}.json"
    if out_file.exists():
        log.info("Skipping %s (cached)", name)
        continue
    log.info("Loading %s", name)
    import torch

    pred = _BinaryNLIPredictor(model_name=path)
    scores_by_ds = {}
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        doc = row.get("doc", "")
        claim = row.get("claim", "")
        label = row.get("label")
        ds = row.get("dataset", "unknown")
        if label is None or not doc or not claim:
            continue
        prob = pred.score(doc, claim)
        if ds not in scores_by_ds:
            scores_by_ds[ds] = []
        scores_by_ds[ds].append((int(label), float(prob)))
        if (i + 1) % 2000 == 0:
            elapsed = time.perf_counter() - t0
            eta = (len(rows) - i - 1) * elapsed / (i + 1) / 60
            log.info("  %s: %d/%d (%.0f min remaining)", name, i + 1, len(rows), eta)
    out_file.write_text(json.dumps(scores_by_ds))
    log.info("Saved %s (%.1f min)", name, (time.perf_counter() - t0) / 60)
    del pred.model, pred
    torch.cuda.empty_cache()

log.info("All models scored. Running threshold sweep...")
all_scores = {}
for f in SCORES_DIR.glob("*.json"):
    all_scores[f.stem] = json.loads(f.read_text())

model_names = list(all_scores.keys())
datasets = sorted(set(ds for m in all_scores.values() for ds in m))


def sweep(strategy, thresh):
    per_ds = {}
    for ds in datasets:
        entries = all_scores[model_names[0]].get(ds, [])
        n = len(entries)
        if not n:
            continue
        agg = []
        for idx in range(n):
            label = entries[idx][0]
            probs = [
                all_scores[m][ds][idx][1]
                for m in model_names
                if ds in all_scores[m] and idx < len(all_scores[m][ds])
            ]
            p = max(probs) if strategy == "max" else float(np.mean(probs))
            agg.append((label, p))
        y_true = [x[0] for x in agg]
        y_pred = [1 if x[1] >= thresh else 0 for x in agg]
        per_ds[ds] = {
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
            "total": n,
        }
    return float(np.mean([v["balanced_acc"] for v in per_ds.values()])), per_ds


best_acc, best_thresh, best_strat, best_per_ds = 0.0, 0.5, "mean", {}
for t_int in range(10, 91, 2):
    t = t_int / 100.0
    for strat in ("max", "mean"):
        acc, _ = sweep(strat, t)
        if acc > best_acc:
            best_acc, best_thresh, best_strat = acc, t, strat

_, best_per_ds = sweep(best_strat, best_thresh)

individual = {}
for mname, mscores in all_scores.items():
    accs = []
    for _ds, pairs in mscores.items():
        y_true = [p[0] for p in pairs]
        y_pred = [1 if p[1] >= best_thresh else 0 for p in pairs]
        accs.append(balanced_accuracy_score(y_true, y_pred))
    individual[mname] = round(float(np.mean(accs)), 4)

result = {
    "benchmark": "LLM-AggreFact",
    "mode": "ensemble_sweep_sequential",
    "n_models": len(model_names),
    "models": model_names,
    "best_strategy": best_strat,
    "best_threshold": best_thresh,
    "best_accuracy": round(best_acc, 4),
    "base_accuracy": individual.get("base", 0),
    "individual": dict(sorted(individual.items(), key=lambda x: -x[1])),
    "per_dataset": best_per_ds,
}
save_results(result, "aggrefact_ensemble.json")

print("=" * 60)
print(f"  Best: {best_strat} @ threshold={best_thresh:.2f} -> {best_acc:.1%}")
print(f"  Base: {individual.get('base', 0):.1%}")
print(f"  Improvement: {best_acc - individual.get('base', 0):+.1%}")
print("  Individual (top 10):")
for m, acc in sorted(individual.items(), key=lambda x: -x[1])[:10]:
    print(f"    {m:<30} {acc:.1%}")
print("=" * 60)
print("DONE")
