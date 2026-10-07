# Complete MLOps Project Tutorial

Walk through a small, reproducible training loop you can version, track, and test without a cloud account. Everything below uses a tiny synthetic table so nothing is downloaded.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Freeze a reproducible train script](#step-1-freeze-a-reproducible-train-script)
- [Step 2: Log params and metrics to a local file](#step-2-log-params-and-metrics-to-a-local-file)
- [Step 3: Pin data and code versions](#step-3-pin-data-and-code-versions)
- [Step 4: Add a smoke test in CI shape](#step-4-add-a-smoke-test-in-ci-shape)
- [Step 5: Register a model artifact path](#step-5-register-a-model-artifact-path)
- [Step 6: Sketch a promote-or-reject gate](#step-6-sketch-a-promote-or-reject-gate)

---

## Project Overview

**Goal.** Train a tiny classifier, write metrics to disk, and keep the loop honest enough that a later DVC or MLflow layer has something real to wrap.

**What you will produce.**

1. A `train.py`-style function that returns metrics.
2. A JSON run log (params + metrics + artifact path).
3. A pytest-shaped smoke check you can paste into CI.

---

## Step 1: Freeze a reproducible train script

```python
from pathlib import Path
import json
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib


def build_dataset(seed=42):
    X, y = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=seed,
    )
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)


def train_and_eval(C=1.0, seed=42, artifact_dir="artifacts"):
    X_train, X_test, y_train, y_test = build_dataset(seed=seed)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=500, random_state=seed)),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
    }
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.joblib"
    joblib.dump(pipe, model_path)
    return metrics, str(model_path)


metrics, model_path = train_and_eval()
assert metrics["accuracy"] >= 0.75
print(metrics, model_path)
```

Keep the seed and the split logic in one place. Changing either later must bump a version note.

---

## Step 2: Log params and metrics to a local file

Before you install MLflow, write a JSON run record. The shape matches what most trackers expect later.

```python
from datetime import datetime, timezone
from pathlib import Path
import json

params = {"C": 1.0, "seed": 42, "model": "logreg_pipeline"}
metrics, model_path = train_and_eval(C=params["C"], seed=params["seed"])

run = {
    "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    "params": params,
    "metrics": metrics,
    "artifacts": {"model": model_path},
}
Path("runs").mkdir(exist_ok=True)
run_path = Path("runs") / f"{run['run_id']}.json"
run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
print("wrote", run_path)
```

When you add MLflow later, map `params` and `metrics` 1:1. Do not invent a second logging schema.

---

## Step 3: Pin data and code versions

Even without DVC you can record hashes so a teammate can spot drift.

```python
import hashlib
import json
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Stand-in for a frozen training CSV. Replace with your real file bytes later.
payload = b"synthetic-v1|n_samples=400|seed=42"
data_hash = sha256_bytes(payload)
code_hash = sha256_bytes(Path(__file__).read_bytes()) if "__file__" in globals() else "inline"

manifest = {
    "data_hash": data_hash,
    "code_hash": code_hash,
    "framework": "sklearn",
}
Path("runs/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(manifest)
```

DVC would replace the manual hash step with `dvc add` and a pipeline stage. The habit is the same. Pin inputs before you trust a metric.

---

## Step 4: Add a smoke test in CI shape

```python
def test_train_smoke():
    metrics, model_path = train_and_eval(C=1.0, seed=0, artifact_dir="artifacts_ci")
    assert metrics["accuracy"] >= 0.7
    assert metrics["f1"] >= 0.7
    assert Path(model_path).exists()


test_train_smoke()
print("ci smoke OK")
```

Wire the same function into GitHub Actions with `pytest` when you are ready. Keep the threshold modest. The job is "did training break", not "did we beat last week's champion".

Example workflow sketch (do not paste secrets):

```yaml
name: ml-smoke
on: [push]
jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install scikit-learn joblib pytest
      - run: pytest -q
```

---

## Step 5: Register a model artifact path

A registry entry is just a pointer plus a stage. Start local.

```python
registry = {
    "name": "toy_logreg",
    "version": 1,
    "stage": "Staging",
    "artifact": model_path,
    "metrics": metrics,
}
Path("runs/registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
print(registry)
```

MLflow Model Registry later wraps the same idea with `register_model` and stage transitions. Your local JSON keeps the mental model clear.

---

## Step 6: Sketch a promote-or-reject gate

```python
def promote_or_reject(metrics, min_acc=0.8, min_f1=0.75):
    if metrics["accuracy"] >= min_acc and metrics["f1"] >= min_f1:
        return "Production"
    return "Rejected"


stage = promote_or_reject(metrics)
print("next stage:", stage)
```

Only promote when both gates pass. Log the decision next to the run id so you can audit why a model moved.

---

## Try next

- Swap the JSON logger for a local MLflow tracking URI when you install it.
- Add a second param (`C=0.1` vs `C=1.0`) and keep both run files.
- Continue to [Module 15 · Time series](../15-time-series-analysis/README.md) or a beginner project under Module 16.
