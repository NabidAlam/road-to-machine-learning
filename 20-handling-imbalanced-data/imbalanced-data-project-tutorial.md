# Imbalanced Data Project Tutorial

Work a fraud-style imbalance end to end on synthetic rows. No downloads. Measure minority recall and PR quality, not accuracy alone.

## Project: Fraud Detection (synthetic)

### Step 1: Build a skewed label set and split with stratify

```python
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

rng = np.random.default_rng(42)
n_maj, n_min = 980, 20
X_maj = rng.normal(size=(n_maj, 5))
X_min = rng.normal(loc=1.2, size=(n_min, 5))
X = np.vstack([X_maj, X_min])
y = np.array([0] * n_maj + [1] * n_min)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("train counts:", Counter(y_train))
print("test counts:", Counter(y_test))
assert y_train.sum() >= 1 and y_test.sum() >= 1
```

Always stratify. A random split can drop every fraud row from the test fold.

### Step 2: Baseline without resampling

```python
baseline = RandomForestClassifier(n_estimators=100, random_state=42)
baseline.fit(X_train, y_train)
pred_b = baseline.predict(X_test)
proba_b = baseline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, pred_b, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, proba_b), 4))
print("PR-AUC:", round(average_precision_score(y_test, proba_b), 4))
print("confusion:\n", confusion_matrix(y_test, pred_b))
```

High accuracy with near-zero recall on class `1` is the trap. Prefer PR-AUC when positives are rare.

### Step 3: Class weight without synthetic rows

```python
weighted = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42
)
weighted.fit(X_train, y_train)
pred_w = weighted.predict(X_test)
proba_w = weighted.predict_proba(X_test)[:, 1]

print("weighted report:\n", classification_report(y_test, pred_w, digits=3))
print("weighted PR-AUC:", round(average_precision_score(y_test, proba_w), 4))
```

This changes the loss weight. It does not invent new minority samples.

### Step 4: Optional SMOTE on the train fold only

Fit any resampler on **train only**. Never touch the test fold.

```python
try:
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print("after SMOTE:", Counter(y_res))

    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    model.fit(X_res, y_res)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print("SMOTE report:\n", classification_report(y_test, pred, digits=3))
    print("SMOTE PR-AUC:", round(average_precision_score(y_test, proba), 4))
except ImportError:
    print("imblearn not installed. Skip SMOTE and keep the weighted model.")
    pred, proba = pred_w, proba_w
```

If `imbalanced-learn` is missing, the weighted model still teaches the evaluation lesson.

### Step 5: Compare on the minority class

```python
def minority_recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return float(tp / (tp + fn)) if (tp + fn) else 0.0


print("baseline minority recall:", round(minority_recall(y_test, pred_b), 3))
print("chosen minority recall:", round(minority_recall(y_test, pred), 3))
```

Write down which lever helped. Weighting, resampling, or a lower decision threshold.

### Step 6: Threshold sweep on validation-style scores

```python
thresholds = np.linspace(0.1, 0.9, 9)
rows = []
for t in thresholds:
    p = (proba >= t).astype(int)
    rows.append((round(float(t), 2), round(minority_recall(y_test, p), 3)))
print("threshold -> minority recall:", rows)
```

Pick a threshold from a validation fold in real work. Here the test fold is only for teaching the curve shape.

---

## Try next

- Plot a precision-recall curve for the minority class.
- Try `class_weight="balanced"` without SMOTE and compare PR-AUC.
- Open [imbalanced-data.md](imbalanced-data.md) for cost-sensitive metrics.
