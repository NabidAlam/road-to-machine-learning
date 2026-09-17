# Imbalanced Data Project Tutorial

Step-by-step tutorial for handling imbalanced classification on fraud-style data.

## Project: Fraud Detection

### Step 1: Load Data and Analyze Imbalance

```python
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Example: replace with your local fraud table when you have one.
# For demo, simulate imbalance:
import numpy as np
rng = np.random.default_rng(42)
X = rng.normal(size=(1000, 5))
y = np.array([0] * 980 + [1] * 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Class counts (train):", Counter(y_train))
```

### Step 2: Baseline Model (No Resampling)

```python
baseline = RandomForestClassifier(random_state=42)
baseline.fit(X_train, y_train)
pred_b = baseline.predict(X_test)
proba_b = baseline.predict_proba(X_test)[:, 1]

print("Baseline report:\n", classification_report(y_test, pred_b))
print("Baseline ROC-AUC:", round(roc_auc_score(y_test, proba_b), 4))
```

High accuracy with poor recall on the minority class is common here.

### Step 3: Apply SMOTE

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
print("After SMOTE:", Counter(y_res))
```

### Step 4: Train Balanced Model and Compare

```python
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_res, y_res)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print("SMOTE + balanced RF report:\n", classification_report(y_test, pred))
print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
```

Compare precision and recall on class `1` against the baseline.

---

## Try next

- Try `class_weight="balanced"` without SMOTE and compare
- Plot PR curve for the minority class
- Open [imbalanced-data.md](imbalanced-data.md) for cost-sensitive metrics
