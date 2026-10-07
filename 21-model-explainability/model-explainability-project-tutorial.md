# Model Explainability Project Tutorial

Explain a small tabular classifier with built-in importances first. Add SHAP or LIME only when those packages are installed.

## Project: Explain a credit-style score

### Step 1: Train a readable baseline

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.inspection import permutation_importance

X_np, y = make_classification(
    n_samples=800, n_features=8, n_informative=5, random_state=42
)
feature_names = [f"feature_{i}" for i in range(X_np.shape[1])]
X = pd.DataFrame(X_np, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("test accuracy:", round(model.score(X_test, y_test), 4))
```

### Step 2: Impurity-based importances (fast sanity check)

```python
imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(
    ascending=False
)
print(imp.head())
assert imp.sum() > 0
```

These scores can favor high-cardinality features. Treat them as a first pass, not the final story.

### Step 3: Permutation importance on the holdout fold

```python
perm = permutation_importance(
    model, X_test, y_test, n_repeats=8, random_state=42, scoring="accuracy"
)
perm_s = pd.Series(perm.importances_mean, index=feature_names).sort_values(
    ascending=False
)
print(perm_s.head())
```

Permutation importance asks how much score drops when a column is shuffled. It uses the real metric you care about.

### Step 4: One local explanation without extra deps

```python
row = X_test.iloc[[0]]
proba = model.predict_proba(row)[0]
print("row 0 proba:", proba)

# Contribution sketch from the tree ensemble path is package-specific.
# A dep-free check: flip each feature toward the train median and watch proba move.
median = X_train.median(numeric_only=True)
deltas = {}
base = float(proba[1])
for col in feature_names:
    tweaked = row.copy()
    tweaked[col] = median[col]
    new_p = float(model.predict_proba(tweaked)[0, 1])
    deltas[col] = base - new_p
delta_s = pd.Series(deltas).sort_values(key=np.abs, ascending=False)
print("largest local moves when replaced by median:\n", delta_s.head())
```

This is a teaching probe, not SHAP. It shows which columns move the score for one row.

### Step 5: Optional SHAP (skip if missing)

```python
try:
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Binary forests often return a list [class0, class1]
    values = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs = np.abs(values).mean(axis=0)
    shap_rank = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    print("SHAP mean |value| rank:\n", shap_rank.head())
except ImportError:
    print("shap not installed. Keep permutation importance as the global view.")
```

### Step 6: Optional LIME on one row (skip if missing)

```python
try:
    from lime.lime_tabular import LimeTabularExplainer

    lime_explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=["reject", "approve"],
        mode="classification",
    )
    exp = lime_explainer.explain_instance(
        X_test.iloc[0].values,
        model.predict_proba,
        num_features=5,
    )
    print(exp.as_list())
except ImportError:
    print("lime not installed. Use the median-tweak local probe above.")
```

Compare the top local drivers across methods. Large disagreement is a signal to dig into correlated features.

### Step 7: Stakeholder note (short)

Write three bullets you could paste into a review:

1. Global top features from permutation importance.
2. One row where the decision flipped after a median tweak.
3. Limits. Correlated inputs can share credit. Importances are not causal effects.

---

## Try next

- Plot partial dependence for the top permutation feature with `PartialDependenceDisplay`.
- Continue to [model-explainability.md](model-explainability.md).
