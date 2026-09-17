# Model Explainability Project Tutorial

Step-by-step explainability project for a tabular classifier.

## Project: Explain Credit Scoring

### Step 1: Train Model

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=800, n_features=8, n_informative=5, random_state=42
)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Test accuracy:", round(model.score(X_test, y_test), 4))
```

### Step 2: SHAP Global Explanation

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classification, class 1 is often index 1
shap.summary_plot(shap_values[1], X_test, show=False)
```

Review which features push predictions toward approval vs rejection.

### Step 3: SHAP Local Explanation

```python
import shap

sample_idx = 0
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][sample_idx],
    X_test.iloc[sample_idx],
    matplotlib=True,
    show=False,
)
```

### Step 4: LIME Local Explanation

```python snippet-skip
from lime.lime_tabular import LimeTabularExplainer

# Optional local dep (lime); skip in CI when not installed
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
exp.show_in_notebook()  # or exp.as_list() in scripts
```

Compare SHAP and LIME on the same row. They should highlight similar drivers.

---

## Try next

- Plot partial dependence for the top SHAP feature
- Document explanations for a stakeholder audience
- Continue to [model-explainability.md](model-explainability.md)
