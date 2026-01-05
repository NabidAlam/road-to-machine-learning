# Model Explainability Project Tutorial

Step-by-step explainability project.

## Project: Explain Credit Scoring

### Step 1: Train Model

```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

### Step 2: SHAP Explanation

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

### Step 3: LIME Explanation

```python
from lime import lime_tabular
explainer = LimeTabularExplainer(X_train.values, feature_names=feature_names)
explanation = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
explanation.show_in_notebook()
```

---

**Congratulations!** You've explained your model!

