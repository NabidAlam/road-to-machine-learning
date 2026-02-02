# Advanced Model Explainability Topics

Advanced explainability techniques.

## Table of Contents

- [SHAP Advanced Features](#shap-advanced-features)
- [LIME for Different Data Types](#lime-for-different-data-types)
- [Model-Specific Explanations](#model-specific-explanations)
- [Common Pitfalls](#common-pitfalls)

---

## SHAP Advanced Features

### Interaction Values

```python
import shap

# SHAP interaction values
explainer = shap.TreeExplainer(model)
shap_interaction_values = explainer.shap_interaction_values(X_test[:10])

# Visualize interactions
shap.summary_plot(shap_interaction_values, X_test[:10])
```

---

## LIME for Different Data Types

### Text Data

```python
from lime import lime_text
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
explanation = explainer.explain_instance(text, model.predict_proba)
```

---

## Model-Specific Explanations

Whenever possible, use model-specific explanations first. They’re usually faster and easier to interpret.

### Linear models (coefficients)

For standardized features, coefficients show direction and strength:

```python
import numpy as np

coef = model.coef_.ravel()
top = np.argsort(np.abs(coef))[::-1][:10]
top_features = [(feature_names[i], coef[i]) for i in top]
```

### Tree-based models (feature importance)

```python
import numpy as np

imp = model.feature_importances_
top = np.argsort(imp)[::-1][:10]
top_features = [(feature_names[i], imp[i]) for i in top]
```

### Permutation importance (model-agnostic, strong baseline)

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42)
```

---

## Common Pitfalls

- **Explaining leakage**: if your model leaks future information, explanations are meaningless
- **Correlation confusion**: SHAP/importance can split credit across correlated features unpredictably
- **Causality mistake**: explanations describe the model, not the real world cause-effect
- **Over-trusting local explanations**: LIME/SHAP for one row can be unstable
- **Not validating explanations**: sanity-check with ablation (remove a feature and re-evaluate)

---

## Key Takeaways

1. **Advanced SHAP**: Interaction values, global explanations
2. **Different Data Types**: Text, images, tabular
3. **Model-Specific**: Use built-in methods when available

---

**Remember**: Advanced techniques provide deeper insights!

