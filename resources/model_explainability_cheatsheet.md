# Model Explainability Cheatsheet

Comprehensive quick reference for explaining and interpreting ML models.

## Table of Contents

- [SHAP (SHapley Additive exPlanations)](#shap-shapley-additive-explanations)
- [LIME (Local Interpretable Model-agnostic Explanations)](#lime-local-interpretable-model-agnostic-explanations)
- [Feature Importance](#feature-importance)
- [Partial Dependence Plots](#partial-dependence-plots)
- [Permutation Importance](#permutation-importance)
- [Model-Specific Methods](#model-specific-methods)

---

## SHAP (SHapley Additive exPlanations)

### Tree Models

```python
import shap

# Tree explainer (fast for tree models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Waterfall plot (single prediction)
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0]
))

# Force plot (single prediction)
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0]
)
```

### Kernel SHAP (Model-agnostic)

```python
import shap

# Kernel explainer (works for any model)
explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
shap_values = explainer.shap_values(X_test.iloc[0])

# Summary plot
shap.summary_plot(shap_values, X_test)
```

### Deep Learning Models

```python
import shap

# Deep explainer
explainer = shap.DeepExplainer(model, X_train[:100])
shap_values = explainer.shap_values(X_test[:10])

# Summary plot
shap.summary_plot(shap_values, X_test[:10])
```

### SHAP Values Interpretation

```python
# SHAP values show feature contributions
# Positive value: increases prediction
# Negative value: decreases prediction
# Magnitude: strength of contribution

# Expected value (baseline)
baseline = explainer.expected_value

# Prediction = baseline + sum of SHAP values
prediction = baseline + shap_values.sum()
```

---

## LIME (Local Interpretable Model-agnostic Explanations)

### Tabular Data

```python
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer

# Create explainer
explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

# Explain single instance
explanation = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)

# Show explanation
explanation.show_in_notebook(show_table=True)

# Get explanation as list
explanation.as_list()
```

### Text Data

```python
from lime import lime_text
from lime.lime_text import LimeTextExplainer

# Create explainer
explainer = LimeTextExplainer(class_names=class_names)

# Explain single text
explanation = explainer.explain_instance(
    text,
    model.predict_proba,
    num_features=10
)

# Show explanation
explanation.show_in_notebook()
```

### Image Data

```python
from lime import lime_image
from lime.lime_image import LimeImageExplainer

# Create explainer
explainer = LimeImageExplainer()

# Explain single image
explanation = explainer.explain_instance(
    image,
    model.predict_proba,
    top_labels=5,
    hide_color=0,
    num_samples=1000
)

# Get explanation
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=True
)
```

---

## Feature Importance

### Tree-Based Models

```python
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Feature importance
importance = model.feature_importances_

# Create DataFrame
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

# Plot
import matplotlib.pyplot as plt
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
```

### Permutation Importance

```python
from sklearn.inspection import permutation_importance

# Calculate permutation importance
perm_importance = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42
)

# Get importance
importance = perm_importance.importances_mean

# Plot
plt.barh(feature_names, importance)
plt.xlabel('Permutation Importance')
plt.title('Feature Importance (Permutation)')
plt.show()
```

---

## Partial Dependence Plots

### Single Feature

```python
from sklearn.inspection import PartialDependenceDisplay

# Create PDP
PartialDependenceDisplay.from_estimator(
    model, X_train, features=[0],
    feature_names=feature_names
)
plt.show()
```

### Two Features

```python
# Interaction plot
PartialDependenceDisplay.from_estimator(
    model, X_train, features=[(0, 1)],
    feature_names=feature_names
)
plt.show()
```

### Multiple Features

```python
# Multiple features
PartialDependenceDisplay.from_estimator(
    model, X_train, features=[0, 1, 2],
    feature_names=feature_names
)
plt.show()
```

---

## Model-Specific Methods

### Linear Models

```python
from sklearn.linear_model import LogisticRegression

# Coefficients show feature importance
model = LogisticRegression()
model.fit(X_train, y_train)

# Coefficients
coefficients = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

# Positive coefficient: increases probability
# Negative coefficient: decreases probability
```

### Neural Networks

```python
import shap

# Deep explainer for neural networks
explainer = shap.DeepExplainer(model, X_train[:100])
shap_values = explainer.shap_values(X_test[:10])

# Summary plot
shap.summary_plot(shap_values, X_test[:10])
```

---

## Comparison: SHAP vs LIME

| Aspect | SHAP | LIME |
|--------|------|------|
| **Theoretical Foundation** | Game theory (Shapley values) | Local linear approximation |
| **Global vs Local** | Both | Local only |
| **Consistency** | Yes (additive) | No |
| **Speed** | Fast for trees, slow for others | Fast |
| **Interpretability** | High | High |
| **Model-agnostic** | Yes | Yes |

### When to Use

- **SHAP**: When you need consistent, theoretically grounded explanations
- **LIME**: When you need fast local explanations for any model

---

## Best Practices

### Do's

- Use SHAP for tree models (fast and accurate)
- Use LIME for model-agnostic explanations
- Combine multiple methods for validation
- Explain both global and local behavior
- Visualize explanations clearly
- Document interpretation guidelines
- Validate explanations with domain experts

### Don'ts

- Don't rely on single explanation method
- Don't ignore feature interactions
- Don't over-interpret small SHAP values
- Don't forget to explain preprocessing
- Don't use explanations without context

---

## Quick Reference

### SHAP Explainer Selection

| Model Type | Explainer | Speed |
|------------|-----------|-------|
| **Tree models** | TreeExplainer | Very Fast |
| **Deep learning** | DeepExplainer | Fast |
| **Linear models** | LinearExplainer | Fast |
| **Any model** | KernelExplainer | Slow |

### Common Visualizations

```python
# SHAP
shap.summary_plot(shap_values, X_test)           # Summary
shap.waterfall_plot(explanation)                 # Single prediction
shap.force_plot(base_value, shap_values, data)   # Force diagram
shap.dependence_plot(0, shap_values, X_test)     # Dependence plot

# LIME
explanation.show_in_notebook()                   # Interactive
explanation.as_list()                            # List format
explanation.as_pyplot_figure()                   # Matplotlib figure
```

### Interpretation Guide

| SHAP Value | Meaning |
|------------|---------|
| **Positive** | Feature increases prediction |
| **Negative** | Feature decreases prediction |
| **Large magnitude** | Strong influence |
| **Small magnitude** | Weak influence |
| **Zero** | No influence |

---

## Code Templates

### Complete SHAP Workflow

```python
import shap
import pandas as pd
import matplotlib.pyplot as plt

# 1. Create explainer
explainer = shap.TreeExplainer(model)

# 2. Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# 3. Visualize
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# 4. Explain single prediction
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=feature_names
))
```

### Complete LIME Workflow

```python
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer

# 1. Create explainer
explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

# 2. Explain instance
explanation = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)

# 3. Visualize
explanation.show_in_notebook(show_table=True)
```

---

**Remember**: Explainability builds trust and helps debug models. Use multiple methods for comprehensive understanding!

