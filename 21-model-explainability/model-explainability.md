# Model Explainability Complete Guide

Comprehensive guide to understanding and explaining machine learning models.

## Table of Contents

- [Introduction to Model Explainability](#introduction-to-model-explainability)
- [Feature Importance](#feature-importance)
- [SHAP (SHapley Additive exPlanations)](#shap-shapley-additive-explanations)
- [LIME (Local Interpretable Model-agnostic Explanations)](#lime-local-interpretable-model-agnostic-explanations)
- [Partial Dependence Plots](#partial-dependence-plots)
- [Best Practices](#best-practices)
- [Practice Exercises](#practice-exercises)

---

## Introduction to Model Explainability

### Why Explainability Matters

**Reasons:**
- **Trust**: Users need to understand predictions
- **Debugging**: Find model errors
- **Compliance**: Regulatory requirements (GDPR, etc.)
- **Fairness**: Detect bias and discrimination
- **Improvement**: Understand what model learns

### Types of Explainability

1. **Global**: Understand model behavior overall
2. **Local**: Understand individual predictions
3. **Model-Specific**: Techniques for specific models
4. **Model-Agnostic**: Work with any model

---

## Feature Importance

### Tree-Based Models

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature importance
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance)

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(importance['feature'], importance['importance'])
plt.xlabel('Importance', fontsize=12)
plt.title('Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Permutation Importance

Model-agnostic feature importance.

```python
from sklearn.inspection import permutation_importance

# Permutation importance
perm_importance = permutation_importance(
    model, X_test, y_test, 
    n_repeats=10, 
    random_state=42
)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print(importance_df)
```

---

## SHAP (SHapley Additive exPlanations)

SHAP values explain individual predictions.

### Tree SHAP

```python
try:
    import shap
    
    # Tree explainer (for tree-based models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    
    # Waterfall plot for single prediction
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0][0],
            base_values=explainer.expected_value,
            data=X_test.iloc[0],
            feature_names=feature_names
        )
    )
    
except ImportError:
    print("Install SHAP: pip install shap")
```

### Kernel SHAP

Model-agnostic SHAP.

```python
import shap

# Kernel explainer (works with any model)
explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
shap_values = explainer.shap_values(X_test[0:5])

# Force plot
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][0],
    X_test.iloc[0],
    feature_names=feature_names
)
```

---

## LIME (Local Interpretable Model-agnostic Explanations)

Explains individual predictions locally.

```python
try:
    from lime import lime_tabular
    from lime.lime_tabular import LimeTabularExplainer
    
    # Create explainer
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=['Class 0', 'Class 1'],
        mode='classification'
    )
    
    # Explain single prediction
    explanation = explainer.explain_instance(
        X_test.iloc[0].values,
        model.predict_proba,
        num_features=10
    )
    
    # Show explanation
    explanation.show_in_notebook(show_table=True)
    
    # Get explanation as list
    explanation_list = explanation.as_list()
    print("\nFeature Contributions:")
    for feature, value in explanation_list:
        print(f"  {feature}: {value:.4f}")
    
except ImportError:
    print("Install LIME: pip install lime")
```

---

## Partial Dependence Plots

Understand feature effects.

```python
from sklearn.inspection import PartialDependenceDisplay

# Partial dependence plot
PartialDependenceDisplay.from_estimator(
    model, X_train, features=[0, 1],
    feature_names=feature_names,
    grid_resolution=20
)
plt.tight_layout()
plt.show()
```

---

## Best Practices

### When to Use What

| Technique | Use Case | Pros | Cons |
|-----------|----------|------|------|
| **Feature Importance** | Tree models | Fast, built-in | Model-specific |
| **SHAP** | Any model | Unified framework | Computationally expensive |
| **LIME** | Local explanations | Model-agnostic | Approximate |
| **PDP** | Feature effects | Visual, intuitive | Assumes independence |

### Workflow

1. **Start simple**: Feature importance for tree models
2. **Use SHAP**: For comprehensive explanations
3. **Use LIME**: For local explanations
4. **Visualize**: Plots help understanding
5. **Document**: Explain findings clearly

---

## Practice Exercises

### Exercise 1: Explain Model Predictions

**Task:** Use SHAP and LIME to explain predictions.

**Solution:**
```python
# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# SHAP explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:10])
shap.summary_plot(shap_values, X_test[:10])

# LIME explanation
lime_explainer = LimeTabularExplainer(X_train.values, feature_names=feature_names)
explanation = lime_explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
explanation.show_in_notebook()
```

---

## Key Takeaways

1. **Explainability is crucial**: Especially for production models
2. **Multiple techniques**: Use appropriate method for your needs
3. **SHAP is powerful**: Unified framework for explanations
4. **LIME for local**: Great for individual predictions
5. **Visualize**: Plots help communicate findings

---

**Remember**: Explainable models build trust and enable debugging!

