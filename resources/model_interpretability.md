# Model Interpretability & Explainability

Comprehensive guide to understanding, interpreting, and explaining machine learning model predictions.

## Table of Contents

- [Introduction](#introduction)
- [Why Interpretability Matters](#why-interpretability-matters)
- [Types of Interpretability](#types-of-interpretability)
- [Global vs Local Interpretability](#global-vs-local-interpretability)
- [Interpretability Methods](#interpretability-methods)
- [Tools and Libraries](#tools-and-libraries)
- [Best Practices](#best-practices)
- [Resources](#resources)

---

## Introduction

**Model Interpretability** refers to the ability to understand and explain how a machine learning model makes predictions. As ML models become more complex and are used in critical applications (healthcare, finance, legal), understanding model decisions becomes essential.

### Key Terms

- **Interpretability**: Ability to understand model behavior
- **Explainability**: Ability to explain model predictions to humans
- **Transparency**: Model's internal workings are visible
- **Black Box**: Model whose internal logic is not easily understood

---

## Why Interpretability Matters

### 1. Trust and Adoption

- Users need to trust model predictions
- Stakeholders require explanations for decisions
- Regulatory compliance (GDPR, AI Act)

### 2. Debugging and Improvement

- Identify model errors and biases
- Understand failure cases
- Improve model performance

### 3. Fairness and Ethics

- Detect discriminatory patterns
- Ensure fair treatment across groups
- Comply with ethical guidelines

### 4. Business Value

- Explain decisions to business stakeholders
- Gain insights from model behavior
- Build confidence in ML systems

### 5. Regulatory Compliance

- GDPR: Right to explanation
- Financial regulations: Explain credit decisions
- Healthcare: Explainability for medical AI

---

## Types of Interpretability

### 1. Intrinsic Interpretability

Models that are inherently interpretable:

- **Linear Models**: Coefficients show feature importance
- **Decision Trees**: Rules are explicit
- **Rule-based Systems**: Clear if-then rules

**Example:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target

model = LogisticRegression()
model.fit(X, y)

# Feature importance from coefficients
feature_importance = abs(model.coef_[0])
feature_names = data.feature_names

# Most important features
top_features = sorted(zip(feature_names, feature_importance), 
                     key=lambda x: x[1], reverse=True)[:5]

for feature, importance in top_features:
    print(f"{feature}: {importance:.4f}")
```

### 2. Post-hoc Interpretability

Methods applied after model training:

- **Feature Importance**: Which features matter most
- **SHAP Values**: Unified measure of feature contribution
- **LIME**: Local interpretable model-agnostic explanations
- **Partial Dependence Plots**: Effect of individual features

---

## Global vs Local Interpretability

### Global Interpretability

Understanding the model's overall behavior:

- Which features are most important globally?
- How does the model work in general?
- What patterns does the model learn?

**Methods:**
- Feature importance
- Partial dependence plots
- Global surrogate models

### Local Interpretability

Understanding individual predictions:

- Why did the model predict this for this specific instance?
- Which features contributed to this prediction?
- What would change the prediction?

**Methods:**
- SHAP values (local)
- LIME
- Counterfactual explanations

---

## Interpretability Methods

### 1. Feature Importance

#### Permutation Importance

Measures how much model performance decreases when a feature is shuffled.

```python
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate permutation importance
perm_importance = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42
)

# Get feature importance
feature_importance = perm_importance.importances_mean
feature_names = data.feature_names

# Sort by importance
sorted_idx = feature_importance.argsort()[::-1]

for idx in sorted_idx[:10]:
    print(f"{feature_names[idx]}: {feature_importance[idx]:.4f}")
```

#### Tree-based Feature Importance

```python
import matplotlib.pyplot as plt
import numpy as np

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot top 10 features
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(10), importances[indices[:10]])
plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=45)
plt.tight_layout()
plt.show()
```

### 2. SHAP (SHapley Additive exPlanations)

Unified framework for explaining model predictions.

#### Installation
```bash
pip install shap
```

#### Tree SHAP (Fast for tree models)
```python
import shap

# Train a tree model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot (global)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Waterfall plot for single prediction
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0][0],
        base_values=explainer.expected_value[0],
        data=X_test[0],
        feature_names=feature_names
    )
)

# Force plot for single prediction
shap.force_plot(
    explainer.expected_value[0],
    shap_values[0][0],
    X_test[0],
    feature_names=feature_names
)
```

#### Kernel SHAP (Model-agnostic)
```python
# Works with any model
explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
shap_values = explainer.shap_values(X_test[0:5])

shap.summary_plot(shap_values, X_test[0:5], feature_names=feature_names)
```

#### SHAP Values Interpretation

- **Positive SHAP value**: Feature pushes prediction higher
- **Negative SHAP value**: Feature pushes prediction lower
- **Magnitude**: How much the feature contributes
- **Sum**: SHAP values sum to prediction - base_value

### 3. LIME (Local Interpretable Model-agnostic Explanations)

Explains individual predictions by approximating the model locally.

#### Installation
```bash
pip install lime
```

#### Example
```python
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer

# Create explainer
explainer = LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=['Benign', 'Malignant'],
    mode='classification'
)

# Explain a single prediction
explanation = explainer.explain_instance(
    X_test[0],
    model.predict_proba,
    num_features=10
)

# Show explanation
explanation.show_in_notebook(show_table=True)

# Get explanation as list
exp_list = explanation.as_list()
for feature, value in exp_list:
    print(f"{feature}: {value:.4f}")
```

### 4. Partial Dependence Plots (PDP)

Show the marginal effect of a feature on predictions.

```python
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create PDP for a feature
features = [0, 1]  # Feature indices
PartialDependenceDisplay.from_estimator(
    model, X_train, features,
    feature_names=feature_names,
    grid_resolution=20
)
plt.show()
```

### 5. Individual Conditional Expectation (ICE) Plots

Show how predictions change for individual instances.

```python
from sklearn.inspection import PartialDependenceDisplay

# ICE plots
PartialDependenceDisplay.from_estimator(
    model, X_train, features,
    kind='individual',
    feature_names=feature_names
)
plt.show()
```

### 6. Counterfactual Explanations

"What would need to change for a different prediction?"

```python
import numpy as np

def generate_counterfactual(model, instance, target_class, feature_names):
    """
    Simple counterfactual generation (conceptual example)
    """
    current_pred = model.predict([instance])[0]
    
    if current_pred == target_class:
        return "Already predicted as target class"
    
    # Find minimal changes (simplified)
    instance_copy = instance.copy()
    feature_importance = model.feature_importances_
    
    # Sort features by importance
    sorted_features = np.argsort(feature_importance)[::-1]
    
    changes = []
    for idx in sorted_features:
        # Try changing this feature
        test_instance = instance_copy.copy()
        test_instance[idx] = np.mean(X_train[:, idx])  # Change to mean
        
        if model.predict([test_instance])[0] == target_class:
            changes.append({
                'feature': feature_names[idx],
                'original': instance[idx],
                'new': test_instance[idx]
            })
            break
    
    return changes

# Example usage
counterfactual = generate_counterfactual(
    model, X_test[0], target_class=1, feature_names=feature_names
)
print(counterfactual)
```

### 7. Attention Visualization (for Neural Networks)

For transformer and attention-based models:

```python
import torch
from transformers import AutoTokenizer, AutoModel

# Load a transformer model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Get attention weights
text = "This is a sample text"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs, output_attentions=True)

# Visualize attention (requires additional libraries)
# Use libraries like bertviz for visualization
```

---

## Tools and Libraries

### 1. SHAP
- **Purpose**: Unified framework for model explanations
- **Installation**: `pip install shap`
- **Documentation**: https://shap.readthedocs.io/
- **Best for**: Tree models, neural networks, any model

### 2. LIME
- **Purpose**: Local explanations for any model
- **Installation**: `pip install lime`
- **Documentation**: https://github.com/marcotcr/lime
- **Best for**: Tabular data, text, images

### 3. ELI5
- **Purpose**: Debug and explain ML models
- **Installation**: `pip install eli5`
- **Documentation**: https://eli5.readthedocs.io/
- **Best for**: Scikit-learn models, text classification

```python
import eli5

# Explain a prediction
eli5.show_prediction(model, X_test[0], feature_names=feature_names)

# Show feature weights
eli5.show_weights(model, feature_names=feature_names)
```

### 4. Yellowbrick
- **Purpose**: Visual analysis and diagnostic tools
- **Installation**: `pip install yellowbrick`
- **Documentation**: https://www.scikit-yb.org/
- **Best for**: Model selection, feature analysis

```python
from yellowbrick.model_selection import FeatureImportances

visualizer = FeatureImportances(model)
visualizer.fit(X_train, y_train)
visualizer.show()
```

### 5. InterpretML
- **Purpose**: Microsoft's interpretability toolkit
- **Installation**: `pip install interpret`
- **Documentation**: https://interpret.ml/
- **Best for**: Explainable boosting machines, glassbox models

### 6. Alibi
- **Purpose**: Algorithms for explaining ML models
- **Installation**: `pip install alibi`
- **Documentation**: https://docs.seldon.io/projects/alibi/
- **Best for**: Counterfactuals, adversarial detection

### 7. Captum (PyTorch)
- **Purpose**: Model interpretability for PyTorch
- **Installation**: `pip install captum`
- **Documentation**: https://captum.ai/
- **Best for**: PyTorch models, neural networks

```python
from captum.attr import IntegratedGradients

# For PyTorch models
ig = IntegratedGradients(model)
attributions = ig.attribute(input_tensor, target=target_class)
```

---

## Best Practices

### 1. Choose the Right Method

- **Tree models**: Use SHAP TreeExplainer (fast)
- **Neural networks**: Use SHAP KernelExplainer or Captum
- **Local explanations**: Use LIME or SHAP
- **Global understanding**: Use feature importance, PDP

### 2. Validate Explanations

- Check if explanations make sense
- Verify with domain experts
- Test on multiple instances
- Compare different explanation methods

### 3. Communicate Clearly

- Use visualizations
- Avoid technical jargon
- Focus on actionable insights
- Provide context

### 4. Consider Your Audience

- **Technical team**: Detailed SHAP values, feature importance
- **Business stakeholders**: High-level summaries, visualizations
- **End users**: Simple, intuitive explanations
- **Regulators**: Comprehensive documentation

### 5. Document Everything

- Which methods you used
- Why you chose them
- Limitations of explanations
- How to interpret results

### 6. Performance Considerations

- SHAP can be slow for large datasets
- Use sampling for large models
- Cache explanations when possible
- Consider approximate methods

---

## Example: Complete Interpretability Workflow

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 1. Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(10), importances[indices[:10]])
plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=45)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# 2. SHAP Values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.savefig('shap_summary.png', bbox_inches='tight')
plt.close()

# 3. Local explanation for one instance
shap.force_plot(
    explainer.expected_value[0],
    shap_values[0][0],
    X_test[0],
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.savefig('shap_local.png', bbox_inches='tight')
plt.close()

# 4. Permutation Importance
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42
)

sorted_idx = perm_importance.importances_mean.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(10), perm_importance.importances_mean[sorted_idx[:10]])
plt.yticks(range(10), [feature_names[i] for i in sorted_idx[:10]])
plt.xlabel('Permutation Importance')
plt.tight_layout()
plt.savefig('permutation_importance.png')
plt.close()

print("Interpretability analysis complete!")
print(f"Top 5 features by importance: {[feature_names[i] for i in indices[:5]]}")
```

---

## Resources

### Books

1. **"Interpretable Machine Learning"** by Christoph Molnar
   - [Online Book](https://christophm.github.io/interpretable-ml-book/)
   - Comprehensive guide to interpretability methods

2. **"Explainable AI: Interpreting, Explaining and Visualizing Deep Learning"**
   - Collection of research papers on explainability

### Papers

1. **"A Unified Approach to Interpreting Model Predictions"** (SHAP paper)
   - Lundberg & Lee, NIPS 2017
   - [Paper](https://arxiv.org/abs/1705.07874)

2. **"Why Should I Trust You?"** (LIME paper)
   - Ribeiro et al., KDD 2016
   - [Paper](https://arxiv.org/abs/1602.04938)

### Online Courses

1. **"Interpretable Machine Learning"** - Coursera
   - [Course](https://www.coursera.org/learn/interpretable-machine-learning)

2. **"Explainable AI"** - edX
   - [Course](https://www.edx.org/course/explainable-ai)

### Tools Documentation

- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://github.com/marcotcr/lime)
- [Captum Documentation](https://captum.ai/)
- [InterpretML Documentation](https://interpret.ml/)

### Articles

1. **"The Myth of Model Interpretability"** - Medium
   - Discusses challenges in interpretability

2. **"Explainable AI: A Review"** - Towards Data Science
   - Overview of explainability methods

---

## Key Takeaways

1. **Interpretability is Context-Dependent**: Choose methods based on your use case and audience

2. **No One-Size-Fits-All**: Different methods provide different insights

3. **Validate Explanations**: Don't blindly trust explanation methods

4. **Balance Accuracy and Interpretability**: Sometimes simpler models are better

5. **Document Everything**: Keep records of interpretation methods and findings

6. **Consider Regulations**: Ensure compliance with GDPR, AI Act, etc.

7. **Start Simple**: Begin with feature importance, then move to more complex methods

---

**Remember**: Interpretability is not just about understanding models—it's about building trust, ensuring fairness, and creating better ML systems.

