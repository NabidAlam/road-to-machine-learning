# Model Explainability Complete Guide

Comprehensive guide to understanding and explaining machine learning models. Learn to make your models interpretable and trustworthy with detailed explanations, code examples, and real-world applications.

## Table of Contents

- [Introduction to Model Explainability](#introduction-to-model-explainability)
- [Types of Explainability](#types-of-explainability)
- [Feature Importance](#feature-importance)
- [SHAP (SHapley Additive exPlanations)](#shap-shapley-additive-explanations)
- [LIME (Local Interpretable Model-agnostic Explanations)](#lime-local-interpretable-model-agnostic-explanations)
- [Partial Dependence Plots](#partial-dependence-plots)
- [ICE Plots](#ice-plots)
- [Complete Workflow Example](#complete-workflow-example)
- [Best Practices](#best-practices)
- [Common Pitfalls](#common-pitfalls)
- [Practice Exercises](#practice-exercises)
- [Additional Resources](#additional-resources)

---

## Introduction to Model Explainability

### Why Explainability Matters

**Critical Reasons:**

1. **Trust and Adoption**
   - Users need to understand why a model made a prediction
   - Builds confidence in model decisions
   - Essential for user acceptance

2. **Debugging and Improvement**
   - Identify model errors and biases
   - Understand failure cases
   - Improve model performance

3. **Regulatory Compliance**
   - GDPR: Right to explanation
   - Financial regulations: Explain credit decisions
   - Healthcare: Explain medical diagnoses
   - Legal requirements in many industries

4. **Fairness and Ethics**
   - Detect bias and discrimination
   - Ensure fair treatment
   - Build ethical AI systems

5. **Business Understanding**
   - Understand what drives predictions
   - Make informed business decisions
   - Communicate insights to stakeholders

6. **Model Validation**
   - Verify model makes sense
   - Catch unexpected behaviors
   - Validate against domain knowledge

### Real-World Examples

**Healthcare:**
- Doctor needs to understand why AI diagnosed a disease
- Patient has right to explanation
- Regulatory requirement for medical devices

**Finance:**
- Bank must explain credit denial
- Regulatory requirement (Equal Credit Opportunity Act)
- Customer trust and satisfaction

**Criminal Justice:**
- Explain risk assessment scores
- Ensure fairness and transparency
- Legal and ethical requirements

**Hiring:**
- Explain why candidate was rejected
- Detect and prevent bias
- Legal compliance (EEOC)

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

SHAP (SHapley Additive exPlanations) is a unified framework for explaining model predictions. It's based on Shapley values from cooperative game theory and provides a mathematically principled way to explain any machine learning model.

### SHAP Concepts

**Core Idea:**
SHAP values explain the output of any model by computing the contribution of each feature to the prediction. The sum of all SHAP values equals the difference between the model's prediction and the expected value.

**Key Properties:**
1. **Efficiency**: Sum of SHAP values = prediction - expected value
2. **Symmetry**: Features with equal marginal contributions get equal SHAP values
3. **Dummy**: Features that don't affect output get zero SHAP value
4. **Additivity**: SHAP values are additive across features

**Mathematical Foundation:**
SHAP values are based on Shapley values from game theory, which fairly distribute the "payout" (prediction) among "players" (features).

### Installation

```bash
pip install shap
```

### Tree SHAP

Tree SHAP is optimized for tree-based models (Random Forest, XGBoost, LightGBM, etc.). It's fast and exact.

```python
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create Tree explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values is a list
# shap_values[0] for class 0, shap_values[1] for class 1
# For multi-class, list of arrays for each class

# Summary plot (global interpretation)
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)

# Summary plot with bar chart (mean absolute SHAP values)
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names, show=False)

# Waterfall plot for single prediction
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0][0],  # First instance, first class
        base_values=explainer.expected_value[0],
        data=X_test.iloc[0],
        feature_names=feature_names
    )
)

# Force plot (interactive)
shap.force_plot(
    explainer.expected_value[0],
    shap_values[0][0],
    X_test.iloc[0],
    feature_names=feature_names,
    matplotlib=True
)
```

### Kernel SHAP

Kernel SHAP is model-agnostic and works with any model. It's slower but more flexible.

```python
import shap

# Kernel explainer (works with any model)
# Background data: representative sample of training data
explainer = shap.KernelExplainer(
    model.predict_proba,  # Prediction function
    X_train[:100]  # Background data (small sample for speed)
)

# Calculate SHAP values for test instances
shap_values = explainer.shap_values(X_test[0:5])

# Force plot
shap.force_plot(
    explainer.expected_value[1],  # Expected value for class 1
    shap_values[1][0],  # SHAP values for first instance, class 1
    X_test.iloc[0],
    feature_names=feature_names,
    matplotlib=True
)

# Summary plot
shap.summary_plot(shap_values[1], X_test[0:5], feature_names=feature_names, show=False)
```

### Effect of Background Data

**What is Background Data?**
Background data is a representative sample used to compute expected values and SHAP values. It represents the "average" or "baseline" prediction.

**Impact of Background Data:**
- **Size**: Larger background = more accurate but slower
- **Representativeness**: Should represent your data distribution
- **Selection**: Random sample or stratified sample

**Example:**
```python
# Small background (fast, less accurate)
explainer_small = shap.KernelExplainer(model.predict_proba, X_train[:50])

# Large background (slow, more accurate)
explainer_large = shap.KernelExplainer(model.predict_proba, X_train[:500])

# Stratified background (balanced classes)
from sklearn.model_selection import train_test_split
_, X_background = train_test_split(
    X_train, 
    test_size=100, 
    stratify=y_train,
    random_state=42
)
explainer_stratified = shap.KernelExplainer(model.predict_proba, X_background)
```

### SHAP for Regression

```python
from sklearn.ensemble import RandomForestRegressor
import shap

# Train regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tree explainer for regression
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)

# Dependence plot (feature interaction)
shap.dependence_plot(0, shap_values, X_test, feature_names=feature_names, show=False)

# Waterfall plot
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=feature_names
    )
)
```

### SHAP for Classification

```python
from sklearn.ensemble import RandomForestClassifier
import shap

# Train classification model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tree explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classification
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]  # Class 1 SHAP values
else:
    shap_values_class1 = shap_values

# Summary plot
shap.summary_plot(shap_values_class1, X_test, feature_names=feature_names, show=False)

# Summary plot with class labels
shap.summary_plot(shap_values, X_test, feature_names=feature_names, class_names=['Class 0', 'Class 1'], show=False)
```

### SHAP for Neural Networks (ANN)

```python
import tensorflow as tf
from tensorflow import keras
import shap

# Train neural network
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=0)

# Deep explainer (for neural networks)
explainer = shap.DeepExplainer(model, X_train[:100])  # Background data
shap_values = explainer.shap_values(X_test[:10])

# Summary plot
shap.summary_plot(shap_values[0], X_test[:10], feature_names=feature_names, show=False)

# Force plot
shap.force_plot(
    explainer.expected_value[0],
    shap_values[0][0],
    X_test.iloc[0],
    feature_names=feature_names,
    matplotlib=True
)
```

### SHAP for Convolutional Neural Networks (CNN)

```python
import tensorflow as tf
from tensorflow import keras
import shap
import numpy as np

# Train CNN (example for image classification)
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_images, y_train, epochs=5, verbose=0)

# Deep explainer for CNN
# Use subset of training images as background
background_images = X_train_images[:50]
explainer = shap.DeepExplainer(model, background_images)

# Explain test images
test_images = X_test_images[:5]
shap_values = explainer.shap_values(test_images)

# For multi-class, shap_values is a list
# Visualize SHAP values for first class
shap.image_plot(shap_values[0], test_images, show=False)

# Summary plot (flattened)
shap.summary_plot(
    shap_values[0].reshape(len(test_images), -1),
    test_images.reshape(len(test_images), -1),
    feature_names=[f"pixel_{i}" for i in range(test_images[0].size)],
    show=False
)
```

### Global vs Local Interpretations

**Local Interpretation:**
Explains a single prediction.

```python
# Explain single instance
instance_idx = 0
shap_values_single = explainer.shap_values(X_test.iloc[instance_idx:instance_idx+1])

# Waterfall plot (local)
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_single[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[instance_idx],
        feature_names=feature_names
    )
)

# Force plot (local)
shap.force_plot(
    explainer.expected_value,
    shap_values_single[0],
    X_test.iloc[instance_idx],
    feature_names=feature_names,
    matplotlib=True
)
```

**Global Interpretation:**
Explains the model's overall behavior.

```python
# Summary plot (global - shows all instances)
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)

# Bar plot (mean absolute SHAP values - global feature importance)
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names, show=False)

# Dependence plot (feature interaction - global)
shap.dependence_plot(
    "feature1",  # Feature to analyze
    shap_values,
    X_test,
    feature_names=feature_names,
    interaction_index="feature2",  # Feature to show interaction with
    show=False
)
```

### Math Behind SHAP (Shapley Values)

**Shapley Value Formula:**

For a feature i, the Shapley value is:

```
φᵢ = Σ [|S|!(n-|S|-1)!/n!] × [f(S ∪ {i}) - f(S)]
```

Where:
- S: Subset of features (coalition)
- n: Total number of features
- f(S): Model prediction with features in S
- f(S ∪ {i}): Model prediction with feature i added

**Intuition:**
- Shapley value = Average marginal contribution of feature i
- Marginal contribution = How much feature i adds to prediction
- Averaged over all possible feature combinations

**Properties:**
1. **Efficiency**: Σᵢ φᵢ = f(N) - f(∅)
2. **Symmetry**: If features i and j contribute equally, φᵢ = φⱼ
3. **Dummy**: If feature i never affects output, φᵢ = 0
4. **Additivity**: For combined models, SHAP values add up

**Example Calculation:**
```python
# Simplified example: 3 features
# Model: f(x1, x2, x3) = x1 + 2*x2 + 3*x3
# For instance: x1=1, x2=2, x3=3
# Prediction: 1 + 4 + 9 = 14
# Expected value (average): 0

# Feature 1 (x1) Shapley value:
# S={}: f({1}) - f({}) = 1 - 0 = 1
# S={2}: f({1,2}) - f({2}) = 5 - 4 = 1
# S={3}: f({1,3}) - f({3}) = 10 - 9 = 1
# S={2,3}: f({1,2,3}) - f({2,3}) = 14 - 13 = 1
# Average: 1

# Similarly for features 2 and 3
# SHAP values: [1, 4, 9] (matches feature contributions)
```

**Computational Complexity:**
- Exact calculation: O(2ⁿ) where n = number of features
- Tree SHAP: O(TLD²) where T = trees, L = leaves, D = depth
- Kernel SHAP: Approximates with sampling

### SHAP Visualizations

**1. Summary Plot:**
Shows feature importance and impact on predictions.

```python
# Summary plot (beeswarm)
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)

# Bar plot (mean absolute SHAP)
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names, show=False)
```

**2. Waterfall Plot:**
Shows how SHAP values accumulate from base value to prediction.

```python
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=feature_names
    )
)
```

**3. Force Plot:**
Interactive visualization showing feature contributions.

```python
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0],
    feature_names=feature_names,
    matplotlib=True
)
```

**4. Dependence Plot:**
Shows how a feature's value affects its SHAP value.

```python
shap.dependence_plot(
    "feature1",
    shap_values,
    X_test,
    feature_names=feature_names,
    interaction_index="feature2",  # Color by interaction feature
    show=False
)
```

**5. Heatmap:**
Shows SHAP values for multiple instances.

```python
shap.plots.heatmap(
    shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_test,
        feature_names=feature_names
    ),
    show=False
)
```

---

## LIME (Local Interpretable Model-agnostic Explanations)

LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by approximating the model locally with an interpretable model (usually linear).

### LIME Concepts

**Core Idea:**
LIME creates a local, interpretable approximation of the model around a specific prediction. It:
1. Generates perturbed samples around the instance
2. Gets predictions from the black-box model
3. Fits a simple interpretable model (linear) to these predictions
4. Uses the simple model's coefficients as explanations

**Key Properties:**
- **Local**: Explains individual predictions, not the whole model
- **Model-agnostic**: Works with any model
- **Interpretable**: Uses simple models (linear) for explanation
- **Approximate**: Local approximation, not exact

**When to Use LIME:**
- Need local explanations (single predictions)
- Model is complex black-box
- SHAP is too slow
- Want to understand specific predictions

### Installation

```bash
pip install lime
```

### LIME for Tabular Data

```python
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer

# Create explainer
explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=['Class 0', 'Class 1'],
    mode='classification',
    training_labels=y_train,
    discretize_continuous=True  # Discretize continuous features
)

# Explain single prediction
explanation = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10,  # Top 10 features
    top_labels=1  # Explain top predicted class
)

# Show explanation in notebook
explanation.show_in_notebook(show_table=True, show_all=False)

# Get explanation as list
explanation_list = explanation.as_list(label=1)  # For class 1
print("\nFeature Contributions:")
for feature, value in explanation_list:
    print(f"  {feature}: {value:.4f}")

# Get explanation as map (feature index -> contribution)
explanation_map = explanation.as_map()[1]  # For class 1
print("\nFeature Contributions (by index):")
for idx, value in explanation_map:
    print(f"  Feature {idx} ({feature_names[idx]}): {value:.4f}")

# Visualize
explanation.show_in_notebook()
```

### LIME for Regression

```python
from lime import lime_tabular

# Create explainer for regression
explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    mode='regression',
    discretize_continuous=True
)

# Explain single prediction
explanation = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict,  # Regression: predict, not predict_proba
    num_features=10
)

# Show explanation
explanation.show_in_notebook(show_table=True)

# Get explanation
explanation_list = explanation.as_list()
for feature, value in explanation_list:
    print(f"{feature}: {value:.4f}")
```

### LIME for Text Data

```python
from lime import lime_text
from lime.lime_text import LimeTextExplainer

# Create text explainer
explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

# Explain text prediction
text = "This movie was terrible and boring"
explanation = explainer.explain_instance(
    text,
    model.predict_proba,  # Text classification model
    num_features=10
)

# Show explanation
explanation.show_in_notebook()

# Get explanation
explanation_list = explanation.as_list()
print("\nWord Contributions:")
for word, value in explanation_list:
    print(f"  {word}: {value:.4f}")
```

### LIME for Image Data

```python
from lime import lime_image
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# Create image explainer
explainer = LimeImageExplainer()

# Explain image prediction
explanation = explainer.explain_instance(
    image,  # Input image (numpy array)
    model.predict,  # Image classification model
    top_labels=5,  # Top 5 classes
    hide_color=0,  # Color to hide segments
    num_samples=1000  # Number of perturbed samples
)

# Get explanation for top class
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],  # Top predicted class
    positive_only=True,  # Only show positive contributions
    num_features=5,  # Top 5 segments
    hide_rest=True
)

# Visualize
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.title(f"LIME Explanation for Class {explanation.top_labels[0]}")
plt.axis('off')
plt.show()
```

### Comparing SHAP and LIME

| Aspect | SHAP | LIME |
|--------|------|------|
| **Scope** | Global and local | Local only |
| **Theoretical Foundation** | Game theory (Shapley values) | Local linear approximation |
| **Accuracy** | Exact (for tree models) or approximate | Approximate |
| **Speed** | Fast (Tree SHAP) or slow (Kernel SHAP) | Moderate |
| **Interpretability** | Feature contributions | Feature contributions |
| **Consistency** | Consistent (efficiency property) | May vary with sampling |
| **Use Case** | Comprehensive explanations | Quick local explanations |

**When to Use SHAP:**
- Need global and local explanations
- Want theoretical guarantees
- Tree-based models (use Tree SHAP)
- Need consistent explanations

**When to Use LIME:**
- Only need local explanations
- Model is very complex
- Quick explanations needed
- Text or image data

**Example: Using Both:**
```python
# Use SHAP for global understanding
shap_explainer = shap.TreeExplainer(model)
shap_values = shap_explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Use LIME for specific predictions
lime_explainer = LimeTabularExplainer(X_train.values, feature_names=feature_names)
for idx in [0, 5, 10]:  # Explain specific instances
    explanation = lime_explainer.explain_instance(
        X_test.iloc[idx].values,
        model.predict_proba,
        num_features=10
    )
    explanation.show_in_notebook()
```

---

## SHAP Case Studies

### Case Study 1: Regression

**Problem:** Predict house prices using features like size, location, age, etc.

```python
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load and prepare data
# X_train, X_test, y_train, y_test = ...

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# SHAP Explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global interpretation
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)

# Local interpretation for specific house
house_idx = 0
print(f"\nActual Price: ${y_test.iloc[house_idx]:,.2f}")
print(f"Predicted Price: ${y_pred[house_idx]:,.2f}")
print(f"Expected Value (Average): ${explainer.expected_value:,.2f}")

# Waterfall plot
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[house_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[house_idx],
        feature_names=feature_names
    )
)

# Feature contributions
contributions = pd.DataFrame({
    'Feature': feature_names,
    'Value': X_test.iloc[house_idx].values,
    'SHAP Value': shap_values[house_idx]
})
contributions['Contribution'] = contributions['SHAP Value']
contributions = contributions.sort_values('Contribution', key=abs, ascending=False)
print("\nFeature Contributions:")
print(contributions.head(10))
```

**Interpretation:**
- Positive SHAP value: Feature increases prediction
- Negative SHAP value: Feature decreases prediction
- Magnitude: How much the feature affects prediction

### Case Study 2: Classification

**Problem:** Classify customer churn using customer features.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# SHAP Explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classification, explain class 1 (churn)
shap_values_churn = shap_values[1]

# Global interpretation
shap.summary_plot(shap_values_churn, X_test, feature_names=feature_names, show=False)

# Local interpretation for specific customer
customer_idx = 0
predicted_class = model.predict(X_test.iloc[customer_idx:customer_idx+1])[0]
predicted_proba = model.predict_proba(X_test.iloc[customer_idx:customer_idx+1])[0]

print(f"\nCustomer ID: {customer_idx}")
print(f"Predicted Class: {predicted_class} ({'Churn' if predicted_class == 1 else 'No Churn'})")
print(f"Predicted Probability: {predicted_proba[1]:.4f}")

# Waterfall plot
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_churn[customer_idx],
        base_values=explainer.expected_value[1],
        data=X_test.iloc[customer_idx],
        feature_names=feature_names
    )
)

# Top features driving churn prediction
contributions = pd.DataFrame({
    'Feature': feature_names,
    'Value': X_test.iloc[customer_idx].values,
    'SHAP Value': shap_values_churn[customer_idx]
})
contributions = contributions.sort_values('SHAP Value', ascending=False)
print("\nTop Features Increasing Churn Probability:")
print(contributions.head(5))
print("\nTop Features Decreasing Churn Probability:")
print(contributions.tail(5))
```

### Case Study 3: Artificial Neural Network (ANN)

**Problem:** Classify images or tabular data using neural networks.

```python
import tensorflow as tf
from tensorflow import keras
import shap
import numpy as np

# Build and train ANN
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# SHAP Explanation with Deep Explainer
# Use subset of training data as background
background_data = X_train[:100].values
explainer = shap.DeepExplainer(model, background_data)

# Explain test instances
test_data = X_test[:20].values
shap_values = explainer.shap_values(test_data)

# Summary plot
shap.summary_plot(shap_values[0], test_data, feature_names=feature_names, show=False)

# Local explanation
instance_idx = 0
shap.force_plot(
    explainer.expected_value[0],
    shap_values[0][instance_idx],
    test_data[instance_idx],
    feature_names=feature_names,
    matplotlib=True
)

# Compare with model's feature importance (if available)
print(f"\nExpected Value: {explainer.expected_value[0]:.4f}")
print(f"Prediction: {model.predict(test_data[instance_idx:instance_idx+1])[0][0]:.4f}")
print(f"Sum of SHAP values: {np.sum(shap_values[0][instance_idx]):.4f}")
print(f"Expected + SHAP sum: {explainer.expected_value[0] + np.sum(shap_values[0][instance_idx]):.4f}")
```

### Case Study 4: Convolutional Neural Network (CNN)

**Problem:** Classify images using CNN.

```python
import tensorflow as tf
from tensorflow import keras
import shap
import numpy as np
import matplotlib.pyplot as plt

# Build and train CNN (example: CIFAR-10)
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train_images, y_train, epochs=10, verbose=0)

# SHAP Explanation for CNN
# Use subset of training images as background
background_images = X_train_images[:50]
explainer = shap.DeepExplainer(model, background_images)

# Explain test images
test_images = X_test_images[:5]
shap_values = explainer.shap_values(test_images)

# Visualize SHAP values for first class
class_idx = 0
shap.image_plot(shap_values[class_idx], test_images, show=False)

# For specific image
image_idx = 0
predicted_class = np.argmax(model.predict(test_images[image_idx:image_idx+1])[0])
print(f"Predicted Class: {predicted_class}")

# Visualize SHAP values for predicted class
shap.image_plot(
    shap_values[predicted_class][image_idx:image_idx+1],
    test_images[image_idx:image_idx+1],
    show=False
)
```

**Interpretation:**
- Red regions: Increase probability of predicted class
- Blue regions: Decrease probability of predicted class
- Intensity: Magnitude of contribution

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

## ICE Plots (Individual Conditional Expectation)

ICE plots show how each individual prediction changes as a feature varies.

```python
from sklearn.inspection import PartialDependenceDisplay
import numpy as np

# ICE plots show individual curves
# Useful for detecting heterogeneous effects
PartialDependenceDisplay.from_estimator(
    model, X_train, features=[0, 1],
    kind='individual',  # Show ICE plots
    ice_lines=True,
    feature_names=feature_names,
    grid_resolution=20
)
plt.tight_layout()
plt.show()

# Compare average (PDP) vs individual (ICE)
PartialDependenceDisplay.from_estimator(
    model, X_train, features=[0],
    kind='both',  # Show both PDP and ICE
    feature_names=feature_names
)
plt.tight_layout()
plt.show()
```

**When to Use ICE Plots:**
- Detect heterogeneous effects (different individuals respond differently)
- Identify subgroups with different behaviors
- Understand individual-level effects

## Complete Workflow Example

Let's walk through a complete explainability workflow:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
from lime import lime_tabular
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

# Step 1: Load and prepare data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

print(f"Model Accuracy: {model.score(X_test_scaled, y_test):.3f}")

# Step 3: Feature Importance (Tree-based)
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features (Tree-based):")
print(feature_importance.head(10))

# Step 4: Permutation Importance (Model-agnostic)
perm_importance = permutation_importance(
    model, X_test_scaled, y_test,
    n_repeats=10, random_state=42, n_jobs=-1
)

perm_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("\nTop 10 Features (Permutation):")
print(perm_df.head(10))

# Step 5: SHAP Values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled[:100])  # Use subset for speed

# Summary plot
shap.summary_plot(shap_values[1], X_test_scaled[:100], feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

# Waterfall plot for single prediction
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[1][0],
        base_values=explainer.expected_value[1],
        data=X_test_scaled[0],
        feature_names=feature_names
    ),
    show=False
)
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()

# Step 6: LIME for local explanation
lime_explainer = lime_tabular.LimeTabularExplainer(
    X_train_scaled,
    feature_names=feature_names,
    class_names=['Benign', 'Malignant'],
    mode='classification'
)

# Explain a single instance
explanation = lime_explainer.explain_instance(
    X_test_scaled[0],
    model.predict_proba,
    num_features=10
)

print("\nLIME Explanation for First Test Instance:")
explanation.show_in_notebook(show_table=True)

# Step 7: Partial Dependence Plots
top_features = feature_importance.head(4)['feature'].tolist()
feature_indices = [list(feature_names).index(f) for f in top_features]

PartialDependenceDisplay.from_estimator(
    model, X_train_scaled, feature_indices,
    feature_names=feature_names,
    grid_resolution=20
)
plt.tight_layout()
plt.savefig('pdp_plots.png', dpi=150, bbox_inches='tight')
plt.close()

# Step 8: Compare explanations
print("\n=== Explanation Comparison ===")
print("Tree-based Importance Top 5:")
print(feature_importance.head(5)[['feature', 'importance']])

print("\nPermutation Importance Top 5:")
print(perm_df.head(5)[['feature', 'importance_mean']])

print("\nSHAP Mean Absolute Values Top 5:")
shap_mean_abs = pd.DataFrame({
    'feature': feature_names,
    'shap_importance': np.abs(shap_values[1]).mean(0)
}).sort_values('shap_importance', ascending=False)
print(shap_mean_abs.head(5))
```

## Common Pitfalls

### Pitfall 1: Using Feature Importance for All Models

**Problem:** Feature importance only works for tree-based models.

**Solution:** Use permutation importance or SHAP for any model:
```python
# Works for any model
perm_importance = permutation_importance(model, X_test, y_test)
```

### Pitfall 2: Interpreting Correlated Features

**Problem:** Feature importance can be misleading with correlated features.

**Solution:** 
- Use SHAP which handles correlations better
- Consider feature groups
- Use domain knowledge

### Pitfall 3: Over-interpreting Local Explanations

**Problem:** LIME explanations are approximations and can vary.

**Solution:**
- Run LIME multiple times and average
- Use SHAP for more stable local explanations
- Validate with domain experts

### Pitfall 4: Ignoring Model Complexity

**Problem:** Complex models are harder to explain.

**Solution:**
- Use simpler models when possible
- Use model-agnostic methods (SHAP, LIME)
- Consider surrogate models

## Additional Resources

### Online Resources

1. **SHAP Documentation** (https://shap.readthedocs.io/) - Comprehensive SHAP library documentation
2. **LIME Documentation** (https://github.com/marcotcr/lime) - LIME library and examples
3. **Interpretable Machine Learning Book** (https://christophm.github.io/interpretable-ml-book/) - Free online book by Christoph Molnar
4. **AI Explainability 360** (https://aix360.mybluemix.net/) - IBM's explainability toolkit

### Research Papers

1. **SHAP Paper** (Lundberg & Lee, 2017) - "A Unified Approach to Interpreting Model Predictions"
2. **LIME Paper** (Ribeiro et al., 2016) - "Why Should I Trust You?"
3. **PDP Paper** (Friedman, 2001) - "Greedy Function Approximation"

### Books

1. "Interpretable Machine Learning" by Christoph Molnar - Comprehensive guide
2. "Explainable AI" by Ajay Thampi - Practical guide to XAI

### Tools and Libraries

1. **SHAP** - Unified framework for model explanations
2. **LIME** - Local interpretable model-agnostic explanations
3. **ELI5** - Debug machine learning classifiers
4. **Alibi** - Algorithms for monitoring and explaining ML models
5. **Captum** - Model interpretability for PyTorch

---

## Key Takeaways

1. **Explainability is crucial**: Especially for production models and regulatory compliance
2. **Multiple techniques exist**: Feature importance, SHAP, LIME, PDP, ICE plots
3. **SHAP is powerful**: Unified framework that works with any model
4. **LIME for local**: Great for explaining individual predictions
5. **Visualize everything**: Plots help communicate findings to stakeholders
6. **Consider context**: Choose method based on model type and use case
7. **Validate explanations**: Check against domain knowledge
8. **Document findings**: Keep records of explanations for compliance

---

**Remember**: Explainable models build trust, enable debugging, and ensure compliance! Start with simple methods (feature importance) and progress to advanced techniques (SHAP) as needed. Always validate explanations with domain experts.

