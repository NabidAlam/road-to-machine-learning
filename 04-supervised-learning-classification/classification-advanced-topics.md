# Advanced Classification Topics

Comprehensive guide to advanced classification techniques, handling imbalanced data, probability calibration, and model interpretation.

## Table of Contents

- [Handling Imbalanced Datasets](#handling-imbalanced-datasets)
- [Probability Calibration](#probability-calibration)
- [Threshold Tuning](#threshold-tuning)
- [Precision-Recall Curves](#precision-recall-curves)
- [Model Interpretation](#model-interpretation)
- [Multiclass Classification Strategies](#multiclass-classification-strategies)
- [Common Classification Pitfalls](#common-classification-pitfalls)

---

## Handling Imbalanced Datasets

### What is Class Imbalance?

When classes are not equally represented in the dataset.

**Example:**
- 99% class A, 1% class B
- Model predicts A all the time → 99% accuracy but useless!

### Detecting Imbalance

```python
import pandas as pd
import numpy as np
from collections import Counter

# Check class distribution
class_counts = Counter(y_train)
print("Class Distribution:")
for class_label, count in class_counts.items():
    percentage = count / len(y_train) * 100
    print(f"  Class {class_label}: {count} ({percentage:.1f}%)")

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
pd.Series(y_train).value_counts().plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

# Calculate imbalance ratio
minority_class = min(class_counts.values())
majority_class = max(class_counts.values())
imbalance_ratio = majority_class / minority_class
print(f"\nImbalance Ratio: {imbalance_ratio:.1f}:1")
if imbalance_ratio > 10:
    print("Warning: Severe class imbalance!")
```

### Strategy 1: Resampling

#### Oversampling (SMOTE)

```python
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

# SMOTE: Synthetic Minority Oversampling Technique
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(f"  Original shape: {X_train.shape}")
print(f"  Resampled shape: {X_resampled.shape}")
print(f"  Class distribution: {Counter(y_resampled)}")

# Train on resampled data
model = LogisticRegression()
model.fit(X_resampled, y_resampled)
```

#### Undersampling

```python
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

# Random undersampling
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

# Tomek Links (removes borderline samples)
tomek = TomekLinks()
X_resampled, y_resampled = tomek.fit_resample(X_train, y_train)
```

#### Combined Sampling

```python
from imblearn.combine import SMOTETomek, SMOTEENN

# SMOTE + Tomek Links
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
```

### Strategy 2: Class Weights

```python
# Option 1: Balanced weights (automatic)
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Option 2: Custom weights
class_weights = {0: 1.0, 1: 10.0}  # Penalize class 1 misclassification more
model = LogisticRegression(class_weight=class_weights)
model.fit(X_train, y_train)

# Option 3: Calculate weights inversely proportional to frequency
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print(f"Class weights: {class_weight_dict}")

model = LogisticRegression(class_weight=class_weight_dict)
model.fit(X_train, y_train)
```

### Strategy 3: Threshold Tuning

```python
# Instead of default 0.5 threshold, find optimal threshold
from sklearn.metrics import f1_score

# Get probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Try different thresholds
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = []

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    f1_scores.append(f1)

# Find best threshold
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best threshold: {best_threshold:.2f}")
print(f"Best F1-score: {max(f1_scores):.3f}")

# Use best threshold
y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
```

### Strategy 4: Ensemble Methods

```python
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier

# Balanced Random Forest
balanced_rf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
balanced_rf.fit(X_train, y_train)

# Balanced Bagging
balanced_bag = BalancedBaggingClassifier(random_state=42)
balanced_bag.fit(X_train, y_train)
```

### Comparison of Strategies

```python
from sklearn.metrics import classification_report, roc_auc_score

strategies = {
    'Original (No handling)': (X_train, y_train, LogisticRegression()),
    'SMOTE': (X_resampled_smote, y_resampled_smote, LogisticRegression()),
    'Class Weights': (X_train, y_train, LogisticRegression(class_weight='balanced')),
    'Balanced RF': (X_train, y_train, BalancedRandomForestClassifier())
}

results = {}
for name, (X_tr, y_tr, model) in strategies.items():
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {'F1': f1, 'AUC': auc}
    print(f"\n{name}:")
    print(f"  F1-Score: {f1:.3f}")
    print(f"  ROC-AUC: {auc:.3f}")
```

---

## Probability Calibration

### What is Calibration?

Well-calibrated probabilities mean: if model predicts 80% probability, it should be correct 80% of the time.

### Checking Calibration

```python
from sklearn.calibration import calibration_curve

# Get predicted probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_pred_proba, n_bins=10
)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True)
plt.show()
```

### Calibrating Probabilities

```python
from sklearn.calibration import CalibratedClassifierCV

# Platt Scaling (sigmoid calibration)
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
calibrated_model.fit(X_train, y_train)

# Isotonic Regression (non-parametric)
calibrated_model_iso = CalibratedClassifierCV(model, method='isotonic', cv=3)
calibrated_model_iso.fit(X_train, y_train)

# Compare
models_to_compare = {
    'Original': model,
    'Platt Scaling': calibrated_model,
    'Isotonic': calibrated_model_iso
}

for name, mod in models_to_compare.items():
    y_proba = mod.predict_proba(X_test)[:, 1]
    fraction_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
    
    # Calculate Brier score (lower is better)
    from sklearn.metrics import brier_score_loss
    brier = brier_score_loss(y_test, y_proba)
    print(f"{name}: Brier Score = {brier:.4f}")
```

---

## Threshold Tuning

### Why Tune Threshold?

Default 0.5 threshold may not be optimal for your problem.

### Finding Optimal Threshold

```python
from sklearn.metrics import precision_recall_curve, f1_score

# Get probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate precision, recall at different thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Calculate F1 at each threshold
f1_scores = 2 * (precision * recall) / (precision + recall)
f1_scores = f1_scores[:-1]  # Remove last (undefined)

# Find threshold with best F1
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]
best_f1 = f1_scores[best_threshold_idx]

print(f"Best Threshold: {best_threshold:.3f}")
print(f"Best F1-Score: {best_f1:.3f}")

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.plot(thresholds, f1_scores, label='F1-Score')
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision, Recall, F1 vs Threshold')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(recall[:-1], precision[:-1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Business-Cost Based Threshold

```python
# If false negatives cost more than false positives
# Example: Medical diagnosis - missing disease is worse than false alarm

cost_fp = 1.0  # Cost of false positive
cost_fn = 10.0  # Cost of false negative (10x more expensive)

# Calculate cost at each threshold
costs = []
for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn = np.sum((y_test == 0) & (y_pred_thresh == 0))
    fp = np.sum((y_test == 0) & (y_pred_thresh == 1))
    fn = np.sum((y_test == 1) & (y_pred_thresh == 0))
    tp = np.sum((y_test == 1) & (y_pred_thresh == 1))
    
    total_cost = fp * cost_fp + fn * cost_fn
    costs.append(total_cost)

# Find threshold with minimum cost
best_cost_threshold = thresholds[np.argmin(costs)]
print(f"Optimal threshold (cost-based): {best_cost_threshold:.3f}")
print(f"Minimum cost: {min(costs):.2f}")
```

---

## Precision-Recall Curves

### When to Use PR Curve vs ROC Curve

- **ROC Curve**: Use when classes are balanced
- **PR Curve**: Use when classes are imbalanced (more informative)

### PR Curve

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# Calculate PR curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
average_precision = average_precision_score(y_test, y_pred_proba)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AP = {average_precision:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

print(f"Average Precision: {average_precision:.3f}")

# Interpretation:
# - AP = 1.0: Perfect classifier
# - AP = 0.5: Random classifier
# - Higher is better
```

### Comparing Multiple Models

```python
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(probability=True)
}

plt.figure(figsize=(10, 6))

for name, mod in models.items():
    mod.fit(X_train, y_train)
    y_proba = mod.predict_proba(X_test)[:, 1]
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    
    plt.plot(recall, precision, label=f'{name} (AP = {ap:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Model Interpretation

### Feature Importance (Tree-based)

```python
# Random Forest feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()
```

### Coefficient Interpretation (Logistic Regression)

```python
# Logistic regression coefficients
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)

coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lr.coef_[0],
    'Odds Ratio': np.exp(lr.coef_[0])
}).sort_values('Coefficient', ascending=False)

print("Logistic Regression Coefficients:")
print(coefficients)

# Interpretation:
# - Positive coefficient: Feature increases probability of positive class
# - Negative coefficient: Feature decreases probability
# - Odds Ratio > 1: Increases odds
# - Odds Ratio < 1: Decreases odds
```

### SHAP Values (Advanced)

```python
# Install: pip install shap
try:
    import shap
    
    # Tree explainer for Random Forest
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    
    # Waterfall plot for single prediction
    shap.waterfall_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0])
except ImportError:
    print("SHAP not installed. Install with: pip install shap")
```

---

## Multiclass Classification Strategies

### One-vs-Rest (OvR)

```python
# Automatically used by most sklearn classifiers
model = LogisticRegression(multi_class='ovr')  # One-vs-Rest
model.fit(X_train, y_train)

# Creates one binary classifier per class
# Each classifier: one class vs all others
```

### One-vs-One (OvO)

```python
from sklearn.multiclass import OneVsOneClassifier

# One-vs-One strategy
ovo = OneVsOneClassifier(LogisticRegression())
ovo.fit(X_train, y_train)

# Creates binary classifier for each pair of classes
# More classifiers but potentially more accurate
```

### Multinomial (Softmax)

```python
# Direct multiclass (for Logistic Regression)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# Uses softmax function
# Directly models all classes together
```

### Comparison

```python
strategies = {
    'One-vs-Rest': LogisticRegression(multi_class='ovr'),
    'One-vs-One': OneVsOneClassifier(LogisticRegression()),
    'Multinomial': LogisticRegression(multi_class='multinomial', solver='lbfgs')
}

for name, model in strategies.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: Accuracy = {accuracy:.3f}")
```

---

## Common Classification Pitfalls

### Pitfall 1: Using Accuracy for Imbalanced Data

**Problem**: 99% accuracy on 99:1 imbalanced data is misleading.

**Solution**: Use Precision, Recall, F1, ROC-AUC, or PR-AUC.

### Pitfall 2: Not Tuning Threshold

**Problem**: Default 0.5 threshold may not be optimal.

**Solution**: Tune threshold based on business costs or F1-score.

### Pitfall 3: Ignoring Probability Calibration

**Problem**: Probabilities may not be well-calibrated.

**Solution**: Use CalibratedClassifierCV.

### Pitfall 4: Overfitting on Minority Class

**Problem**: SMOTE can create overfitting if used incorrectly.

**Solution**: Apply SMOTE only on training set, not validation/test.

### Pitfall 5: Wrong Evaluation Metric

**Problem**: Using wrong metric for problem type.

**Solution**:
- Medical diagnosis: High Recall (don't miss cases)
- Spam detection: High Precision (don't mark real emails as spam)
- General: F1-Score or ROC-AUC

### Pitfall 6: Data Leakage in Resampling

**Problem**: Resampling before train/test split.

**Solution**: Always split first, then resample only training data.

---

## Key Takeaways

1. **Imbalanced Data**: Use appropriate metrics, resampling, or class weights
2. **Threshold Tuning**: Find optimal threshold for your problem
3. **Probability Calibration**: Ensure probabilities are meaningful
4. **PR Curves**: Better than ROC for imbalanced data
5. **Model Interpretation**: Understand what features matter
6. **Multiclass Strategies**: Choose appropriate strategy

---

**Remember**: Classification requires careful handling of class imbalance and appropriate evaluation metrics!

