# Handling Imbalanced Data Complete Guide

Comprehensive guide to handling imbalanced datasets in machine learning.

## Table of Contents

- [Introduction to Imbalanced Data](#introduction-to-imbalanced-data)
- [Understanding the Problem](#understanding-the-problem)
- [Resampling Techniques](#resampling-techniques)
- [Algorithm-Level Solutions](#algorithm-level-solutions)
- [Evaluation Metrics](#evaluation-metrics)
- [Best Practices](#best-practices)
- [Practice Exercises](#practice-exercises)

---

## Introduction to Imbalanced Data

### What is Imbalanced Data?

Imbalanced data occurs when classes in a classification problem are not represented equally.

**Examples:**
- Fraud detection: 99% legitimate, 1% fraud
- Medical diagnosis: 95% healthy, 5% disease
- Spam detection: 90% ham, 10% spam
- Customer churn: 85% retained, 15% churned

### Why is it a Problem?

**Challenges:**
- Models tend to predict majority class
- Accuracy is misleading (99% accuracy with 99% majority class)
- Minority class is often more important
- Standard algorithms assume balanced classes

**Example:**
```python
# Imbalanced dataset: 99% class 0, 1% class 1
# Naive model: Always predict class 0
# Accuracy: 99% (misleading!)
# But fails to detect any fraud (class 1)
```

---

## Understanding the Problem

### Measuring Imbalance

```python
import pandas as pd
import numpy as np
from collections import Counter

# Check class distribution
def check_imbalance(y):
    """Check class distribution"""
    counter = Counter(y)
    total = len(y)
    
    print("Class Distribution:")
    for class_label, count in counter.items():
        percentage = (count / total) * 100
        print(f"  Class {class_label}: {count} ({percentage:.2f}%)")
    
    # Calculate imbalance ratio
    counts = list(counter.values())
    imbalance_ratio = max(counts) / min(counts)
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 10:
        print("⚠️  Highly imbalanced!")
    elif imbalance_ratio > 5:
        print("⚠️  Moderately imbalanced")
    else:
        print("✓ Relatively balanced")

# Example
y = np.array([0]*900 + [1]*100)  # 90% vs 10%
check_imbalance(y)
```

### Why Algorithms Fail

**Most ML algorithms assume:**
- Balanced class distribution
- Equal misclassification costs
- Maximize overall accuracy

**Result:**
- Model learns to always predict majority class
- High accuracy but poor performance on minority class
- Important minority cases are missed

---

## Resampling Techniques

### Oversampling

Increase minority class samples.

#### Random Oversampling

```python
from imblearn.over_sampling import RandomOverSampler

# Random oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

print(f"Original: {Counter(y_train)}")
print(f"After oversampling: {Counter(y_resampled)}")
```

#### SMOTE (Synthetic Minority Oversampling Technique)

Creates synthetic samples.

```python
from imblearn.over_sampling import SMOTE

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# SMOTE variants
from imblearn.over_sampling import BorderlineSMOTE, ADASYN, SVMSMOTE

# Borderline SMOTE (focuses on borderline samples)
borderline_smote = BorderlineSMOTE(random_state=42)
X_borderline, y_borderline = borderline_smote.fit_resample(X_train, y_train)

# ADASYN (adaptive synthetic sampling)
adasyn = ADASYN(random_state=42)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)

# SVM SMOTE (uses SVM to find support vectors)
svm_smote = SVMSMOTE(random_state=42)
X_svm, y_svm = svm_smote.fit_resample(X_train, y_train)
```

### Undersampling

Reduce majority class samples.

#### Random Undersampling

```python
from imblearn.under_sampling import RandomUnderSampler

# Random undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

#### Tomek Links

Remove borderline majority samples.

```python
from imblearn.under_sampling import TomekLinks

# Tomek Links
tomek = TomekLinks()
X_resampled, y_resampled = tomek.fit_resample(X_train, y_train)
```

#### Edited Nearest Neighbors

```python
from imblearn.under_sampling import EditedNearestNeighbours

# Edited Nearest Neighbors
enn = EditedNearestNeighbours()
X_resampled, y_resampled = enn.fit_resample(X_train, y_train)
```

### Combined Methods

Combine oversampling and undersampling.

```python
from imblearn.combine import SMOTETomek, SMOTEENN

# SMOTE + Tomek Links
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# SMOTE + Edited Nearest Neighbors
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
```

---

## Algorithm-Level Solutions

### Class Weights

Penalize misclassifying minority class more.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Method 1: Balanced (automatic)
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Method 2: Custom weights
class_weights = {0: 1, 1: 10}  # Penalize class 1 misclassification 10x
model = LogisticRegression(class_weight=class_weights, random_state=42)
model.fit(X_train, y_train)

# Method 3: Compute from data
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

model = RandomForestClassifier(class_weight=class_weight_dict, random_state=42)
model.fit(X_train, y_train)
```

### Threshold Tuning

Adjust classification threshold.

```python
from sklearn.metrics import precision_recall_curve

# Get probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# F1 score for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")

# Predict with optimal threshold
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
```

### Cost-Sensitive Learning

```python
# Use cost matrix
from sklearn.metrics import make_scorer

def cost_sensitive_score(y_true, y_pred):
    """Custom scoring with different costs"""
    # Cost of false negative (missing fraud) is higher
    cost_matrix = {
        (0, 0): 0,   # True negative
        (0, 1): 1,   # False positive
        (1, 0): 10,  # False negative (expensive!)
        (1, 1): 0    # True positive
    }
    
    total_cost = 0
    for true, pred in zip(y_true, y_pred):
        total_cost += cost_matrix[(true, pred)]
    
    return -total_cost  # Negative because higher is better

cost_scorer = make_scorer(cost_sensitive_score, greater_is_better=True)
```

---

## Evaluation Metrics

### Appropriate Metrics for Imbalanced Data

**Don't use:**
- Accuracy (misleading)

**Use instead:**
- Precision-Recall Curve
- F1-Score
- ROC-AUC (but less informative than PR-AUC for imbalanced data)
- Precision-Recall AUC

```python
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, roc_auc_score,
    average_precision_score
)

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
print(f"\nPrecision-Recall AUC: {pr_auc:.3f}")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC: {roc_auc:.3f}")
```

---

## Best Practices

### When to Use What

| Technique | When to Use | Pros | Cons |
|-----------|-------------|------|------|
| **SMOTE** | Moderate imbalance | Creates synthetic samples | Can create noise |
| **Class Weights** | Any imbalance | No data loss | May not work for extreme imbalance |
| **Undersampling** | Large dataset | Faster training | Loss of information |
| **Threshold Tuning** | After training | Simple, effective | Need validation set |
| **Ensemble** | Severe imbalance | Robust | Complex |

### Workflow

1. **Analyze imbalance**: Check class distribution
2. **Choose strategy**: Based on dataset size and imbalance ratio
3. **Use appropriate metrics**: PR-AUC, F1-score
4. **Validate properly**: Stratified cross-validation
5. **Compare methods**: Try multiple approaches

---

## Practice Exercises

### Exercise 1: Handle Imbalanced Data

**Task:** Apply different techniques to imbalanced dataset and compare.

**Solution:**
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Create imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    weights=[0.9, 0.1],  # 90% vs 10%
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Strategy 1: No resampling (baseline)
model1 = RandomForestClassifier(random_state=42)
model1.fit(X_train, y_train)
score1 = model1.score(X_test, y_test)

# Strategy 2: SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
model2 = RandomForestClassifier(random_state=42)
model2.fit(X_smote, y_smote)
score2 = model2.score(X_test, y_test)

# Strategy 3: Class weights
model3 = RandomForestClassifier(class_weight='balanced', random_state=42)
model3.fit(X_train, y_train)
score3 = model3.score(X_test, y_test)

print(f"Baseline: {score1:.3f}")
print(f"SMOTE: {score2:.3f}")
print(f"Class Weights: {score3:.3f}")
```

---

## Key Takeaways

1. **Imbalanced data is common**: Especially in real-world problems
2. **Accuracy is misleading**: Use appropriate metrics
3. **Multiple solutions**: Resampling, class weights, threshold tuning
4. **Choose wisely**: Based on dataset and problem
5. **Validate properly**: Use stratified splits

---

**Remember**: Handling imbalanced data is crucial for real-world ML applications!

