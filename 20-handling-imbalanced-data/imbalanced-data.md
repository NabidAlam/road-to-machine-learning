# Handling Imbalanced Data Complete Guide

Comprehensive guide to handling imbalanced datasets in machine learning. Learn to build effective models when classes are not equally represented, with detailed explanations, code examples, and real-world applications.

## Table of Contents

- [Introduction to Imbalanced Data](#introduction-to-imbalanced-data)
- [Understanding the Problem](#understanding-the-problem)
- [Measuring Imbalance](#measuring-imbalance)
- [Resampling Techniques](#resampling-techniques)
- [Algorithm-Level Solutions](#algorithm-level-solutions)
- [Evaluation Metrics](#evaluation-metrics)
- [Complete Workflow Example](#complete-workflow-example)
- [Best Practices](#best-practices)
- [Common Pitfalls](#common-pitfalls)
- [Practice Exercises](#practice-exercises)
- [Additional Resources](#additional-resources)

---

## Introduction to Imbalanced Data

### What is Imbalanced Data?

Imbalanced data occurs when classes in a classification problem are not represented equally. This is extremely common in real-world machine learning problems.

**Real-World Examples:**
- **Fraud Detection**: 99% legitimate transactions, 1% fraudulent
- **Medical Diagnosis**: 95% healthy patients, 5% with disease
- **Spam Detection**: 90% legitimate emails, 10% spam
- **Customer Churn**: 85% retained customers, 15% churned
- **Credit Default**: 98% good loans, 2% defaults
- **Anomaly Detection**: 99.9% normal, 0.1% anomalies
- **Rare Disease Detection**: 99.5% healthy, 0.5% with rare condition

### Why is it a Problem?

**The Core Issue:**
Most machine learning algorithms are designed assuming balanced classes. When classes are imbalanced, these algorithms tend to:
1. Predict the majority class most of the time
2. Achieve high accuracy by simply predicting the majority class
3. Ignore the minority class, which is often the most important

**Example:**
```python
# Imbalanced dataset: 99% class 0 (normal), 1% class 1 (fraud)
# Naive model: Always predict class 0
# Result:
#   - Accuracy: 99% (looks great!)
#   - Precision for fraud: 0% (catches no fraud)
#   - Recall for fraud: 0% (misses all fraud)
#   - Business impact: Catastrophic! All fraud goes undetected
```

**Why Accuracy is Misleading:**
```python
from sklearn.metrics import accuracy_score

# Imbalanced dataset
y_true = [0]*990 + [1]*10  # 990 normal, 10 fraud
y_pred = [0]*1000  # Always predict normal

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2%}")  # 99% - looks great!
print("But caught 0 fraud cases!")  # Actually terrible!
```

### When Does Imbalance Matter?

**Imbalance is a problem when:**
- Minority class is important (fraud, disease, churn)
- Cost of missing minority class is high
- You need to detect rare events
- Business metrics focus on minority class

**Imbalance might not matter when:**
- Minority class is not important
- You only care about majority class
- Imbalance is mild (e.g., 60:40 ratio)

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

## Complete Workflow Example

Let's walk through a complete example of handling imbalanced data from start to finish.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create imbalanced dataset
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.95, 0.05],  # 95% vs 5% - highly imbalanced
    random_state=42
)

# Step 2: Check imbalance
print("Class distribution:")
print(f"Class 0: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.2f}%)")
print(f"Class 1: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.2f}%)")

# Step 3: Split data (use stratified split!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Baseline model (no handling)
print("\n=== Baseline Model (No Handling) ===")
baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_baseline))
print(f"\nROC-AUC: {roc_auc_score(y_test, baseline_model.predict_proba(X_test)[:, 1]):.3f}")
print(f"PR-AUC: {average_precision_score(y_test, baseline_model.predict_proba(X_test)[:, 1]):.3f}")

# Step 5: Apply SMOTE
print("\n=== Model with SMOTE ===")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - Class 0: {np.sum(y_train_smote == 0)}, Class 1: {np.sum(y_train_smote == 1)}")

smote_model = RandomForestClassifier(n_estimators=100, random_state=42)
smote_model.fit(X_train_smote, y_train_smote)
y_pred_smote = smote_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_smote))
print(f"\nROC-AUC: {roc_auc_score(y_test, smote_model.predict_proba(X_test)[:, 1]):.3f}")
print(f"PR-AUC: {average_precision_score(y_test, smote_model.predict_proba(X_test)[:, 1]):.3f}")

# Step 6: Use class weights
print("\n=== Model with Class Weights ===")
weighted_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
weighted_model.fit(X_train, y_train)
y_pred_weighted = weighted_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_weighted))
print(f"\nROC-AUC: {roc_auc_score(y_test, weighted_model.predict_proba(X_test)[:, 1]):.3f}")
print(f"PR-AUC: {average_precision_score(y_test, weighted_model.predict_proba(X_test)[:, 1]):.3f}")

# Step 7: Threshold tuning
print("\n=== Model with Threshold Tuning ===")
from sklearn.metrics import precision_recall_curve, f1_score

y_proba = weighted_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Find optimal threshold (maximize F1)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
y_pred_tuned = (y_proba >= optimal_threshold).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned))
print(f"\nF1-Score: {f1_score(y_test, y_pred_tuned):.3f}")

# Step 8: Visualize results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion matrices
models = [
    (y_pred_baseline, "Baseline"),
    (y_pred_smote, "SMOTE"),
    (y_pred_tuned, "Threshold Tuned")
]

for idx, (y_pred, title) in enumerate(models):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'{title}\nF1: {f1_score(y_test, y_pred):.3f}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()
```

## Common Pitfalls

### Pitfall 1: Resampling Before Train-Test Split

**Wrong:**
```python
# Don't do this!
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)
# Data leakage! Test set information leaked into training
```

**Correct:**
```python
# Always split first!
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# Test set remains untouched
```

### Pitfall 2: Using Accuracy as Metric

**Wrong:**
```python
accuracy = model.score(X_test, y_test)  # Misleading for imbalanced data!
```

**Correct:**
```python
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
```

### Pitfall 3: Overfitting to Minority Class

**Problem:** Oversampling can cause overfitting if done incorrectly.

**Solution:** Use cross-validation and monitor both classes:
```python
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    model, X_train_resampled, y_train_resampled,
    cv=StratifiedKFold(5),
    scoring=['precision', 'recall', 'f1', 'roc_auc'],
    return_train_score=True
)
```

### Pitfall 4: Ignoring Business Context

**Problem:** Choosing technique without considering business costs.

**Solution:** Incorporate cost-sensitive learning:
```python
# Define cost matrix
cost_matrix = {
    (0, 0): 0,    # True negative: no cost
    (0, 1): 1,    # False positive: small cost
    (1, 0): 100,  # False negative: high cost (missing fraud)
    (1, 1): 0     # True positive: no cost
}
```

## Additional Resources

### Online Resources

1. **Imbalanced-Learn Documentation** (https://imbalanced-learn.org/) - Comprehensive library for handling imbalanced data
2. **SMOTE Paper** (Chawla et al., 2002) - Original SMOTE algorithm paper
3. **Learning from Imbalanced Data** (He & Garcia, 2009) - Survey paper on techniques

### Books

1. "Learning from Imbalanced Data Sets" by Alberto Fernández et al. - Comprehensive book on the topic
2. "Applied Predictive Modeling" by Max Kuhn - Chapter on class imbalance

### Datasets for Practice

1. **Credit Card Fraud Detection** (Kaggle) - Highly imbalanced fraud dataset
2. **Churn Prediction** - Customer churn datasets
3. **Medical Diagnosis** - Rare disease detection datasets

---

## Key Takeaways

1. **Imbalanced data is common**: Especially in real-world problems (fraud, medical, etc.)
2. **Accuracy is misleading**: Use precision, recall, F1-score, PR-AUC instead
3. **Multiple solutions exist**: Resampling, class weights, threshold tuning, ensemble methods
4. **Choose wisely**: Consider dataset size, imbalance ratio, and business context
5. **Validate properly**: Always use stratified splits and cross-validation
6. **Avoid data leakage**: Never resample before splitting data
7. **Consider costs**: Incorporate business costs into your solution
8. **Combine techniques**: Often best results come from combining multiple approaches

---

**Remember**: Handling imbalanced data is crucial for real-world ML applications! The minority class is often the most important, so don't let high accuracy fool you. Focus on metrics that matter for your business problem.

